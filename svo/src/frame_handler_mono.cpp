// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>

#ifdef USE_BUNDLE_ADJUSTMENT

#include <svo/bundle_adjustment.h>

#endif

namespace svo
{

    FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera *cam) :
            FrameHandlerBase(),
            cam_(cam),
            reprojector_(cam_, map_),
            depth_filter_(NULL)
    {
        initialize();
    }
    //单目相机帧句柄初始化
    void FrameHandlerMono::initialize()
    {
        //定义跟踪特征检测器，这里使用FAST特征点
        feature_detection::DetectorPtr feature_detector(
                new feature_detection::FastDetector(
                        cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
        //定义深度滤波器的回调函数。
        DepthFilter::callback_t depth_filter_cb = boost::bind(
                &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
        //初始化深度滤波器，初始化参数包括特征检测器以及深度滤波器回调函数。
        depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
        //深度滤波器线程启动。
        depth_filter_->startThread();
    }

    FrameHandlerMono::~FrameHandlerMono()
    {
        delete depth_filter_;
    }

    void FrameHandlerMono::addImage(const cv::Mat &img, const double timestamp)
    {
/*
 * 判断svo的当前状态，当前状态有，STAGE_PAUSED,STAGE_FIRST_FRAME,STAGE_SECOND_FRAME,STAGE_DEFAULT_FRAME,STAGE_RELOCALIZING，一共五种。
 * 此函数中，只判断是否为STAGE_PAUSED，若是则返回false。
 * 同时，此函数中会判断是否重新开始，若是则将stage_ = STAGE_FIRST_FRAME。
 * 最后会调用map_.emptyTrash();
 * */
        if (!startFrameProcessingCommon(timestamp))
            return;

        // some cleanup from last iteration, can't do before because of visualization
        // 清空关键帧，这些关键点是临近的关键帧
        core_kfs_.clear();
        // 含有重叠视野的所有关键帧，其中的每个元素是一个pair结构，pair中分别为关键帧与关键帧中所看到的地图点
        overlap_kfs_.clear();

        // create new frame
        SVO_START_TIMER("pyramid_creation");
        //重建一个新的图像帧，这里的图像帧中使用了图像金字塔，其中金字塔的使用的是缩放比例为0.5的缩小金字塔，即金字塔中的所有图像均小于等于输入图像
        new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
        SVO_STOP_TIMER("pyramid_creation");

        // process frame
        //开始处理图像帧。
        UpdateResult res = RESULT_FAILURE;
/* 根据当前状态判断使用的函数，首帧使用processFirstFrame函数，
 * 第二帧使用processSecondFrame函数，
 * 从第三帧开始为一般帧，使用processFrame函数，
 * 如果当前状态为STAGE_RELOCALIZING，则开始进行中定位检测。
 * */
        if (stage_ == STAGE_DEFAULT_FRAME)
            res = processFrame();
        else if (stage_ == STAGE_SECOND_FRAME)
            res = processSecondFrame();
        else if (stage_ == STAGE_FIRST_FRAME)
            res = processFirstFrame();
        else if (stage_ == STAGE_RELOCALIZING)
            res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                                  map_.getClosestKeyframe(last_frame_));

        // set last frame
        // 存储上一帧。
        last_frame_ = new_frame_;
        new_frame_.reset();
        // finish processing
        finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
    }

    FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
    {
        // 初始化第一帧，首先初始化新帧的变换矩阵，该矩阵用SE3表示。
        new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
        // 向单应矩阵求解器中添加新的帧，判断是否为关键帧。
        if (klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
            return RESULT_NO_KEYFRAME;
        //猜测为将当前帧设置为关键帧，并将关键帧插入地图系统中
        new_frame_->setKeyframe();
        map_.addKeyframe(new_frame_);
        //第一阶段结束，设置stage_ = STAGE_SECOND_FRAME并返回RESULT_IS_KEYFRAME
        stage_ = STAGE_SECOND_FRAME;
        SVO_INFO_STREAM("Init: Selected first frame.");
        return RESULT_IS_KEYFRAME;
    }

    FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
    {
        //处理第二帧图像
        //首先将第二帧图像输入KLT跟踪器中，开始跟踪，然后判断跟踪结果，若初始化失败，则返回RESULT_FAILURE，若没有关键帧，则返回RESULT_NO_KEYFRAME
        //计算第二帧图像相对于第一帧图像的位姿变化，并存储地图点（疑问，此处没有找到在哪里进行的三角化）
        initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);

        if (res == initialization::FAILURE)
            return RESULT_FAILURE;
        else if (res == initialization::NO_KEYFRAME)
            return RESULT_NO_KEYFRAME;
        //根据宏定义来判断是否进行两帧的BUNDLE_ADJUSTMENT
        // two-frame bundle adjustment
        //TBD!
#ifdef USE_BUNDLE_ADJUSTMENT
        ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif
        //将当前帧设置为关键帧，然后取得当前场景的深度平均值和深度最小值

        new_frame_->setKeyframe();
        double depth_mean, depth_min;
        frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
        //将当前帧作为关键帧插入深度滤波器，使深度滤波器的平均深度为当前的平均深度，使深度滤波器的最小深度为当前最小深度的一半

        depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5 * depth_min);

        //将当前帧插入地图系统中

        // add frame to map
        map_.addKeyframe(new_frame_);//这里只是将新的帧加入到地图中，应该有另一个线程在进行地图生成
        stage_ = STAGE_DEFAULT_FRAME;
        klt_homography_init_.reset();
        SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
        return RESULT_IS_KEYFRAME;
    }

    FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
    {
        // Set initial pose TODO use prior
        new_frame_->T_f_w_ = last_frame_->T_f_w_;

        // sparse image align
        SVO_START_TIMER("sparse_img_align");
        //稀疏图像对齐？
        SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                                 30, SparseImgAlign::GaussNewton, false, false);
        size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
        SVO_STOP_TIMER("sparse_img_align");
        SVO_LOG(img_align_n_tracked);
        SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

        // map reprojection & feature alignment
        //地图重映射和特征对齐
        SVO_START_TIMER("reproject");
        reprojector_.reprojectMap(new_frame_, overlap_kfs_);
        SVO_STOP_TIMER("reproject");
        const size_t repr_n_new_references = reprojector_.n_matches_;
        const size_t repr_n_mps = reprojector_.n_trials_;
        SVO_LOG2(repr_n_mps, repr_n_new_references);
        SVO_DEBUG_STREAM("Reprojection:\t nPoints = " << repr_n_mps << "\t \t nMatches = " << repr_n_new_references);
        if (repr_n_new_references < Config::qualityMinFts()) {
            SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
            new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
            tracking_quality_ = TRACKING_INSUFFICIENT;
            return RESULT_FAILURE;
        }

        // pose optimization
        SVO_START_TIMER("pose_optimizer");
        size_t sfba_n_edges_final;
        double sfba_thresh, sfba_error_init, sfba_error_final;
        pose_optimizer::optimizeGaussNewton(
                Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
                new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
        SVO_STOP_TIMER("pose_optimizer");
        SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
        SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = " << sfba_error_init << "px\t thresh = " << sfba_thresh);
        SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = " << sfba_error_final << "px\t nObsFin. = " << sfba_n_edges_final);
        if (sfba_n_edges_final < 20)
            return RESULT_FAILURE;

        // structure optimization
        SVO_START_TIMER("point_optimizer");
        optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
        SVO_STOP_TIMER("point_optimizer");

        // select keyframe
        core_kfs_.insert(new_frame_);
        setTrackingQuality(sfba_n_edges_final);
        if (tracking_quality_ == TRACKING_INSUFFICIENT) {
            new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
            return RESULT_FAILURE;
        }
        double depth_mean, depth_min;
        frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
        if (!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD) {
            depth_filter_->addFrame(new_frame_);
            return RESULT_NO_KEYFRAME;
        }
        new_frame_->setKeyframe();
        SVO_DEBUG_STREAM("New keyframe selected.");

        // new keyframe selected
        for (Features::iterator it = new_frame_->fts_.begin(); it != new_frame_->fts_.end(); ++it)
            if ((*it)->point != NULL)
                (*it)->point->addFrameRef(*it);
        map_.point_candidates_.addCandidatePointToFrame(new_frame_);

        // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
        if (Config::lobaNumIter() > 0) {
            SVO_START_TIMER("local_ba");
            setCoreKfs(Config::coreNKfs());
            size_t loba_n_erredges_init, loba_n_erredges_fin;
            double loba_err_init, loba_err_fin;
            ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                        loba_n_erredges_init, loba_n_erredges_fin,
                        loba_err_init, loba_err_fin);
            SVO_STOP_TIMER("local_ba");
            SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
            SVO_DEBUG_STREAM(
                    "Local BA:\t RemovedEdges {" << loba_n_erredges_init << ", " << loba_n_erredges_fin << "} \t "
                                                                                                           "Error {"
                                                 << loba_err_init << ", " << loba_err_fin << "}");
        }
#endif

        // init new depth-filters
        depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5 * depth_min);

        // if limited number of keyframes, remove the one furthest apart
        if (Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs()) {
            FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
            depth_filter_->removeKeyframe(
                    furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
            map_.safeDeleteFrame(furthest_frame);
        }

        // add keyframe to map
        map_.addKeyframe(new_frame_);

        return RESULT_IS_KEYFRAME;
    }

    FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
            const SE3 &T_cur_ref,
            FramePtr ref_keyframe)
    {
        SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
        if (ref_keyframe == nullptr) {
            SVO_INFO_STREAM("No reference keyframe.");
            return RESULT_FAILURE;
        }
        SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                                 30, SparseImgAlign::GaussNewton, false, false);
        size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
        if (img_align_n_tracked > 30) {
            SE3 T_f_w_last = last_frame_->T_f_w_;
            last_frame_ = ref_keyframe;
            FrameHandlerMono::UpdateResult res = processFrame();
            if (res != RESULT_FAILURE) {
                stage_ = STAGE_DEFAULT_FRAME;
                SVO_INFO_STREAM("Relocalization successful.");
            } else
                new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
            return res;
        }
        return RESULT_FAILURE;
    }

    bool FrameHandlerMono::relocalizeFrameAtPose(
            const int keyframe_id,
            const SE3 &T_f_kf,
            const cv::Mat &img,
            const double timestamp)
    {
        FramePtr ref_keyframe;
        if (!map_.getKeyframeById(keyframe_id, ref_keyframe))
            return false;
        new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
        UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
        if (res != RESULT_FAILURE) {
            last_frame_ = new_frame_;
            return true;
        }
        return false;
    }

    void FrameHandlerMono::resetAll()
    {
        resetCommon();
        last_frame_.reset();
        new_frame_.reset();
        core_kfs_.clear();
        overlap_kfs_.clear();
        depth_filter_->reset();
    }

    void FrameHandlerMono::setFirstFrame(const FramePtr &first_frame)
    {
        resetAll();
        last_frame_ = first_frame;
        last_frame_->setKeyframe();
        map_.addKeyframe(last_frame_);
        stage_ = STAGE_DEFAULT_FRAME;
    }

    bool FrameHandlerMono::needNewKf(double scene_depth_mean)
    {
        for (auto it = overlap_kfs_.begin(), ite = overlap_kfs_.end(); it != ite; ++it) {
            Vector3d relpos = new_frame_->w2f(it->first->pos());
            if (fabs(relpos.x()) / scene_depth_mean < Config::kfSelectMinDist() &&
                fabs(relpos.y()) / scene_depth_mean < Config::kfSelectMinDist() * 0.8 &&
                fabs(relpos.z()) / scene_depth_mean < Config::kfSelectMinDist() * 1.3)
                return false;
        }
        return true;
    }

    void FrameHandlerMono::setCoreKfs(size_t n_closest)
    {
        size_t n = min(n_closest, overlap_kfs_.size() - 1);
        std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin() + n, overlap_kfs_.end(),
                          boost::bind(&pair<FramePtr, size_t>::second, _1) >
                          boost::bind(&pair<FramePtr, size_t>::second, _2));
        std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(),
                      [&](pair<FramePtr, size_t> &i) { core_kfs_.insert(i.first); });
    }

} // namespace svo
