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
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo
{
    namespace initialization
    {
/*
 * 首先清空当前的特征点和参考帧
 * 然后进行FAST特征点检测，
 * 如果特征点的数量少于100，则判断关键帧输入失败，返回FAILURE。
 * 否则，设置保存参考帧为当前帧（当前帧为关键帧）
 * 并且将检测到的特征点添加进px_cur_中。
 * 最后返回SUCCESS
 * */
        InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
        {
            reset();
            detectFeatures(frame_ref, px_ref_, f_ref_);
            if (px_ref_.size() < 100) {
                SVO_WARN_STREAM_THROTTLE(2.0,
                                         "First image has less than 100 features. Retry in more textured environment.");
                return FAILURE;
            }
            frame_ref_ = frame_ref;
            px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
            return SUCCESS;
        }

        InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
        {
            //开始进行KLT跟踪，求从参考帧到当前帧的光流信息，其中还包含了视察信息。
            trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
            SVO_INFO_STREAM("Init: KLT tracked " << disparities_.size() << " features");
            //判断成功跟踪点的数量是否大于预设的最小跟踪点数量
            //此处的disparities_.size()应该和px_ref_.size()以及px_cur_.size()相等
            if (disparities_.size() < Config::initMinTracked())
                return FAILURE;
            //获取视差的中位数，并判断其与预设最小视差的关系，若小于最小视差，则确定当前帧不是关键帧，并返回
            double disparity = vk::getMedian(disparities_);
            SVO_INFO_STREAM("Init: KLT " << disparity << "px average disparity.");
            if (disparity < Config::initMinDisparity())
                return NO_KEYFRAME;
            //计算单应矩阵，此处使用的是PTAM的单应矩阵计算法模块,在完成单应矩阵的计算之后，直接分解得到T(R,t)
            computeHomography(
                    f_ref_, f_cur_,
                    frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
                    inliers_, xyz_in_cur_, T_cur_from_ref_);
            SVO_INFO_STREAM("Init: Homography RANSAC " << inliers_.size() << " inliers.");
            //判断匹配之后的内点数量，若内点数量少于预设的最小内点数，则返回失败
            if (inliers_.size() < Config::initMinInliers()) {
                SVO_WARN_STREAM("Init WARNING: " << Config::initMinInliers() << " inliers minimum required.");
                return FAILURE;
            }

            // Rescale the map such that the mean scene depth is equal to the specified scale
            //重置地图尺度，使场景尺度等于给定值。
            vector<double> depth_vec;
            //统计场景深度
            for (size_t i = 0; i < xyz_in_cur_.size(); ++i)
                depth_vec.push_back((xyz_in_cur_[i]).z());
            //取得场景深度的均值
            double scene_depth_median = vk::getMedian(depth_vec);
            //计算尺度值，其值等于地图尺度/场景深度
            double scale = Config::mapScale() / scene_depth_median;
            //计算当前帧相对于初始位置的转换矩阵
            frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
            //公式1
            frame_cur->T_f_w_.translation() =
                    -frame_cur->T_f_w_.rotation_matrix() *
                    (frame_ref_->pos() + scale * (frame_cur->pos() - frame_ref_->pos()));

            // For each inlier create 3D point and add feature in both frames
            // 为每个内点建立3D点，并在生成该3D点的两帧图像中加入特征
            // 取转置，获得当前坐标系到世界坐标系的转换
            SE3 T_world_cur = frame_cur->T_f_w_.inverse();
            //
            for (vector<int>::iterator it = inliers_.begin(); it != inliers_.end(); ++it) {
                Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
                Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
                // 判断当前帧的点和参考帧的点是否在当前帧的有效范围内，这里是边距10的范围之内
                // 同时，要判断当前点的3D点在摄像机之前
                if (frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) &&
                    frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0) {
                    //将当前3D点进行缩放，在转换到世界坐标系下，从而得到缩放后的世界坐标系
                    Vector3d pos = T_world_cur * (xyz_in_cur_[*it] * scale);
                    Point *new_point = new Point(pos);
                    //在当前帧上建立一个跟踪点
                    Feature *ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
                    frame_cur->addFeature(ftr_cur);
                    //添加该点的可见帧，就是说ftr_cur上能够看见这个点
                    new_point->addFrameRef(ftr_cur);
                    //在参考帧上建立一个跟踪特征
                    Feature *ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
                    frame_ref_->addFeature(ftr_ref);
                    //添加该点的可见帧，就是说frame_ref_上能够看见这个点
                    new_point->addFrameRef(ftr_ref);
                }
            }
            return SUCCESS;
        }

        void KltHomographyInit::reset()
        {
            px_cur_.clear();
            frame_ref_.reset();
        }

        void detectFeatures(
                FramePtr frame,
                vector<cv::Point2f> &px_vec,
                vector<Vector3d> &f_vec)
        {
            Features new_features;
            feature_detection::FastDetector detector(
                    frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
            detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

            // now for all maximum corners, initialize a new seed
            px_vec.clear();
            px_vec.reserve(new_features.size());
            f_vec.clear();
            f_vec.reserve(new_features.size());
            std::for_each(new_features.begin(), new_features.end(), [&](Feature *ftr) {
                px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
                f_vec.push_back(ftr->f);
                delete ftr;
            });
        }
        //KLT光流跟踪算法，其中最大迭代次数为30，收敛判定为0.001，klt跟踪框尺寸为30
        void trackKlt(
                FramePtr frame_ref,
                FramePtr frame_cur,
                vector<cv::Point2f> &px_ref,
                vector<cv::Point2f> &px_cur,
                vector<Vector3d> &f_ref,
                vector<Vector3d> &f_cur,
                vector<double> &disparities)
        {
            const double klt_win_size = 30.0;
            const int klt_max_iter = 30;
            const double klt_eps = 0.001;
            vector<uchar> status;
            vector<float> error;
            //这个向量貌似没有用
            //vector<float> min_eig_vec;
            //klt停止条件
            cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, klt_max_iter, klt_eps);
            //开始klt跟踪
            cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                                     px_ref, px_cur,
                                     status, error,
                                     cv::Size2i(klt_win_size, klt_win_size),
                                     4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);
            //建立参考帧点与当前帧点的迭代器
            vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
            vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
            //f_ref和f_cur存储的是跟踪点的摄像机坐标系下的三维归一化坐标
            vector<Vector3d>::iterator f_ref_it = f_ref.begin();
            f_cur.clear();
            f_cur.reserve(px_cur.size());
            //清空视差存储向量，并从新分配其默认长度
            disparities.clear();
            disparities.reserve(px_cur.size());
            for (size_t i = 0; px_ref_it != px_ref.end(); ++i) {
                //判断跟踪情况，若跟踪失败，则删除该跟踪点
                if (!status[i]) {
                    px_ref_it = px_ref.erase(px_ref_it);
                    px_cur_it = px_cur.erase(px_cur_it);
                    f_ref_it = f_ref.erase(f_ref_it);
                    continue;
                }
                //计算三维归一化坐标（相当于进行了一次非齐次坐标到其次坐标的转换）
                f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
                //计算视差
                disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
                ++px_ref_it;
                ++px_cur_it;
                ++f_ref_it;
            }
        }

        void computeHomography(
                const vector<Vector3d> &f_ref,
                const vector<Vector3d> &f_cur,
                double focal_length,
                double reprojection_threshold,
                vector<int> &inliers,
                vector<Vector3d> &xyz_in_cur,
                SE3 &T_cur_from_ref)
        {
            vector<Vector2d> uv_ref(f_ref.size());
            vector<Vector2d> uv_cur(f_cur.size());
            for (size_t i = 0, i_max = f_ref.size(); i < i_max; ++i) {
                uv_ref[i] = vk::project2d(f_ref[i]);
                uv_cur[i] = vk::project2d(f_cur[i]);
            }
            vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
            Homography.computeSE3fromMatches();
            vector<int> outliers;
            vk::computeInliers(f_cur, f_ref,
                               Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                               reprojection_threshold, focal_length,
                               xyz_in_cur, inliers, outliers);
            T_cur_from_ref = Homography.T_c2_from_c1;
        }


    } // namespace initialization
} // namespace svo
