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

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_filter.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/config.h>
#include <svo/feature_detection.h>

namespace svo
{

    int Seed::batch_counter = 0;
    int Seed::seed_counter = 0;
    //赋初值，函数体为空
    Seed::Seed(Feature *ftr, float depth_mean, float depth_min) :
            batch_id(batch_counter),
            id(seed_counter++),
            ftr(ftr),
            a(10),
            b(10),
            mu(1.0 / depth_mean),
            z_range(1.0 / depth_min),
            sigma2(z_range * z_range / 36) {}

    DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector, callback_t seed_converged_cb) :
            feature_detector_(feature_detector),
            seed_converged_cb_(seed_converged_cb),
            seeds_updating_halt_(false),
            thread_(NULL),
            new_keyframe_set_(false),
            new_keyframe_min_depth_(0.0),
            new_keyframe_mean_depth_(0.0) {}

    DepthFilter::~DepthFilter()
    {
        stopThread();
        SVO_INFO_STREAM("DepthFilter destructed.");
    }

    void DepthFilter::startThread()
    {
        thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
    }

    void DepthFilter::stopThread()
    {
        SVO_INFO_STREAM("DepthFilter stop thread invoked.");
        if (thread_ != NULL)
        {
            SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
            seeds_updating_halt_ = true;
            thread_->interrupt();
            thread_->join();
            thread_ = NULL;
        }
    }

    void DepthFilter::addFrame(FramePtr frame)
    {
        if (thread_ != NULL)
        {
            {
                lock_t lock(frame_queue_mut_);
                if (frame_queue_.size() > 2)
                    frame_queue_.pop();
                frame_queue_.push(frame);
            }
            seeds_updating_halt_ = false;
            frame_queue_cond_.notify_one();
        } else
            updateSeeds(frame);
    }

//深度滤波器
    void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
    {
        new_keyframe_min_depth_ = depth_min;
        new_keyframe_mean_depth_ = depth_mean;
        //判断线程是否在跑
        if (thread_ != NULL)
        {
            //若线程在跑，那么向深度滤波器中加入新的帧，并设置新帧标志位与种子点更新挂起标志位，同时发出帧队列调整信号。
            new_keyframe_ = frame;
            new_keyframe_set_ = true;
            seeds_updating_halt_ = true;
            frame_queue_cond_.notify_one();
        } else
            initializeSeeds(frame);//若线程没在跑，那么初始化种子点。
    }
    //初始化种子点
    void DepthFilter::initializeSeeds(FramePtr frame)
    {
        //定义一系列跟踪点
        Features new_features;
        //根据起始点，设置跟踪网格占用
        feature_detector_->setExistingFeatures(frame->fts_);
        //进行特征点检测，这里使用的是FAST特征
        feature_detector_->detect(frame.get(), frame->img_pyr_,
                                  Config::triangMinCornerScore(), new_features);

        // initialize a seed for every new feature
        seeds_updating_halt_ = true;
        //种子点跟新挂起
        lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
        //线程锁
        ++Seed::batch_counter;

        std::for_each(new_features.begin(), new_features.end(), [&](Feature *ftr) {
            seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
        });

        if (options_.verbose)
            SVO_INFO_STREAM("DepthFilter: Initialized " << new_features.size() << " new seeds");
        //解除种子点跟新挂起
        seeds_updating_halt_ = false;
    }

    void DepthFilter::removeKeyframe(FramePtr frame)
    {
        seeds_updating_halt_ = true;
        lock_t lock(seeds_mut_);
        std::list<Seed>::iterator it = seeds_.begin();
        size_t n_removed = 0;
        while (it != seeds_.end())
        {
            if (it->ftr->frame == frame.get())
            {
                it = seeds_.erase(it);
                ++n_removed;
            } else
                ++it;
        }
        seeds_updating_halt_ = false;
    }

    void DepthFilter::reset()
    {
        seeds_updating_halt_ = true;
        {
            lock_t lock(seeds_mut_);
            seeds_.clear();
        }
        lock_t lock();
        while (!frame_queue_.empty())
            frame_queue_.pop();
        seeds_updating_halt_ = false;

        if (options_.verbose)
            SVO_INFO_STREAM("DepthFilter: RESET.");
    }
    //深度滤波器核心线程
    void DepthFilter::updateSeedsLoop()
    {   //判断是否发生终端请求
        while (!boost::this_thread::interruption_requested())
        {
            FramePtr frame;
            {
                //设置帧队列锁
                lock_t lock(frame_queue_mut_);
                //判断帧队列是不是空，是不是有新的帧进来，是不是出发触发了帧队列调整通知。在接到通知后，拿回lock的所有权
                while (frame_queue_.empty() && new_keyframe_set_ == false)
                    frame_queue_cond_.wait(lock);
                if (new_keyframe_set_)//有新的关键帧插入进来，通过这里可以看出，真的是判断当前帧是不是关键帧。
                {
                    //如果是关键帧，那么恢复new_keyframe_set_和seeds_updating_halt_，同时清空帧队列
                    new_keyframe_set_ = false;
                    seeds_updating_halt_ = false;
                    clearFrameQueue();
                    frame = new_keyframe_;
                } else//没有新的关键帧，但是有新的普通帧
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }
            updateSeeds(frame);//可能有两种情况进这里，一种是新的关键帧，一种是新的普通帧
            if (frame->isKeyframe())
                initializeSeeds(frame);
        }
    }
    //深度滤波器的核心函数，这里需要好好理解！！！
    void DepthFilter::updateSeeds(FramePtr frame)//可能有两种情况进这里，一种是新的关键帧，一种是新的普通帧
    {
        // update only a limited number of seeds, because we don't have time to do it
        // for all the seeds in every frame!
        //由于时间的问题，我们在每帧图像中只更新一些有限的点。
        size_t n_updates = 0, n_failed_matches = 0, n_seeds = seeds_.size();
        //线程加锁
        lock_t lock(seeds_mut_);

        std::list<Seed>::iterator it = seeds_.begin();

        const double focal_length = frame->cam_->errorMultiplier2();//这里用的是fx
        double px_noise = 1.0;
        double px_error_angle = atan(px_noise / (2.0 * focal_length)) * 2.0; // law of chord (sehnensatz)//弦长公式

        while (it != seeds_.end())//遍历所有的种子点
        {
            // set this value true when seeds updating should be interrupted
            if (seeds_updating_halt_)//判断种子点的更新是否被挂起了，如果挂起了则停止更新
                return;

            // check if seed is not already too old
            if ((Seed::batch_counter - it->batch_id) >
                options_.max_n_kfs)
            {//通过与当前种子的帧数与采集时的帧数来判断是否该被删除-----有问题，会丢掉原地徘徊的种子点，极远点也会丢掉。
                it = seeds_.erase(it);
                continue;
            }

            // check if point is visible in the current image
            //获得当前帧到参考帧的变换矩阵。
            SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
            //
            const Vector3d xyz_f(T_ref_cur.inverse() * (1.0 / it->mu * it->ftr->f));//这个公式需要整理
            if (xyz_f.z() < 0.0)
            {
                ++it; // behind the camera
                continue;
            }
            //判断摄像机坐标系下的点在图像坐标系下是否在帧的有效区域中
            if (!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>()))
            {
                ++it; // point does not project in image
                continue;
            }

            // we are using inverse depth coordinates
            // 这里使用逆深度的坐标，公式需要整理
            float z_inv_min = it->mu + sqrt(it->sigma2);
            float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);
            double z;
            //在极线上进行搜索，找到最大值
            if (!matcher_.findEpipolarMatchDirect(
                    *it->ftr->frame, *frame, *it->ftr, 1.0 / it->mu, 1.0 / z_inv_min, 1.0 / z_inv_max, z))//在极线上找深度
            {
                //如果没找到的话，那么认为这个点是外点的概率提高。
                it->b++; // increase outlier probability when no match was found
                ++it;
                ++n_failed_matches;
                continue;
            }
            //如果找到了，那么开始自己算不确定度
            // compute tau
            double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);//tau是测量不确定性
            // 因为采用了逆深度，那么不确定度也需要用逆表示。
            double tau_inverse = 0.5 * (1.0 / max(0.0000001, z - tau) - 1.0 / (z + tau));
            //更新种子点估计
            // update the estimate
            updateSeed(1. / z, tau_inverse * tau_inverse, &*it);
            ++n_updates;
            //判断是否是关键帧，如果是则对网格进行调整
            if (frame->isKeyframe())
            {
                // The feature detector should not initialize new seeds close to this location
                feature_detector_->setGridOccpuancy(matcher_.px_cur_);
            }
            //如果种子点收敛了，那么我们初始化一个备选点，并删除种子点。
            //收敛判断条件：标准差小于深度范围/协方差阈值
            // if the seed has converged, we initialize a new candidate point and remove the seed
            if (sqrt(it->sigma2) < it->z_range / options_.seed_convergence_sigma2_thresh)
            {
                assert(it->ftr->point == NULL); // TODO this should not happen anymore

                Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0 / it->mu)));
                Point *point = new Point(xyz_world, it->ftr);
                it->ftr->point = point;
                /* FIXME it is not threadsafe to add a feature to the frame here.
                if(frame->isKeyframe())
                {
                  Feature* ftr = new Feature(frame.get(), matcher_.px_cur_, matcher_.search_level_);
                  ftr->point = point;
                  point->addFrameRef(ftr);
                  frame->addFeature(ftr);
                  it->ftr->frame->addFeature(it->ftr);
                }
                else
                */
                {
                    //收敛回调函数
                    seed_converged_cb_(point, it->sigma2); // put in candidate list
                }
                it = seeds_.erase(it);
            } else if (isnan(z_inv_min))//如果是无穷大，那么删除该种子点
            {
                SVO_WARN_STREAM("z_min is NaN");
                it = seeds_.erase(it);
            } else
                ++it;
        }
    }

    void DepthFilter::clearFrameQueue()
    {
        while (!frame_queue_.empty())
            frame_queue_.pop();
    }

    void DepthFilter::getSeedsCopy(const FramePtr &frame, std::list<Seed> &seeds)
    {
        lock_t lock(seeds_mut_);
        for (std::list<Seed>::iterator it = seeds_.begin(); it != seeds_.end(); ++it)
        {
            if (it->ftr->frame == frame.get())
                seeds.push_back(*it);
        }
    }
    //种子点更新的核心公式，需要深入理解
    void DepthFilter::updateSeed(const float x, const float tau2, Seed *seed)
    {
        float norm_scale = sqrt(seed->sigma2 + tau2);
        if (std::isnan(norm_scale))
            return;
        boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
        float s2 = 1. / (1. / seed->sigma2 + 1. / tau2);
        float m = s2 * (seed->mu / seed->sigma2 + x / tau2);
        float C1 = seed->a / (seed->a + seed->b) * boost::math::pdf(nd, x);
        float C2 = seed->b / (seed->a + seed->b) * 1. / seed->z_range;
        float normalization_constant = C1 + C2;
        C1 /= normalization_constant;
        C2 /= normalization_constant;
        float f = C1 * (seed->a + 1.) / (seed->a + seed->b + 1.) + C2 * seed->a / (seed->a + seed->b + 1.);
        float e = C1 * (seed->a + 1.) * (seed->a + 2.) / ((seed->a + seed->b + 1.) * (seed->a + seed->b + 2.))
                  + C2 * seed->a * (seed->a + 1.0f) / ((seed->a + seed->b + 1.0f) * (seed->a + seed->b + 2.0f));

        // update parameters
        float mu_new = C1 * m + C2 * seed->mu;
        seed->sigma2 = C1 * (s2 + m * m) + C2 * (seed->sigma2 + seed->mu * seed->mu) - mu_new * mu_new;
        seed->mu = mu_new;
        seed->a = (e - f) / (f - e / f);
        seed->b = seed->a * (1.0f - f) / f;
    }

    double DepthFilter::computeTau(
            const SE3 &T_ref_cur,
            const Vector3d &f,
            const double z,
            const double px_error_angle)
    {
        Vector3d t(T_ref_cur.translation());
        Vector3d a = f * z - t;
        double t_norm = t.norm();
        double a_norm = a.norm();
        double alpha = acos(f.dot(t) / t_norm); // dot product
        double beta = acos(a.dot(-t) / (t_norm * a_norm)); // dot product
        double beta_plus = beta + px_error_angle;
        double gamma_plus = PI - alpha - beta_plus; // triangle angles sum to PI
        double z_plus = t_norm * sin(beta_plus) / sin(gamma_plus); // law of sines
        return (z_plus - z); // tau
    }

} // namespace svo
