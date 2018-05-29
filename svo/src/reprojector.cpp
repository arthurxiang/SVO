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
#include <stdexcept>
#include <svo/reprojector.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/map.h>
#include <svo/config.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <vikit/abstract_camera.h>
#include <vikit/math_utils.h>
#include <vikit/timer.h>

namespace svo
{
//给map_赋初值，并初始化AbstractCamera的网格
    Reprojector::Reprojector(vk::AbstractCamera *cam, Map &map) :
            map_(map)
    {
        //初始化网格，并写入值
        initializeGrid(cam);
    }

    Reprojector::~Reprojector()
    {
        std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *c) { delete c; });
    }

/* 网格划分，将图像按照每个格cell_size大小，划分为rows*cols的网格
 * 每个网格里面放入当前网格的编号，最后随机洗牌，打乱网格内各个cell的顺序
 * */

    void Reprojector::initializeGrid(vk::AbstractCamera *cam)
    {
        grid_.cell_size = Config::gridSize();
        grid_.grid_n_cols = ceil(static_cast<double>(cam->width()) / grid_.cell_size);
        grid_.grid_n_rows = ceil(static_cast<double>(cam->height()) / grid_.cell_size);
        grid_.cells.resize(grid_.grid_n_cols * grid_.grid_n_rows);
        std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *&c) { c = new Cell; });
        grid_.cell_order.resize(grid_.cells.size());
        for (size_t i = 0; i < grid_.cells.size(); ++i)
            grid_.cell_order[i] = i;
        random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); // maybe we should do it at every iteration!
    }

    void Reprojector::resetGrid()
    {
        n_matches_ = 0;
        n_trials_ = 0;
        std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *c) { c->clear(); });
    }

    void Reprojector::reprojectMap(
            FramePtr frame,
            std::vector<std::pair<FramePtr, std::size_t> > &overlap_kfs)
    {
        //清空网格及匹配计数器和验证计数器的内容
        resetGrid();

        // Identify those Keyframes which share a common field of view.
        SVO_START_TIMER("reproject_kfs");
        list<pair<FramePtr, double> > close_kfs;
        //取得与frame有共同视野范围的相邻关键帧。
        map_.getCloseKeyframes(frame, close_kfs);

        // Sort KFs with overlap according to their closeness
        // 对相邻关键帧进行排序，离得近的往前排。
        close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                       boost::bind(&std::pair<FramePtr, double>::second, _2));

        // Reproject all mappoints of the closest N kfs with overlap. We only store
        // in which grid cell the points fall.
        // 重映射最近的N帧关键帧的重叠的地图点，在存储的时候，我们只记录这些点落入到了网格的哪格子里面。
        size_t n = 0;//关键帧计数器
        overlap_kfs.reserve(options_.max_n_kfs);
        //遍历所有的关键帧。
        for (auto it_frame = close_kfs.begin(), ite_frame = close_kfs.end();
             it_frame != ite_frame && n < options_.max_n_kfs; ++it_frame, ++n)
        {

            FramePtr ref_frame = it_frame->first;//first存的是FramePtr，second存的是帧中的点数量。
            overlap_kfs.push_back(pair<FramePtr, size_t>(ref_frame, 0));

            // Try to reproject each mappoint that the other KF observes
            // 尝试对其他关键帧中观测到的地图点进行重投影
            // 遍历所有的feature(跟踪点)
            for (auto it_ftr = ref_frame->fts_.begin(), ite_ftr = ref_frame->fts_.end();
                 it_ftr != ite_ftr; ++it_ftr)
            {
                // 查看feature是否有个一个地图点与之对应
                // check if the feature has a mappoint assigned
                if ((*it_ftr)->point == NULL)
                    continue;

                // make sure we project a point only once
                // 确保每个feature之映射1次，？？？？？我觉得不会发生两次吧
                if ((*it_ftr)->point->last_projected_kf_id_ == frame->id_)
                    continue;
                (*it_ftr)->point->last_projected_kf_id_ = frame->id_;
                //对该点进行重映射
                if (reprojectPoint(frame, (*it_ftr)->point))
                    overlap_kfs.back().second++;
            }
        }
        SVO_STOP_TIMER("reproject_kfs");

        // Now project all point candidates
        // 对所有的候选人点进行重映射
        SVO_START_TIMER("reproject_candidates");
        {
            //线程保护，抢断map线程
            boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
            //开始对map中的候选点进行遍历
            auto it = map_.point_candidates_.candidates_.begin();
            while (it != map_.point_candidates_.candidates_.end())
            {
                //对候选点进行重映射
                if (!reprojectPoint(frame, it->first))
                {
                    //如果映射失败
                    //如果失败则失败映射计数器+3，为啥是+3？
                    it->first->n_failed_reproj_ += 3;
                    if (it->first->n_failed_reproj_ > 30)
                    {
                        //如果失败计数器大于30则删除该候选点。
                        map_.point_candidates_.deleteCandidate(*it);//这里仅仅是清空指针内容，但是保留指针及其在list中的内存空间
                        it = map_.point_candidates_.candidates_.erase(it);//清除内存空间，并后移指针。
                        continue;
                    }
                }
                ++it;
            }
        } // unlock the mutex when out of scope
        SVO_STOP_TIMER("reproject_candidates");

        // Now we go through each grid cell and select one point to match.
        // At the end, we should have at maximum one reprojected point per cell.
        SVO_START_TIMER("feature_align");
        for (size_t i = 0; i < grid_.cells.size(); ++i)
        {
            // we prefer good quality points over unkown quality (more likely to match)
            // and unknown quality over candidates (position not optimized)
            // 首选good点，次选unkown点，最后选candidate点
            // 对所有cell进行重映射，如果成功那么++n_matches_
            if (reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
                ++n_matches_;
            if (n_matches_ > (size_t) Config::maxFts())
                break;
        }
        SVO_STOP_TIMER("feature_align");
    }

    bool Reprojector::pointQualityComparator(Candidate &lhs, Candidate &rhs)
    {
        if (lhs.pt->type_ > rhs.pt->type_)
            return true;
        return false;
    }

    bool Reprojector::reprojectCell(Cell &cell, FramePtr frame)
    {
        //按照点质量进行对比，并进行排序
        cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
        //遍历所有cell
        Cell::iterator it = cell.begin();
        while (it != cell.end())
        {

            ++n_trials_;//在resetGrid中将n_trials_=0
            //如果点的状态为已经删除，则删除该点，不直接删除的目的是为了线程安全
            if (it->pt->type_ == Point::TYPE_DELETED)
            {
                it = cell.erase(it);
                continue;
            }

            bool found_match = true;
            if (options_.find_match_direct)
                found_match = matcher_.findMatchDirect(*it->pt, *frame, it->px);
            //如果直接法没找到，那么判断失败次数，对于UNKNOWN点来说，大于15删除，对于CANDIDATE，大于30则删除。
            if (!found_match)
            {
                it->pt->n_failed_reproj_++;
                if (it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
                    map_.safeDeletePoint(it->pt);
                if (it->pt->type_ == Point::TYPE_CANDIDATE && it->pt->n_failed_reproj_ > 30)
                    map_.point_candidates_.deleteCandidatePoint(it->pt);
                it = cell.erase(it);
                continue;
            }
            //如果找到了，那么修改点类型，如果重映射次数大于10次则重TYPE_UNKNOWN调整至TYPE_GOOD
            it->pt->n_succeeded_reproj_++;
            if (it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
                it->pt->type_ = Point::TYPE_GOOD;

            Feature *new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
            frame->addFeature(new_feature);

            // Here we add a reference in the feature to the 3D point, the other way
            // round is only done if this frame is selected as keyframe.
            new_feature->point = it->pt;

            if (matcher_.ref_ftr_->type == Feature::EDGELET)
            {
                new_feature->type = Feature::EDGELET;
                new_feature->grad = matcher_.A_cur_ref_ * matcher_.ref_ftr_->grad;
                new_feature->grad.normalize();
            }

            // If the keyframe is selected and we reproject the rest, we don't have to
            // check this point anymore.
            it = cell.erase(it);

            // Maximum one point per cell.
            return true;
        }
        return false;
    }

    bool Reprojector::reprojectPoint(FramePtr frame, Point *point)
    {
        Vector2d px(frame->w2c(point->pos_));//摄像机坐标系转换为图像坐标系
        //判断是否在图像的有效区域中，边距8
        if (frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
        {
            //统计投影到第几个格子里面
            const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
                          + static_cast<int>(px[0] / grid_.cell_size);
            //网格存储候选点
            grid_.cells.at(k)->push_back(Candidate(point, px));
            return true;
        }
        return false;
    }

} // namespace svo
