/************************************************************************
Copyright 2025 RoboSense Technology Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
************************************************************************/

#ifndef PERCEPTION_PROMPTDA_PREPROCESS_H
#define PERCEPTION_PROMPTDA_PREPROCESS_H

#include "perception/common/common.h"
#include "perception/promptda/inner_type.h"

namespace robosense {
namespace perception {
  cv::Mat UndistortImg(const cv::Mat& img, const cv::Mat& K, const cv::Mat& D);
  cv::Mat KnnPooling(const cv::Mat& input_matrix, int pool_size, int k);
  cv::Mat KnnPoolingOptimized(const cv::Mat& input_matrix, int pool_size, int k);
  cv::Mat ProjectLidar(const perception::PointCloud::Ptr& points, const Eigen::Matrix4f& ego2sensor_rt, const Eigen::Matrix3f& intrinsic);
  cv::Mat ProjectLidarOptimized(const perception::PointCloud::Ptr& points, const Eigen::Matrix4f& ego2sensor_rt, const Eigen::Matrix3f& intrinsic);
  void ProjectLidarOptimized2(const perception::PointCloud::Ptr& points, const Eigen::Matrix4f& ego2sensor_rt, const Eigen::Matrix3f& intrinsic, const cv::Mat& ori_img);
} // namespace robosense
} // namespace perception
#endif // PERCEPTION_PROMPTDA_PREPROCESS_H