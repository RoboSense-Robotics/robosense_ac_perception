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

#ifndef PERCEPTION_PROMPTDA_INNER_TYPE_H
#define PERCEPTION_PROMPTDA_INNER_TYPE_H

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

namespace robosense {
namespace perception {
struct LidarCalib {
  using Ptr = std::shared_ptr<LidarCalib>;
  Eigen::Affine3f lidar2cam_trans;
  Eigen::Affine3f lidar2imu_trans;
};

struct CameraCalib {
  using Ptr = std::shared_ptr<CameraCalib>;
  cv::Mat K;
  cv::Mat D;
  double mean_error;
};
} // namespace robosense
} // namespace perception

#endif // PERCEPTION_PROMPTDA_INNER_TYPE_H