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
#ifndef PERCEPTION_COMMON_INPUT_MSG_H_
#define PERCEPTION_COMMON_INPUT_MSG_H_

#include <map>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// #include "perception/common/basic_type.h"
namespace robosense {
namespace perception {

struct Image {
  int64_t timestamp;
  cv::Mat mat;
};

struct PointCloud {
  using Ptr = std::shared_ptr<PointCloud>;
  int64_t timestamp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
};
struct DetectionInputMsg {
  using Ptr = std::shared_ptr<DetectionInputMsg>;
  std::map<std::string, Image> camera_data_map;
  PointCloud::Ptr cloud_ptr;
};

}  // namespace perception
}  // namespace robosense

#endif  // PERCEPTION_COMMON_INPUT_MSG_H_
