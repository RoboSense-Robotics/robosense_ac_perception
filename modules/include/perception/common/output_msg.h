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
#ifndef PERCEPTION_COMMON_OUTPUT_MSG_H_
#define PERCEPTION_COMMON_OUTPUT_MSG_H_

#include <opencv2/opencv.hpp>
#include "perception/common/basic_type.h"

namespace robosense {
namespace perception {

struct DetectionOutputMsg {
  using Ptr = std::shared_ptr<DetectionOutputMsg>;
  cv::Mat mat;
  ObjectInnerArray::Ptr object_list_ptr;
  // todo
};
}  // namespace perception
}  // namespace robosense

#endif  // PERCEPTION_COMMON_OUTPUT_MSG_H_
