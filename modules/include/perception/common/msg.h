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
#ifndef PERCEPTION_COMMON_MSG_H_
#define PERCEPTION_COMMON_MSG_H_

#include "perception/common/input_msg.h"
#include "perception/common/output_msg.h"

namespace robosense {
namespace perception {

struct DetectionMsg {
  using Ptr = std::shared_ptr<DetectionMsg>;

  DetectionInputMsg::Ptr input_msg_ptr;
  DetectionOutputMsg::Ptr output_msg_ptr;
};

}  // namespace perception
}  // namespace robosense

#endif  // PERCEPTION_COMMON_MSG_H_
