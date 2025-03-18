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
#ifndef PERCEPTION_DETECTION_INTERFACE_H_
#define PERCEPTION_DETECTION_INTERFACE_H_

#include "perception/base_perception.h"

namespace robosense {
namespace perception {

struct PerceptionInterfaceOptions {
  std::string strategy = "Yolov8";
  YAML::Node  strategy_cfg_ndoe;
  void Load(const YAML::Node& cfg_node) {
    rally::yamlRead(cfg_node, "strategy", strategy);
    rally::yamlSubNode(cfg_node, strategy, strategy_cfg_ndoe);
  }
};

class PerceptionInterface {
 public:
  using Ptr = std::shared_ptr<PerceptionInterface>;

  PerceptionInterface(const PerceptionInterfaceOptions& options) {
    options_ = options;
    std::cout << "Detection Strategy: " << options.strategy << std::endl;
    impl_ptr_ = Factory<BasePerception>::Create(options.strategy);
    if (impl_ptr_ == nullptr) {
      std::cout << "Detection Strategy: " << options.strategy << " not found"
                << std::endl;
    }
    impl_ptr_->Init(options_.strategy_cfg_ndoe);

    process_time_record_.init("Perception Process Time");
  }

  void Process(const DetectionMsg::Ptr& msg_ptr) {
    std::cout  << name() << ": receive input msg!" << std::endl;
    process_time_record_.tic();
    impl_ptr_->Perception(msg_ptr);
    process_time_record_.toc();
  }

 private:
  std::string name() {
    return "PerceptionInterface";
  }
  PerceptionInterfaceOptions options_;
  std::unique_ptr<BasePerception> impl_ptr_;
  TimeRecord process_time_record_;
};

}  // namespace perception
}  // namespace robosense

#endif  // PERCEPTION_DETECTION_INTERFACE_H_
