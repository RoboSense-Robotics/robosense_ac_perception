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

#ifndef PERCEPTION_PPSEG_CONFIG_H
#define PERCEPTION_PPSEG_CONFIG_H

#include<memory>
#include<string>

#include "perception/common/yaml.h"

namespace robosense {
namespace perception {

class PPSegConfig {
public:
  using Ptr = std::unique_ptr<PPSegConfig>;

  void Init(const YAML::Node &cfg_node);

  std::string sub_topic;
  std::string model_path;
  std::string calib_file;
  int batch_size{1};
  // npu core TODO
  int nn_input_width{512};
  int nn_input_height{512};
  bool save_img;

private:
  static std::string Name() { return "PPSegConfig"; }
};

} // namespace robosense
} // namespace perception

#endif // PERCEPTION_PPSEG_CONFIG_H
