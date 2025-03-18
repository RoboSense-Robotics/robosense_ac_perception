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

#ifndef PERCEPTION_YOLOV8_CONFIG_H
#define PERCEPTION_YOLOV8_CONFIG_H

#include<memory>
#include<string>

#include <perception/common/yaml.h>

namespace robosense {
namespace perception {

namespace pvnn2d {
  static const char strategy[] = "Yolov8";
}
class Yolov8DetConfig {
public:
  using Ptr = std::unique_ptr<Yolov8DetConfig>;

  void Init(const YAML::Node &cfg_node);

  std::string sub_topic;
  std::string model_path;
  int batch_size{1};
  // npu core TODO
  float nms_threshold{0.45};
  float box_conf_threshold{0.25};
  int nn_input_width{640};
  int nn_input_height{640};

  bool save_img;
private:
  static std::string Name() { return "Yolov8DetConfig"; }
};

} // namespace robosense
} // namespace perception

#endif // PERCEPTION_YOLOV8_CONFIG_H
