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

#ifndef PERCEPTION_YOLOV8_MODEL_ATTR_H
#define PERCEPTION_YOLOV8_MODEL_ATTR_H

#include<string>
#include<memory>
#include<cassert>
#include "perception/common/utils.h"

namespace robosense {
namespace perception {
class Yolov8DetModelAttr {
public:
  using Ptr = std::unique_ptr<Yolov8DetModelAttr>;

  bool is_quant;
  int n_output;
  std::vector<TensorAttr> output_attrs;

private:
  static std::string Name() { return "Yolov8DetModelAttr"; }
};

} // namespace robosense
} // namespace perception

#endif // PERCEPTION_YOLOV8_MODEL_ATTR_H
