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

#ifndef PERCEPTION_YOLOV8_MSG_H
#define PERCEPTION_YOLOV8_MSG_H
#include <memory>
#include <cstdint>
#include <string>

#include "perception/yolov8/inner_type.h"

#ifdef ENABLE_TENSORRT
#include "NvInfer.h"
#endif

namespace robosense {
namespace perception {

class Yolov8DetMsg {
public:
  using Ptr = std::unique_ptr<Yolov8DetMsg>;
  ~Yolov8DetMsg() {
#ifdef ENABLE_TENSORRT
    for (auto&x : gpu_inputs) {
      cudaFree(x);
    }
    for (auto &x : gpu_outputs) {
      cudaFree(x);
    }
#endif
#ifdef ENABLE_RKNN
    for (auto x : inputs) {
      if (x != nullptr) {
        free(x);
      }
    }
    for (auto& x : nn_outputs) {
      if (x != nullptr) {
        free(x);
      }
    }
#endif
  }
#ifdef ENABLE_TENSORRT
  std::vector<void*> gpu_inputs;
  std::vector<void*> gpu_outputs;
#endif
  std::vector<void*> inputs;
  std::vector<void*> nn_outputs;

  LetterBoxInfo lb;
  ObjectList od_results;

private:
  static std::string Name() { return "Yolov8DetMsg"; }
};

} // namespace robosense
} // namespace perception

#endif // PERCEPTION_YOLOV8_MSG_H
