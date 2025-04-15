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
#ifndef PERCEPTION_YOLOV8_DET_NN_H
#define PERCEPTION_YOLOV8_DET_NN_H

#include "perception/base_perception.h"
#include "perception/common/common.h"
#include "perception/common/trt_utils.h"
#include "perception/yolov8/params.h"
#include "perception/yolov8/preprocess.h"
#include "perception/yolov8/postprocess.h"
#include "NvInfer.h"

namespace robosense {
namespace perception {

class Yolov8DetNN : public BasePerception {
public:
  Yolov8DetNN() {
    params_ptr_ = std::make_shared<Yolov8DetParams>();
  }
  ~Yolov8DetNN() {}
  void Init(const YAML::Node& cfg_node);
  bool LoadEngine(const std::string& engineFile);
  std::shared_ptr<cv::Mat> PreProcess(const Image &image);
  void PostProcess();
  void Perception(const DetectionMsg::Ptr& msg_ptr) override;

  void DrawImg(const DetectionMsg::Ptr &msg_ptr, const Image& image, const ObjectList& od_objects);
  void DetectResultToObjectInner(const ObjectList& od_results,uint64_t timestamp, perception::ObjectInnerArray::Ptr& out_msg);

  std::string Name() { return "Yolov8DetNN"; }

private:
  void InitInfer();
  void CheckNNAttr(nvinfer1::Dims dims);
  void InitMem();
  bool check_flag_ = true;

  Yolov8DetParams::Ptr params_ptr_;

  TrtLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_;

  TimeRecord preprocess_time_record_;
  TimeRecord infer_time_record_;
  TimeRecord postprocess_time_record_;

  static bool Yolov8DetNN_registered_;
};
REGISTER_CLASS(BasePerception, Yolov8DetNN);
} //namespace perception
} //namespace robosense

#endif // PERCEPTION_YOLOV8_DET_NN_H

