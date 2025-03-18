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
#ifndef PERCEPTION_PPSEG_NN_RK_H
#define PERCEPTION_PPSEG_NN_RK_H

#include "perception/base_perception.h"
#include "perception/common/common.h"
#include "perception/common/rknn_utils.h"
#include "perception/ppseg/params.h"
#include "perception/ppseg/preprocess.h"
#include "perception/ppseg/postprocess.h"
#include "perception/ppseg/inner_type.h"

#include "rknn_api.h"

namespace robosense {
namespace perception {
class PPSegNN : public BasePerception {
public:
PPSegNN() {
  params_ptr_ = std::make_shared<PPSegParams>();
}

void Init(const YAML::Node& cfg_node);
bool LoadEngine(const std::string& engineFile);
void PreProcess(Image &image);
void PostProcess(const DetectionMsg::Ptr &msg_ptr, const Image& image);
void Perception(const DetectionMsg::Ptr& msg_ptr) override;

std::string Name() { return "PPSegNN"; }

private:
void InitInfer();
void InitCalibration(const std::string& calib_file);
void InitMem();
void CheckNNAttr(rknn_tensor_attr tenso_attr);
bool check_flag_ = true;
CameraCalib::Ptr camera_calib_;
cv::Mat image_blob_;

PPSegParams::Ptr params_ptr_;
rknn_context context_;
rknn_input_output_num num_io_tensors_;
std::vector<rknn_input> inputs_;
std::vector<rknn_output> outputs_;

TimeRecord preprocess_time_record_;
TimeRecord infer_time_record_;
TimeRecord postprocess_time_record_;

static bool PPSegNN_registered_;
};
REGISTER_CLASS(BasePerception, PPSegNN);
} //namespace perception
} //namespace robosense

#endif // PERCEPTION_PPSEG_NN_RK_H

