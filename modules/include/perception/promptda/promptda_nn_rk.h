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
#ifndef PERCEPTION_PROMPTDA_NN_H
#define PERCEPTION_PROMPTDA_NN_H

#include "perception/base_perception.h"
#include "perception/common/common.h"
#include "perception/common/rknn_utils.h"
#include "perception/promptda/params.h"
#include "perception/promptda/inner_type.h"
#include "perception/promptda/preprocess.h"
#include "perception/promptda/postprocess.h"

#include "rknn_api.h"

namespace robosense {
namespace perception {

class PromptDANN : public BasePerception {
public:
  PromptDANN() {
    params_ptr_ = std::make_shared<PromptDAParams>();
  }
  ~PromptDANN() {
    rknn_destroy(s1_context_);
    rknn_destroy(s2_context_);
  }
  void Init(const YAML::Node& cfg_node);
  bool LoadEngine(const std::string& engineFile, rknn_context& context);
  void PreProcess(const Image &image, const PointCloud::Ptr cloud_ptr);
  void PostProcess(const DetectionMsg::Ptr &msg_ptr, const Image& image, std::pair<float, float> limit_depth = {0, 0});
  void Perception(const DetectionMsg::Ptr& msg_ptr) override;

  std::string Name() { return "PromptDANN"; }

private:
  void InitInfer();
  void InitCalibration(const std::string& calib_file);
  void InitPreProcess();
  void InitMem();
  void loadFixedInput();
  bool check_flag_ = true;
  std::vector<cv::Vec3b> colormap_;

  PromptDAParams::Ptr params_ptr_;
  LidarCalib::Ptr lidar_calib_;
  CameraCalib::Ptr camera_calib_;
  cv::Mat image_blob_;
  std::vector<cv::Mat> depth_blobs_;
  double input_min_, input_max_;

  int crop_start_y_, crop_end_y_, crop_start_x_, crop_end_x_;
  float img_scale_factor_ = 1;
  Eigen::Matrix4f lidar2cam_;
  Eigen::Matrix3f lidar_project_k_;
  float lidar_project_scale_factor_;
  int lidar_w_, lidar_h_;
  int knn_pooling_scale_, knn_k_;

  rknn_context s1_context_, s2_context_;
  rknn_input_output_num s1_num_io_tensors_, s2_num_io_tensors_;
  std::vector<rknn_input> s1_inputs_, s2_inputs_;
  std::vector<rknn_output> s1_outputs_, s2_outputs_;
  std::vector<rknn_tensor_attr> internal_attr_;

  TimeRecord image_preprocess_time_record_;
  TimeRecord lidar_preprocess_time_record_;
  TimeRecord knn_pooling_time_record_;
  TimeRecord preprocess_time_record_;
  TimeRecord infer_time_record_;
  TimeRecord postprocess_time_record_;

  static bool PromptDANN_registered_;
};
REGISTER_CLASS(BasePerception, PromptDANN);
} //namespace perception
} //namespace robosense

#endif // PERCEPTION_PROMPTDA_NN_H
