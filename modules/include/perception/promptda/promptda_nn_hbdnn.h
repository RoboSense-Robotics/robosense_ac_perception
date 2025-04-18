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
#include "perception/common/hbdnn_utils.h"
#include "perception/promptda/params.h"
#include "perception/promptda/inner_type.h"
#include "perception/promptda/preprocess.h"
#include "perception/promptda/postprocess.h"

#include "hb_dnn.h"

namespace robosense {
namespace perception {

class PromptDANN : public BasePerception {
public:
  PromptDANN() {
    params_ptr_ = std::make_shared<PromptDAParams>();
  }
  ~PromptDANN() {
    Infer();
    for (auto& x : inputs_) {
      hbSysFreeMem(&(x.sysMem[0]));
    }
    for (auto& x: outputs_) {
      hbSysFreeMem(&(x.sysMem[0]));
    }
    BASE_HB_CHECK(
      hbDNNRelease(mPackedDnnHandle_),
      "hbDNNRelease failed");
  }
  void Init(const YAML::Node& cfg_node);
  bool LoadEngine(const std::string& engineFile);
  void PreProcess(const Image &image, const PointCloud::Ptr cloud_ptr);
  void PostProcess(const DetectionMsg::Ptr &msg_ptr, const Image& image, std::pair<float, float> limit_depth = {0, 0});
  void Perception(const DetectionMsg::Ptr& msg_ptr) override;
  std::string Name() { return "PromptDANN"; }

private:
  void InitInfer();
  void InitCalibration(const std::string& calib_file);
  void InitPreProcess();
  void InitMem();
  void Infer();
  bool check_flag_ = true;
  std::vector<cv::Vec3b> colormap_;

  PromptDAParams::Ptr params_ptr_;
  LidarCalib::Ptr lidar_calib_;
  CameraCalib::Ptr camera_calib_;
  cv::Mat image_blob_;
  cv::Mat depth_blob_;
  double input_min_, input_max_;

  int crop_start_y_, crop_end_y_, crop_start_x_, crop_end_x_;
  float img_scale_factor_ = 1;
  Eigen::Matrix4f lidar2cam_;
  Eigen::Matrix3f lidar_project_k_;
  float lidar_project_scale_factor_;
  int lidar_w_, lidar_h_;
  int knn_pooling_scale_, knn_k_;

  hbPackedDNNHandle_t mPackedDnnHandle_;
  hbDNNHandle_t mDnnHandle_;
  std::vector<hbDNNTensor> inputs_;
  std::vector<hbDNNTensor> outputs_;

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
