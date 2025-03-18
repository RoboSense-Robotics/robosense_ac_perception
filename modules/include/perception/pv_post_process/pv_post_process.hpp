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
#ifndef PERCEPTION_PV_POST_PROCESS_PV_POST_PROCESS_H
#define PERCEPTION_PV_POST_PROCESS_PV_POST_PROCESS_H
#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_map>

#include "perception/common/basic_type.h"
#include "perception/common/time_record.h"
#include "perception/common/yaml.h"
namespace robosense {
namespace perception {
struct InputMsg {
  using Ptr = std::shared_ptr<InputMsg>;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
  ObjectInnerArray::Ptr input_object;
};
struct OutputMsg {
  using Ptr = std::shared_ptr<OutputMsg>;
  ObjectInnerArray::Ptr output_object;
  // pcl::PointCloud<pcl::PointXYZ>::Ptr debug_cloud_ptr;
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> debug_cloud_ptr;
  cv::Mat debug_image;
  OutputMsg() {
    output_object = std::make_shared<ObjectInnerArray>();
    debug_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    debug_image = cv::Mat::zeros(1080, 1920, CV_8UC3);
  }
};
struct MsgInterface {
  using Ptr = std::shared_ptr<MsgInterface>;
  InputMsg::Ptr input_msg_ptr;
  OutputMsg::Ptr output_msg_ptr;
  MsgInterface() {
    input_msg_ptr = std::make_shared<InputMsg>();
    output_msg_ptr = std::make_shared<OutputMsg>();
  }
};
struct LidarCalib {
  using Ptr = std::shared_ptr<LidarCalib>;
  Eigen::Affine3f lidar2cam_trans;
  Eigen::Affine3f lidar2imu_trans;
};
struct CameraCalib {
  using Ptr = std::shared_ptr<CameraCalib>;
  cv::Mat K;
  cv::Mat D;
  double mean_error;
};

class PvPostProcess {
 public:
  using Ptr = std::shared_ptr<PvPostProcess>;
  bool Init(const YAML::Node& config);
  void SetCalibration(const std::string& calib_file);
  void Perception(MsgInterface::Ptr msg_ptr);
  int image_width_ = 1920;
  int image_height_ = 1080;

 private:
  LidarCalib::Ptr lidar_calib_;
  CameraCalib::Ptr camera_calib_;
  cv::Mat mask_image_;
  bool mask_cover_ped_region_ = true;
  bool debug_image_ = false;
  bool debug_lidar_ = false;
  std::vector<ObjectInner> history_obj_;

  // box cover mask
  void GetCoverMask(const ObjectInnerArray::Ptr& object_array);
  void SegmentPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_ptr,
                         const ObjectInnerArray::Ptr& object_array,
                         std::vector<std::vector<int>>& segment_index);
  void Set3DInfo(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_ptr,
                 const ObjectInnerArray::Ptr& object_array,
                 std::vector<std::vector<int>>& segment_index,
                 ObjectInnerArray::Ptr& out_object,
                 std::vector<std::vector<int>>& segment_result);
  void MatchAndSmooth(ObjectInnerArray::Ptr& out_object);

  perception::TimeRecord cover_record_;
  perception::TimeRecord segment_record_;
  perception::TimeRecord set_record_;
};

}  // namespace perception

}  // namespace robosense

#endif