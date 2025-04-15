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
#ifndef PERCEPTION_NODE_PPSEG_NODE_H
#define PERCEPTION_NODE_PPSEG_NODE_H

#include <thread>
#include <perception/common/queue.h>
#include <perception/node/ros_dep.h>
#include "perception/interface.h"
#if defined(ENABLE_TENSORRT)
#include "perception/ppseg/ppseg_nn_trt.h"
#elif defined(ENABLE_RKNN)
#include "perception/ppseg/ppseg_nn_rk.h"
#elif defined(ENABLE_HBDNN)
#include "perception/ppseg/ppseg_nn_hbdnn.h"
#endif
#include <cv_bridge/cv_bridge.h>

namespace robosense {
namespace perception {

struct PPSegNodeOptions {
  std::string sub_image_topic;
  YAML::Node ppseg_cfg_node;
  void Load(const YAML::Node& cfg_node) {
    rally::yamlRead(cfg_node, "sub_image_topic", sub_image_topic);
    rally::yamlSubNode(cfg_node, "ppseg", ppseg_cfg_node);
  }
};

class PPSegNode {
public:
  PPSegNode(const PPSegNodeOptions& options);
  ~PPSegNode() {
    Stop();
  }

#if defined(USE_ROS2)
  rclcpp::Node::SharedPtr GetRos2Node() {
    return ros2_node_ptr_;
  }
#endif
private:
  void Start();
  void Stop();
  void Core();
  bool run_flag_ = false;
  std::unique_ptr<std::thread> thread_ptr_;
  SyncQueue<perception::DetectionMsg::Ptr> msg_queue_{10};
  std::string Name() {
    return "PPSegNode";
  }
  void ImageCallback(const ImageMsgsConstPtr msg);
#if defined(USE_ROS1)
  ros::NodeHandle nh_;
  ros::Subscriber subscription_;
  ros::Publisher publisher_;
#elif defined(USE_ROS2)
  bool zero_copy_ = false;
  void ZCImageCallback(const robosense_msgs::msg::RsImage::Ptr msg);
  rclcpp::Node::SharedPtr ros2_node_ptr_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Subscription<robosense_msgs::msg::RsImage>::SharedPtr subscription_zc_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
#endif
  PPSegNodeOptions options_;
  perception::PerceptionInterface::Ptr seg_ptr_;
};

} // namespace perception
} // namespace robosense

#endif  // PERCEPTION_NODE_PPSEG_NODE_H
