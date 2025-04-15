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
#ifndef PERCEPTION_NODE_PROMPT_DA_NODE_H
#define PERCEPTION_NODE_PROMPT_DA_NODE_H
#include <thread>
#include <perception/common/queue.h>
#include <perception/node/ros_dep.h>
#include "perception/interface.h"
#if defined(ENABLE_TENSORRT)
#include "perception/promptda/promptda_nn_trt.h"
#elif defined(ENABLE_RKNN)
#include "perception/promptda/promptda_nn_rk.h"
#elif defined(ENABLE_HBDNN)
#include "perception/promptda/promptda_nn_hbdnn.h"
#endif
#include <cv_bridge/cv_bridge.h>
namespace robosense {
namespace perception {

struct PromptDANodeOptions {
  std::string sub_image_topic;
  std::string sub_lidar_topic;
  YAML::Node promptda_cfg_node;
  int64_t time_diff_thresh = 50 * 1e6;
  void Load(const YAML::Node& cfg_node) {
    rally::yamlRead(cfg_node, "sub_image_topic", sub_image_topic);
    rally::yamlRead(cfg_node, "sub_lidar_topic", sub_lidar_topic);
    rally::yamlRead(cfg_node, "time_diff_thresh", time_diff_thresh);
    time_diff_thresh *= 1e6;
    rally::yamlSubNode(cfg_node, "promptda", promptda_cfg_node);
  }
};

class PromptDANode {
public:
  PromptDANode(const PromptDANodeOptions& options);
  ~PromptDANode() {
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
    return "PromptDANode";
  }
  void ImageCallback(const ImageMsgsConstPtr msg);
  void LidarCallback(const PointCloud2MsgsConstPtr msg);
  perception::PointCloud::Ptr FindNearestLidarMsg(const int64_t timestamp);
#if defined(USE_ROS1)
  ros::NodeHandle nh_;
  ros::Subscriber image_subscription_, lidar_subscription_;
  ros::Publisher publisher_;
#elif defined(USE_ROS2)
  bool zero_copy_ = false;
  void ZCImageCallback(const robosense_msgs::msg::RsImage::Ptr msg);
  void ZCLidarCallback(const robosense_msgs::msg::RsPointCloud::Ptr msg);
  rclcpp::Node::SharedPtr ros2_node_ptr_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<robosense_msgs::msg::RsImage>::SharedPtr image_subscription_zc_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
  rclcpp::Subscription<robosense_msgs::msg::RsPointCloud>::SharedPtr lidar_subscription_zc_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
#endif
  std::deque<perception::PointCloud::Ptr> lidar_msg_cache_;
  int max_cache_size_ = 10;
  PromptDANodeOptions options_;
  perception::PerceptionInterface::Ptr promptda_ptr_;
};
} // namespace perception
}  // namespace robosense

#endif  // PERCEPTION_NODE_promptda_NODE_H
