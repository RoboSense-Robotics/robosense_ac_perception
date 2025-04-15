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
#ifndef PERCEPTION_NODE_YOLOV8_NODE_H
#define PERCEPTION_NODE_YOLOV8_NODE_H
#include <thread>
#include <perception/common/queue.h>
#include <perception/node/ros_dep.h>
#include "perception/interface.h"
#if defined(ENABLE_TENSORRT)
#include "perception/yolov8/yolov8_det_nn_trt.h"
#elif defined(ENABLE_RKNN)
#include "perception/yolov8/yolov8_det_nn_rk.h"
#elif defined(ENABLE_HBDNN)
#include "perception/yolov8/yolov8_det_nn_hbdnn.h"
#endif

#if defined(USE_ROS1)
#include "perception_msgs/ObjectInner.h"
#include "perception_msgs/ObjectInnerArray.h"
using ROSObjectInnerArray = perception_msgs::ObjectInnerArray;
#elif defined(USE_ROS2)
#include "perception_msgs/msg/object_inner.hpp"
#include "perception_msgs/msg/object_inner_array.hpp"
using ROSObjectInnerArray = perception_msgs::msg::ObjectInnerArray;
#endif

#include <cv_bridge/cv_bridge.h>
namespace robosense {
namespace perception {

struct Yolov8NodeOptions {
  std::string sub_image_topic;
  YAML::Node yolov8_cfg_node;
  void Load(const YAML::Node& cfg_node) {
    rally::yamlRead(cfg_node, "sub_image_topic", sub_image_topic);
    rally::yamlSubNode(cfg_node, "yolov8", yolov8_cfg_node);
  }
};

class Yolov8Node {
public:
  Yolov8Node(const Yolov8NodeOptions& options);
  ~Yolov8Node() {
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
    return "Yolov8Node";
  }
  void ImageCallback(const ImageMsgsConstPtr msg);
  void ObjectInnerArrayToRosMsg(const perception::ObjectInnerArray::Ptr& output_msg, ROSObjectInnerArray& ros_msg);
#if defined(USE_ROS1)
  ros::NodeHandle nh_;
  ros::Subscriber subscription_;
  ros::Publisher publisher_, box_publisher_;
#elif defined(USE_ROS2)
  bool zero_copy_ = false;
  void ZCImageCallback(const robosense_msgs::msg::RsImage::Ptr msg);
  rclcpp::Node::SharedPtr ros2_node_ptr_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Subscription<robosense_msgs::msg::RsImage>::SharedPtr subscription_zc_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<ROSObjectInnerArray>::SharedPtr box_publisher_;
#endif
  Yolov8NodeOptions options_;
  perception::PerceptionInterface::Ptr detection_ptr_;
};
} // namespace perception
}  // namespace robosense

#endif  // PERCEPTION_NODE_YOLOV8_NODE_H
