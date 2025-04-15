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
#ifndef PERCEPTION_NODE_PV_POST_PROCESS_NODE_H
#define PERCEPTION_NODE_PV_POST_PROCESS_NODE_H
#include <perception/node/ros_dep.h>
#include <cv_bridge/cv_bridge.h>
#include "perception/common/basic_type.h"
#include "perception/common/time_record.h"
#include "perception/pv_post_process/pv_post_process.hpp"
#if defined(USE_ROS1)
#include "perception_msgs/ObjectInnerArray.h"
using ObjectInnerArrayConstPtr = perception_msgs::ObjectInnerArray::ConstPtr;
using ROSObjectInnerArray = perception_msgs::ObjectInnerArray;
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
using VisMarkerArray = visualization_msgs::MarkerArray;
using VisMarker = visualization_msgs::Marker;
using GeometryPoint = geometry_msgs::Point;
#elif defined(USE_ROS2)
#include "perception_msgs/msg/object_inner_array.hpp"
using ObjectInnerArrayConstPtr = perception_msgs::msg::ObjectInnerArray::ConstSharedPtr;
using ROSObjectInnerArray = perception_msgs::msg::ObjectInnerArray;
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
using VisMarkerArray = visualization_msgs::msg::MarkerArray;
using VisMarker = visualization_msgs::msg::Marker;
using GeometryPoint = geometry_msgs::msg::Point;
#endif
namespace robosense {
namespace perception {
class PvPostProcessApp {
public:
  PvPostProcessApp(const YAML::Node& configs);
#if defined(USE_ROS2)
  rclcpp::Node::SharedPtr GetRos2Node() {
    return ros2_node_ptr_;
  }
#endif
 private:
  void ImageCallback(const ImageMsgsConstPtr msg);
  void LidarCallback(const PointCloud2MsgsConstPtr msg);
  void ObjectCallback(const ObjectInnerArrayConstPtr msg);
  pcl::PointCloud<pcl::PointXYZ>::Ptr FindNearestLidarMsg(const int64_t timestamp);
  void MsgToObjectInnerArray(const ObjectInnerArrayConstPtr ros_msg,
      perception::ObjectInnerArray::Ptr& object_inner_array);
  void ObjectInnerArrayToRosMsg(
      const perception::ObjectInnerArray::Ptr& output_msg,
      ROSObjectInnerArray& ros_msg);
  void ObjectInnerArrayToMarker(
      const perception::ObjectInnerArray::Ptr& output_msg,
      const std::string frame_id);
  void DrawDistLabel();
  VisMarker ClearMarker(const std::string frame_id);
  VisMarker DistCircleMarker(const std::string frame_id, const int id, const float dist);
  VisMarker TextMarker(const std::string frame_id, const int id,
                       const Eigen::Vector3d& pose, const std::string& text);
  VisMarker ClearIdMarker(const std::string frame_id, const int id);
  VisMarker ObjectInnerToLineMarker(const perception::ObjectInner& obj, const int id,
      const cv::Scalar& rgb_color_a, const std::string frame_id);
#if defined(USE_ROS1)
  ros::NodeHandle nh_;
  ros::Subscriber image_subscription_, lidar_subscription_, object_subscription_;
  ros::Publisher box_publisher_, lidar_publisher_, marker_publisher_;
#elif defined(USE_ROS2)
  bool zero_copy_ = false;
  rclcpp::Node::SharedPtr ros2_node_ptr_;
  void ZCLidarCallback(const robosense_msgs::msg::RsPointCloud::Ptr zc_msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      lidar_subscription_;
  rclcpp::Subscription<robosense_msgs::msg::RsPointCloud>::SharedPtr lidar_subscription_zc_;
  rclcpp::Subscription<perception_msgs::msg::ObjectInnerArray>::SharedPtr
      object_subscription_;

  rclcpp::Publisher<perception_msgs::msg::ObjectInnerArray>::SharedPtr
      box_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      debug_point_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      out_lidar_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_publisher_;
#endif

  int max_cache_size_ = 10;
  std::deque<ImageMsgsConstPtr> image_msg_cache_;
  std::deque<std::pair<int64_t, pcl::PointCloud<pcl::PointXYZ>::Ptr>>
      lidar_msg_cache_;

  int64_t time_diff_thresh_ = 50 * 1e6;
  int pre_pub_count = 0;

  perception::MsgInterface::Ptr msg_interface_;
  perception::PvPostProcess::Ptr task_ptr_;
  perception::TimeRecord pv_post_process_record_;
  std::string frame_id_ = "base_link";

  VisMarkerArray dist_markers;

  bool ground_update = false;
  float ground_height_ = 0;
  float ground_pitch_ = 0;
};
} // namespace perception
}  // namespace robosense
#endif