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
#ifndef PERCEPTION_NODE_ROS_DEP_H
#define PERCEPTION_NODE_ROS_DEP_H

#ifdef USE_ROS1 // 在ros1环境下的依赖
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

using PointCloud2MsgsConstPtr = sensor_msgs::PointCloud2::ConstPtr;
using ImageMsgsConstPtr = sensor_msgs::ImageConstPtr;
using PointCloud2Msgs = sensor_msgs::PointCloud2;

inline double HeaderToSec(const std_msgs::Header& header) {
  return header.stamp.toSec();
}
inline ros::Time SecToHeaderStamp(const double sec) {
  return ros::Time(sec);
}

inline double HeaderToNanoSec(const std_msgs::Header& header) {
  return header.stamp.toNSec();
}

template<typename... Args>
void RosInfoPrint(const std::string& format, const Args&... args) {
  ROS_INFO(format.c_str(), args...);
}

template<typename... Args>
void RosWarnPrint(const std::string& format, const Args&... args) {
  ROS_WARN(format.c_str(), args...);
}

template<typename... Args>
void RosErrorPrint(const std::string& format, const Args&... args) {
  ROS_ERROR(format.c_str(), args...);
}

inline ros::Time GetTimeNow() { return ros::Time::now(); }

inline double GetTimeNowInSecond() { return ros::Time::now().toSec(); }

inline bool IsRosOK() { return ros::ok(); }

#elif defined(USE_ROS2) // 在ros2环境下的依赖
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <robosense_msgs/msg/rs_image.hpp>
#include <robosense_msgs/msg/rs_point_cloud.hpp>

using PointCloud2MsgsConstPtr = sensor_msgs::msg::PointCloud2::SharedPtr;
using ImageMsgsConstPtr = sensor_msgs::msg::Image::SharedPtr;
using PointCloud2Msgs = sensor_msgs::msg::PointCloud2;

using RosRate = rclcpp::Rate;

inline double HeaderToSec(const std_msgs::msg::Header& header) {
  return rclcpp::Time(header.stamp).seconds();
}
inline rclcpp::Time SecToHeaderStamp(const double sec) {
  return rclcpp::Time(sec);
}
inline double HeaderToNanoSec(const std_msgs::msg::Header& header) {
  return rclcpp::Time(header.stamp).nanoseconds();
}

template<typename... Args>
void RosInfoPrint(const std::string& format, const Args&... args) {
  RCLCPP_INFO(rclcpp::get_logger("global_logger"), format.c_str(), args...);
}

template<typename... Args>
void RosWarnPrint(const std::string& format, const Args&... args) {
  RCLCPP_WARN(rclcpp::get_logger("global_logger"), format.c_str(), args...);
}

template<typename... Args>
void RosErrorPrint(const std::string& format, const Args&... args) {
  RCLCPP_ERROR(rclcpp::get_logger("global_logger"), format.c_str(), args...);
}

inline rclcpp::Time GetTimeNow() { return rclcpp::Clock().now(); }

inline double GetTimeNowInSecond() { return rclcpp::Clock().now().seconds(); }

inline bool IsRosOK() { return rclcpp::ok(); }

PointCloud2MsgsConstPtr ConvertZCPoints(const robosense_msgs::msg::RsPointCloud::Ptr msg) {
  PointCloud2MsgsConstPtr msg_ptr(new PointCloud2Msgs());
  // header
  std_msgs::msg::Header header;
  std::string frame_id;
  for(char c : msg->header.frame_id){
      if(c == '\0') break;
      frame_id.push_back(c);
  }
  header.frame_id = frame_id;
  header.stamp = rclcpp::Time(msg->header.stamp);
  msg_ptr->header = header;
  // other
  msg_ptr->is_bigendian = msg->is_bigendian;
  msg_ptr->point_step = msg->point_step;
  msg_ptr->row_step = msg->row_step;
  msg_ptr->height = msg->height;
  msg_ptr->width = msg->width;
  msg_ptr->is_dense = msg->is_dense;
  // fields
  msg_ptr->fields.resize(msg->fields.size());
  for (const auto& field : msg->fields) {
    sensor_msgs::msg::PointField point_field;
    std::string name;
    for (char c : field.name) {
      if (c == '\0') break;
      name.push_back(c);
    }
    point_field.name = name;
    point_field.offset = field.offset;
    point_field.datatype = field.datatype;
    point_field.count = field.count;
    msg_ptr->fields.push_back(point_field);
  }
  // data
  msg_ptr->data.resize(msg->row_step);
  // copy data
  memcpy(msg_ptr->data.data(), msg->data.data(), msg->row_step);
  return msg_ptr;
}
#endif
#endif  // PERCEPTION_NODE_ROS_DEP_H
