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
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include "perception/common/basic_type.h"
#include "perception/common/time_record.h"
#include "perception/pv_post_process/pv_post_process.hpp"
#include "perception_msgs/msg/object_inner_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
namespace robosense {
class PvPostProcessApp : public rclcpp::Node {
 public:
  PvPostProcessApp(const YAML::Node& configs)
      : rclcpp::Node("pv_post_process_node") {
    std::string image_topic = "image_rgb";
    rally::yamlRead(configs, "sub_image_topic", image_topic);
    std::string lidar_topic = "/rslidar_points_origin_deocc_rgb";
    rally::yamlRead(configs, "sub_lidar_topic", lidar_topic);
    std::string object_topic = "yolov8_object";
    rally::yamlRead(configs, "ground_height", ground_height_);
    rally::yamlRead(configs, "sensor_pitch", ground_pitch_);
    rally::yamlRead(configs, "sub_object_topic", object_topic);
    msg_interface_ = std::make_shared<perception::MsgInterface>();
    task_ptr_ = std::make_shared<perception::PvPostProcess>();
    task_ptr_->Init(configs);
    // pub
    std::string pub_topic = "pv_post_process_object";
    rally::yamlRead(configs, "pub_object_topic", pub_topic);
    box_publisher_ =
        this->create_publisher<perception_msgs::msg::ObjectInnerArray>(
            pub_topic, 10);
    marker_publisher_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            pub_topic + "_marker", 10);
    std::cout << "publish to object topic: " << pub_topic << std::endl;
    bool debug_image = false;
    rally::yamlRead(configs, "debug_image", debug_image);
    if (debug_image) {
      debug_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
          "pv_post_debug_image", 10);
    }
    bool debug_lidar = false;
    rally::yamlRead(configs, "debug_lidar", debug_lidar);
    if (debug_lidar) {
      debug_point_publisher_ =
          this->create_publisher<sensor_msgs::msg::PointCloud2>(
              "pv_post_debug_lidar", 10);
    }

    // Subscribe to the image topic
    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, 10,
        std::bind(&PvPostProcessApp::ImageCallback, this,
                  std::placeholders::_1));
    std::cout << "subscribed to image topic: " << image_topic << std::endl;
    lidar_subscription_ =
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic, 10,
            std::bind(&PvPostProcessApp::LidarCallback, this,
                      std::placeholders::_1));
    std::cout << "subscribed to lidar topic: " << lidar_topic << std::endl;
    object_subscription_ =
        this->create_subscription<perception_msgs::msg::ObjectInnerArray>(
            object_topic, 10,
            std::bind(&PvPostProcessApp::ObjectCallback, this,
                      std::placeholders::_1));
    std::cout << "subscribed to object topic: " << object_topic << std::endl;
    pv_post_process_record_.init("pv_post_process");

    // clear
    visualization_msgs::msg::MarkerArray clear_markers;
    clear_markers.markers.push_back(ClearMarker(frame_id_));
    marker_publisher_->publish(clear_markers);

    // dist circle
    DrawDistLabel();
    marker_publisher_->publish(dist_markers);
  }

 private:
  void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    if (image_msg_cache_.size() >= max_cache_size_) {
      image_msg_cache_.pop_front();
    }
    image_msg_cache_.emplace_back(msg);
    int64_t timestamp = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    std::cout << "Get image msg with timestamp: " << timestamp << std::endl;
  }
  void LidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    int64_t timestamp = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    std::cout << "Get lidar msg with timestamp: " << timestamp << std::endl;
    if (lidar_msg_cache_.size() >= max_cache_size_) {
      lidar_msg_cache_.pop_front();
    }
    // only xyz
    pcl::PointCloud<pcl::PointXYZ>::Ptr ori_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *ori_cloud);
    lidar_msg_cache_.emplace_back(std::make_pair(timestamp, ori_cloud));
    frame_id_ = msg->header.frame_id;
  }
  void ObjectCallback(
      const perception_msgs::msg::ObjectInnerArray::ConstSharedPtr msg) {
    marker_publisher_->publish(dist_markers);
    int64_t timestamp = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    // find lidar
    auto lidar_msg = FindNearestLidarMsg(timestamp);
    int circle = 0;
    while (lidar_msg == nullptr && circle < 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      lidar_msg = FindNearestLidarMsg(timestamp);
      ++circle;
    }
    if (lidar_msg == nullptr) {
      std::cout << "no match lidar msg found!" << std::endl;
      return;
    } else {
      std::cout << " find match lidar and process " << timestamp << std::endl;
    }
    pv_post_process_record_.tic();
    msg_interface_->input_msg_ptr->cloud_ptr = lidar_msg;
    MsgToObjectInnerArray(msg, msg_interface_->input_msg_ptr->input_object);
    task_ptr_->Perception(msg_interface_);
    perception_msgs::msg::ObjectInnerArray out_msg;
    ObjectInnerArrayToRosMsg(msg_interface_->output_msg_ptr->output_object,
                             out_msg);
    out_msg.header = msg->header;
    out_msg.header.frame_id = frame_id_;
    box_publisher_->publish(out_msg);
    ObjectInnerArrayToMarker(msg_interface_->output_msg_ptr->output_object,
                             out_msg.header.frame_id);
    // debug publish
    if (debug_point_publisher_ != nullptr) {
      sensor_msgs::msg::PointCloud2 debug_lidar_msg;
      pcl::toROSMsg(*msg_interface_->output_msg_ptr->debug_cloud_ptr,
                    debug_lidar_msg);
      debug_lidar_msg.header = msg->header;
      debug_lidar_msg.header.frame_id = frame_id_;
      debug_point_publisher_->publish(debug_lidar_msg);
    }
    if (debug_image_publisher_ != nullptr) {
      sensor_msgs::msg::Image::SharedPtr debug_image_msg =
      cv_bridge::CvImage(msg->header, "bgr8", msg_interface_->output_msg_ptr->debug_image).toImageMsg();

      debug_image_publisher_->publish(*debug_image_msg);
    }
    pv_post_process_record_.toc();
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr FindNearestLidarMsg(
      const int64_t timestamp) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_msg = nullptr;
    int64_t min_diff_time = time_diff_thresh_ * 2;
    for (const auto& msg : lidar_msg_cache_) {
      auto time_diff = std::abs(timestamp - msg.first);
      if (time_diff < time_diff_thresh_ && time_diff < min_diff_time) {
        min_diff_time = time_diff;
        nearest_msg = msg.second;
      }
    }
    return nearest_msg;
  }
  void MsgToObjectInnerArray(
      const perception_msgs::msg::ObjectInnerArray::ConstSharedPtr ros_msg,
      perception::ObjectInnerArray::Ptr& object_inner_array) {
    if (object_inner_array == nullptr) {
      object_inner_array = std::make_shared<perception::ObjectInnerArray>();
    }
    int64_t timestamp =
        ros_msg->header.stamp.sec * 1e9 + ros_msg->header.stamp.nanosec;
    object_inner_array->header.frame_id = ros_msg->header.frame_id;
    object_inner_array->header.time = timestamp;

    object_inner_array->object_list.resize(ros_msg->object_list.size());
    for (int i = 0; i < ros_msg->object_list.size(); ++i) {
      auto& inner_obj = object_inner_array->object_list[i];
      const auto& obj = ros_msg->object_list[i];
      inner_obj.object_id = obj.object_id;
      inner_obj.type = static_cast<perception::ObjectType>(obj.type);
      inner_obj.type_confidence = obj.type_confidence;
      inner_obj.box_full.x = obj.box_full.x;
      inner_obj.box_full.y = obj.box_full.y;
      inner_obj.box_full.width = std::min(
          obj.box_full.width, task_ptr_->image_width_ - obj.box_full.x - 1);
      inner_obj.box_full.height = std::min(
          obj.box_full.height, task_ptr_->image_height_ - obj.box_full.y - 1);
      inner_obj.box_visible.x = obj.box_visible.x;
      inner_obj.box_visible.y = obj.box_visible.y;
      inner_obj.box_visible.width = obj.box_visible.width;
      inner_obj.box_visible.height = obj.box_visible.height;
      inner_obj.box_center_base.x = obj.box_center_base.x;
      inner_obj.box_center_base.y = obj.box_center_base.y;
      inner_obj.box_center_base.z = obj.box_center_base.z;
      inner_obj.box_size.length = obj.box_size.length;
      inner_obj.box_size.width = obj.box_size.width;
      inner_obj.box_size.height = obj.box_size.height;
      inner_obj.yaw_base = obj.yaw_base;
    }
  }
  void ObjectInnerArrayToRosMsg(
      const perception::ObjectInnerArray::Ptr& output_msg,
      perception_msgs::msg::ObjectInnerArray& ros_msg) {
    ros_msg.header.frame_id = output_msg->header.frame_id;
    ros_msg.header.stamp = rclcpp::Time(double(output_msg->header.time) / 1e9);
    // ros_msg.header.seq = output_msg->header.seq;

    ros_msg.object_list.resize(output_msg->object_list.size());
    for (int i = 0; i < output_msg->object_list.size(); ++i) {
      auto& ros_obj = ros_msg.object_list[i];
      const auto& obj = output_msg->object_list[i];
      ros_obj.object_id = obj.object_id;
      ros_obj.type = static_cast<uint8_t>(obj.type);
      ros_obj.type_confidence = obj.type_confidence;
      ros_obj.box_full.x = obj.box_full.x;
      ros_obj.box_full.y = obj.box_full.y;
      ros_obj.box_full.width = obj.box_full.width;
      ros_obj.box_full.height = obj.box_full.height;
      ros_obj.box_visible.x = obj.box_visible.x;
      ros_obj.box_visible.y = obj.box_visible.y;
      ros_obj.box_visible.width = obj.box_visible.width;
      ros_obj.box_visible.height = obj.box_visible.height;
      ros_obj.box_center_base.x = obj.box_center_base.x;
      ros_obj.box_center_base.y = obj.box_center_base.y;
      ros_obj.box_center_base.z = obj.box_center_base.z;
      ros_obj.box_size.length = obj.box_size.length;
      ros_obj.box_size.width = obj.box_size.width;
      ros_obj.box_size.height = obj.box_size.height;
      ros_obj.yaw_base = obj.yaw_base;
    }
  }
  void ObjectInnerArrayToMarker(
      const perception::ObjectInnerArray::Ptr& output_msg,
      const std::string frame_id) {
    visualization_msgs::msg::MarkerArray object_markers;
    int count = 0;
    for (const auto& obj : output_msg->object_list) {
      object_markers.markers.emplace_back(ObjectInnerToLineMarker(
          obj, count++, cv::Scalar(0, 255, 0, 0.5), frame_id_));
    }
    for (int i = count; i <= pre_pub_count; ++i) {
      object_markers.markers.emplace_back(ClearIdMarker(frame_id_, i));
    }
    pre_pub_count = count;
    // visualization_msgs::msg::MarkerArray clear_markers;
    // clear_markers.markers.push_back(ClearMarker(frame_id_));
    // marker_publisher_->publish(clear_markers);
    marker_publisher_->publish(object_markers);
  }
  void DrawDistLabel() {
    float pose_param = sqrt(2) / 2;
    for (int i = 0; i < 6; ++i) {
      dist_markers.markers.emplace_back(
          DistCircleMarker(frame_id_, 1000 + i, 3 * (i + 1)));
      std::string text = std::to_string((i + 1) * 3) + "m";
      dist_markers.markers.emplace_back(TextMarker(
          frame_id_, 1000 + i,
          Eigen::Vector3d(pose_param * 3 * (i + 1) + 1,
                          pose_param * 3 * (i + 1) + 1, ground_height_),
          text));
    }
  }
  visualization_msgs::msg::Marker ClearMarker(const std::string frame_id) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->now();
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    return marker;
  }
  visualization_msgs::msg::Marker DistCircleMarker(const std::string frame_id,
                                                   const int id,
                                                   const float dist) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->now();
    marker.ns = "distance";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // line
    marker.pose.orientation.w = std::cos(ground_pitch_ * 0.5 * M_PI / 180);
    marker.pose.orientation.y = std::sin(ground_pitch_ * 0.5 * M_PI / 180);

    marker.scale.x = 0.1;  // 线宽
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 0.5;  // 不透明

    // 生成圆形顶点
    double radius = dist;
    int num_points = 50;  // 离散点数量
    for (int i = 0; i <= num_points; ++i) {
      double theta = 2.0 * M_PI * i / num_points;
      geometry_msgs::msg::Point p;
      p.x = radius * cos(theta);
      p.y = radius * sin(theta);
      p.z = ground_height_;
      marker.points.push_back(p);
    }

    // 闭合路径（连接首尾点）
    return marker;
  }
  visualization_msgs::msg::Marker TextMarker(const std::string frame_id,
                                             const int id,
                                             const Eigen::Vector3d& pose,
                                             const std::string& text) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->now();
    marker.ns = "text";
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.id = id;

    marker.color.r = 1;
    marker.color.g = 1;
    marker.color.b = 1;
    marker.color.a = 0.5;

    marker.scale.x = 1;
    marker.scale.y = 1;
    marker.scale.z = 1;
    marker.pose.position.x = pose[0];
    marker.pose.position.y = pose[1];
    marker.pose.position.z = pose[2];
    marker.text = text;

    return marker;
  }
#ifdef DELETE
#undef DELETE  // 取消 DELETE 宏定义
#endif
  visualization_msgs::msg::Marker ClearIdMarker(const std::string frame_id,
                                                const int id) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.id = id;
    marker.ns = "box3d";
    marker.header.stamp = this->now();
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    marker.color.r = 0;
    marker.color.g = 0;
    marker.color.b = 0;
    marker.color.a = 0;

    return marker;
  }
  visualization_msgs::msg::Marker ObjectInnerToLineMarker(
      const perception::ObjectInner& obj, const int id,
      const cv::Scalar& rgb_color_a, const std::string frame_id) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->now();
    marker.ns = "box3d";
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.id = id;

    marker.color.r = rgb_color_a[0] / 255.0;
    marker.color.g = rgb_color_a[1] / 255.0;
    marker.color.b = rgb_color_a[2] / 255.0;
    marker.color.a = rgb_color_a[3];

    Eigen::Vector3d dir =
        Eigen::Vector3d(std::cos(obj.yaw_base), std::sin(obj.yaw_base), 0);
    std::vector<Eigen::Vector3d> corners;
    corners.resize(8, Eigen::Vector3d::Zero());

    Eigen::Vector3d ortho_dir = Eigen::Vector3d(-dir.y(), dir.x(), 0);
    Eigen::Vector3d z_dir = Eigen::Vector3d(0, 0, 1);
    Eigen::Vector3d center = Eigen::Vector3d(
        obj.box_center_base.x, obj.box_center_base.y, obj.box_center_base.z);

    corners[0] = center + dir * obj.box_size.length * 0.5 +
                 ortho_dir * obj.box_size.width * 0.5 -
                 z_dir * obj.box_size.height * 0.5;
    corners[1] = center - dir * obj.box_size.length * 0.5 +
                 ortho_dir * obj.box_size.width * 0.5 -
                 z_dir * obj.box_size.height * 0.5;
    corners[2] = center - dir * obj.box_size.length * 0.5 -
                 ortho_dir * obj.box_size.width * 0.5 -
                 z_dir * obj.box_size.height * 0.5;
    corners[3] = center + dir * obj.box_size.length * 0.5 -
                 ortho_dir * obj.box_size.width * 0.5 -
                 z_dir * obj.box_size.height * 0.5;

    corners[4] = center + dir * obj.box_size.length * 0.5 +
                 ortho_dir * obj.box_size.width * 0.5 +
                 z_dir * obj.box_size.height * 0.5;
    corners[5] = center - dir * obj.box_size.length * 0.5 +
                 ortho_dir * obj.box_size.width * 0.5 +
                 z_dir * obj.box_size.height * 0.5;
    corners[6] = center - dir * obj.box_size.length * 0.5 -
                 ortho_dir * obj.box_size.width * 0.5 +
                 z_dir * obj.box_size.height * 0.5;
    corners[7] = center + dir * obj.box_size.length * 0.5 -
                 ortho_dir * obj.box_size.width * 0.5 +
                 z_dir * obj.box_size.height * 0.5;

    std::vector<geometry_msgs::msg::Point> pts(8);
    for (int i = 0; i < 8; i++) {
      pts[i].x = corners[i].x();
      pts[i].y = corners[i].y();
      pts[i].z = corners[i].z();
    }

    marker.points.push_back(pts[0]);
    marker.points.push_back(pts[1]);
    marker.points.push_back(pts[1]);
    marker.points.push_back(pts[2]);
    marker.points.push_back(pts[2]);
    marker.points.push_back(pts[3]);
    marker.points.push_back(pts[3]);
    marker.points.push_back(pts[0]);

    marker.points.push_back(pts[4]);
    marker.points.push_back(pts[5]);
    marker.points.push_back(pts[5]);
    marker.points.push_back(pts[6]);
    marker.points.push_back(pts[6]);
    marker.points.push_back(pts[7]);
    marker.points.push_back(pts[7]);
    marker.points.push_back(pts[4]);

    marker.points.push_back(pts[0]);
    marker.points.push_back(pts[4]);
    marker.points.push_back(pts[1]);
    marker.points.push_back(pts[5]);
    marker.points.push_back(pts[2]);
    marker.points.push_back(pts[6]);
    marker.points.push_back(pts[3]);
    marker.points.push_back(pts[7]);

    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    return marker;
  }
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      lidar_subscription_;
  rclcpp::Subscription<perception_msgs::msg::ObjectInnerArray>::SharedPtr
      object_subscription_;

  rclcpp::Publisher<perception_msgs::msg::ObjectInnerArray>::SharedPtr
      box_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      debug_point_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_publisher_;

  int max_cache_size_ = 10;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> image_msg_cache_;
  std::deque<std::pair<int64_t, pcl::PointCloud<pcl::PointXYZ>::Ptr>>
      lidar_msg_cache_;

  int64_t time_diff_thresh_ = 50 * 1e6;
  int pre_pub_count = 0;

  perception::MsgInterface::Ptr msg_interface_;
  perception::PvPostProcess::Ptr task_ptr_;
  perception::TimeRecord pv_post_process_record_;
  std::string frame_id_ = "base_link";

  visualization_msgs::msg::MarkerArray dist_markers;

  bool ground_update = false;
  float ground_height_ = 0;
  float ground_pitch_ = 0;
};

}  // namespace robosense
#endif