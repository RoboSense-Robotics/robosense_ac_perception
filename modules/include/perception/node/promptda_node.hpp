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

#include "perception/interface.h"
#ifdef ENABLE_TENSORRT
#include "perception/promptda/promptda_nn_trt.h"
#endif
#ifdef ENABLE_RKNN
#include "perception/promptda/promptda_nn_rk.h"
#endif
#include "rclcpp/rclcpp.hpp"
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
namespace robosense {

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

class PromptDANode : public rclcpp::Node {
public:
  PromptDANode(const PromptDANodeOptions& options) : rclcpp::Node("promptda_node") {
    options_ = options;
    // Subscribe to the image topic
    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        options_.sub_image_topic, 10, std::bind(&PromptDANode::ImageCallback, this, std::placeholders::_1));
    lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            options_.sub_lidar_topic, 10, std::bind(&PromptDANode::LidarCallback, this, std::placeholders::_1));
    perception::PerceptionInterfaceOptions promptda_options;
    promptda_options.Load(options_.promptda_cfg_node);
    promptda_ptr_.reset(new perception::PerceptionInterface(promptda_options));

    // pub
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("promptda", 10);
  }

private:
  void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    try {
      // Convert ROS image message to OpenCV format
      // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
      // const auto& image = cv_ptr->image;
      cv::Mat image(cv::Size(msg->width, msg->height), CV_8UC3, reinterpret_cast<void*>(const_cast<unsigned char*>(msg->data.data())));

      perception::DetectionMsg::Ptr msg_ptr(new perception::DetectionMsg);
      msg_ptr->input_msg_ptr.reset(new perception::DetectionInputMsg);
      msg_ptr->output_msg_ptr.reset(new perception::DetectionOutputMsg);
      perception::Image tmp_image;
      tmp_image.timestamp = int64_t(msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec);
      tmp_image.mat = image;
      msg_ptr->input_msg_ptr->camera_data_map[options_.sub_image_topic] = tmp_image;
      //lidar
      auto cloud_ptr = FindNearestLidarMsg(tmp_image.timestamp);
      int circle = 0;
      while(cloud_ptr == nullptr && circle < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        cloud_ptr = FindNearestLidarMsg(tmp_image.timestamp);
        circle++;
      }
      if (cloud_ptr == nullptr) {
        std::cout << "no match lidar msg found!" << std::endl;
        return;
      } else {
        msg_ptr->input_msg_ptr->cloud_ptr = cloud_ptr;
        std::cout << " find match lidar and process" << std::endl
                  << "image time stamp: " << tmp_image.timestamp << std::endl
                  << "lidar time stamp: " << cloud_ptr->timestamp << std::endl
                  << "diff: " << (tmp_image.timestamp - cloud_ptr->timestamp)/1e6 << " ms" << std::endl;
      }

      promptda_ptr_->Process(msg_ptr);
      auto out_msg_ptr = cv_bridge::CvImage(msg->header, "bgr8", msg_ptr->output_msg_ptr->mat).toImageMsg();
      publisher_->publish(*out_msg_ptr);
    } catch (std::exception &e) {
      std::cout << e.what();
    }
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
    perception::PointCloud::Ptr cloud_ptr;
    cloud_ptr.reset(new perception::PointCloud());
    cloud_ptr->timestamp = timestamp;
    cloud_ptr->cloud = ori_cloud;
    lidar_msg_cache_.emplace_back(cloud_ptr);
  }

  perception::PointCloud::Ptr FindNearestLidarMsg(const int64_t timestamp) {
    perception::PointCloud::Ptr nearest_msg = nullptr;
    int64_t min_diff_time = options_.time_diff_thresh * 2;
    for (const auto& msg : lidar_msg_cache_) {
      auto time_diff = std::abs(timestamp - msg->timestamp);
      if (time_diff < options_.time_diff_thresh && time_diff < min_diff_time) {
        min_diff_time = time_diff;
        nearest_msg = msg;
      }
    }
    return nearest_msg;
  }

  int max_cache_size_ = 10;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> image_msg_cache_;
  std::deque<perception::PointCloud::Ptr> lidar_msg_cache_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

  PromptDANodeOptions options_;
  perception::PerceptionInterface::Ptr promptda_ptr_;
};

}  // namespace robosense

#endif  // PERCEPTION_NODE_promptda_NODE_H
