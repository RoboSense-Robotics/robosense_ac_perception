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

#include "perception/interface.h"
#ifdef ENABLE_TENSORRT
#include "perception/ppseg/ppseg_nn_trt.h"
#endif
#ifdef ENABLE_RKNN
#include "perception/ppseg/ppseg_nn_rk.h"
#endif
#include "rclcpp/rclcpp.hpp"
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
namespace robosense {

struct PPSegNodeOptions {
  std::string sub_image_topic;
  YAML::Node ppseg_cfg_node;
  void Load(const YAML::Node& cfg_node) {
    rally::yamlRead(cfg_node, "sub_image_topic", sub_image_topic);
    rally::yamlSubNode(cfg_node, "ppseg", ppseg_cfg_node);
  }
};

class PPSegNode : public rclcpp::Node {
public:
  PPSegNode(const PPSegNodeOptions& options) : rclcpp::Node("ppseg_node") {
    options_ = options;
    // Subscribe to the image topic
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        options_.sub_image_topic, 10, std::bind(&PPSegNode::ImageCallback, this, std::placeholders::_1));

    perception::PerceptionInterfaceOptions seg_options;
    seg_options.Load(options_.ppseg_cfg_node);
    seg_ptr_.reset(new perception::PerceptionInterface(seg_options));

    // pub
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("ppseg", 10);
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
      msg_ptr->output_msg_ptr->object_list_ptr.reset(new perception::ObjectInnerArray);
      perception::Image tmp_image;
      tmp_image.timestamp = uint64_t(msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec);
      tmp_image.mat = image;
      msg_ptr->input_msg_ptr->camera_data_map[options_.sub_image_topic] = tmp_image;

      seg_ptr_->Process(msg_ptr);
      auto out_msg_ptr = cv_bridge::CvImage(msg->header, "bgr8", msg_ptr->output_msg_ptr->mat).toImageMsg();
      publisher_->publish(*out_msg_ptr);
    } catch (std::exception &e) {
      std::cout << e.what();
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

  PPSegNodeOptions options_;
  perception::PerceptionInterface::Ptr seg_ptr_;
};

}  // namespace robosense

#endif  // PERCEPTION_NODE_PPSEG_NODE_H
