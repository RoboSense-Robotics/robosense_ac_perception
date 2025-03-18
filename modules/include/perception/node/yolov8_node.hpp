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

#include "perception/interface.h"
#ifdef ENABLE_TENSORRT
#include "perception/yolov8/yolov8_det_nn_trt.h"
#endif
#ifdef ENABLE_RKNN
#include "perception/yolov8/yolov8_det_nn_rk.h"
#endif
#include "rclcpp/rclcpp.hpp"
#include <opencv2/opencv.hpp>
#include "perception_msgs/msg/object_inner.hpp"
#include "perception_msgs/msg/object_inner_array.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
namespace robosense {

struct Yolov8NodeOptions {
  std::string sub_image_topic;
  YAML::Node yolov8_cfg_node;
  void Load(const YAML::Node& cfg_node) {
    rally::yamlRead(cfg_node, "sub_image_topic", sub_image_topic);
    rally::yamlSubNode(cfg_node, "yolov8", yolov8_cfg_node);
  }
};

class Yolov8Node : public rclcpp::Node {
public:
  Yolov8Node(const Yolov8NodeOptions& options) : rclcpp::Node("yolov8_node") {
    options_ = options;
    // Subscribe to the image topic
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        options_.sub_image_topic, 10, std::bind(&Yolov8Node::ImageCallback, this, std::placeholders::_1));

    perception::PerceptionInterfaceOptions detection_options;
    detection_options.Load(options_.yolov8_cfg_node);
    detection_ptr_.reset(new perception::PerceptionInterface(detection_options));

    // pub
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("yolov8", 10);
    box_publisher_=this->create_publisher<perception_msgs::msg::ObjectInnerArray>("yolov8_object", 10);
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

      detection_ptr_->Process(msg_ptr);
      auto out_msg_ptr = cv_bridge::CvImage(msg->header, "bgr8", msg_ptr->output_msg_ptr->mat).toImageMsg();
      publisher_->publish(*out_msg_ptr);
      perception_msgs::msg::ObjectInnerArray ros_msg;
      ObjectInnerArrayToRosMsg(msg_ptr->output_msg_ptr->object_list_ptr,
                               ros_msg);
      ros_msg.header = msg->header;
      box_publisher_->publish(ros_msg);
    } catch (std::exception &e) {
      std::cout << e.what();
    }
  }
  void ObjectInnerArrayToRosMsg(const perception::ObjectInnerArray::Ptr& output_msg,
                               perception_msgs::msg::ObjectInnerArray& ros_msg) {
    ros_msg.header.frame_id = output_msg->header.frame_id;
    ros_msg.header.stamp = rclcpp::Time(double(output_msg->header.time)/1e9);
    // ros_msg.header.seq = output_msg->header.seq;

    ros_msg.object_list.resize(output_msg->object_list.size());
    for (size_t i = 0; i < output_msg->object_list.size();++i){
        auto& ros_obj=ros_msg.object_list[i];
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
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<perception_msgs::msg::ObjectInnerArray>::SharedPtr box_publisher_;

  Yolov8NodeOptions options_;
  perception::PerceptionInterface::Ptr detection_ptr_;
};

}  // namespace robosense

#endif  // PERCEPTION_NODE_YOLOV8_NODE_H
