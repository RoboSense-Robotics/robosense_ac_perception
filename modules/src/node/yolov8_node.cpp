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
#include "perception/common/yaml.h"
#include "perception/node/yolov8_node.hpp"

namespace robosense {
namespace perception {
Yolov8Node::Yolov8Node(const Yolov8NodeOptions& options) {
  options_ = options;
#if defined(USE_ROS1)
  subscription_ = nh_.subscribe(options_.sub_image_topic, 100, &Yolov8Node::ImageCallback, this);
  publisher_ = nh_.advertise<sensor_msgs::Image>("yolov8", 100);
  box_publisher_ = nh_.advertise<ROSObjectInnerArray>("yolov8_object", 10);
#elif defined(USE_ROS2)
  const char* val = std::getenv("RMW_FASTRTPS_USE_QOS_FROM_XML");
  if (val != nullptr && std::string(val) == "1") {
    zero_copy_ = true;
  } else {
    zero_copy_ = false;
  }
  ros2_node_ptr_ = rclcpp::Node::make_shared("yolov8_node");
  if (zero_copy_) {
    subscription_zc_ = ros2_node_ptr_->create_subscription<robosense_msgs::msg::RsImage>(
      options_.sub_image_topic, 10, std::bind(&Yolov8Node::ZCImageCallback, this, std::placeholders::_1));
  } else {
    subscription_ = ros2_node_ptr_->create_subscription<sensor_msgs::msg::Image>(
      options_.sub_image_topic, 10, std::bind(&Yolov8Node::ImageCallback, this, std::placeholders::_1));
  }
  publisher_ = ros2_node_ptr_->create_publisher<sensor_msgs::msg::Image>("yolov8", 10);
  box_publisher_=ros2_node_ptr_->create_publisher<ROSObjectInnerArray>("yolov8_object", 10);
#endif
  perception::PerceptionInterfaceOptions detection_options;
  detection_options.Load(options_.yolov8_cfg_node);
  detection_ptr_.reset(new perception::PerceptionInterface(detection_options));
  Start();
}

void Yolov8Node::Start() {
  run_flag_ = true;
  if (thread_ptr_ == nullptr) {
    thread_ptr_ = std::make_unique<std::thread>(&Yolov8Node::Core, this);
  }
  std::cout << Name() <<": start!" << std::endl;
}
void Yolov8Node::Stop() {
  if (thread_ptr_ != nullptr) {
    run_flag_ = false;
    std::cout << Name() << ": stoped!" << std::endl;
    if (thread_ptr_->joinable()) {
      thread_ptr_->join();
    }
    thread_ptr_.reset(nullptr);
  }
}

void Yolov8Node::Core() {
  while (run_flag_) {
    auto msg_ptr = msg_queue_.pop();
    detection_ptr_->Process(msg_ptr);
    auto out_msg_ptr = cv_bridge::CvImage(msg_ptr->header, "bgr8", msg_ptr->output_msg_ptr->mat).toImageMsg();
#if defined(USE_ROS1)
    publisher_.publish(*out_msg_ptr);
#elif defined(USE_ROS2)
    publisher_->publish(*out_msg_ptr);
#endif
    ROSObjectInnerArray ros_msg;
    ObjectInnerArrayToRosMsg(msg_ptr->output_msg_ptr->object_list_ptr,
                             ros_msg);
    ros_msg.header = msg_ptr->header;
#if defined(USE_ROS1)
  box_publisher_.publish(ros_msg);
#elif defined(USE_ROS2)
  box_publisher_->publish(ros_msg);
#endif
  }
}

void Yolov8Node::ImageCallback(const ImageMsgsConstPtr msg) {
  try {
    // Convert ROS image message to OpenCV format
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    const auto& image = cv_ptr->image;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // cv::Mat image(cv::Size(msg->width, msg->height), CV_8UC3, reinterpret_cast<void*>(const_cast<unsigned char*>(msg->data.data())));
    perception::DetectionMsg::Ptr msg_ptr(new perception::DetectionMsg);
    msg_ptr->input_msg_ptr.reset(new perception::DetectionInputMsg);
    msg_ptr->output_msg_ptr.reset(new perception::DetectionOutputMsg);
    msg_ptr->output_msg_ptr->object_list_ptr.reset(new perception::ObjectInnerArray);
    perception::Image tmp_image;
    tmp_image.timestamp = int64_t(HeaderToNanoSec(msg->header));
    tmp_image.mat = image;
    msg_ptr->input_msg_ptr->camera_data_map[options_.sub_image_topic] = tmp_image;
    msg_ptr->header = msg->header;
    msg_queue_.push(msg_ptr);
  } catch (std::exception &e) {
    std::cout << e.what();
  }
}
#if defined(USE_ROS2)
void Yolov8Node::ZCImageCallback(const robosense_msgs::msg::RsImage::Ptr msg) {
  try {
    // Convert ROS image message to OpenCV format
    // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    // const auto& image = cv_ptr->image;
    cv::Mat image(cv::Size(msg->width, msg->height), CV_8UC3, reinterpret_cast<void*>(const_cast<unsigned char*>(msg->data.data())));
    perception::DetectionMsg::Ptr msg_ptr(new perception::DetectionMsg);
    msg_ptr->input_msg_ptr.reset(new perception::DetectionInputMsg);
    msg_ptr->output_msg_ptr.reset(new perception::DetectionOutputMsg);
    msg_ptr->output_msg_ptr->object_list_ptr.reset(new perception::ObjectInnerArray);
    perception::Image tmp_image;
    std_msgs::msg::Header header;
    std::string frame_id;
    for(char c : msg->header.frame_id){
        if(c == '\0')break;
        frame_id.push_back(c);
    }
    header.frame_id = frame_id;
    header.stamp = rclcpp::Time(msg->header.stamp);
    tmp_image.timestamp = int64_t(HeaderToNanoSec(header));
    tmp_image.mat = image;
    msg_ptr->input_msg_ptr->camera_data_map[options_.sub_image_topic] = tmp_image;
    msg_ptr->header = header;
    msg_queue_.push(msg_ptr);
  } catch (std::exception &e) {
    std::cout << e.what();
  }
}
#endif
void Yolov8Node::ObjectInnerArrayToRosMsg(const perception::ObjectInnerArray::Ptr& output_msg, ROSObjectInnerArray& ros_msg) {
  ros_msg.header.frame_id = output_msg->header.frame_id;
#if defined(USE_ROS1)
  ros_msg.header.stamp = ros::Time(double(output_msg->header.time)/1e9);
#elif defined(USE_ROS2)
  ros_msg.header.stamp = rclcpp::Time(double(output_msg->header.time)/1e9);
#endif

  ros_msg.object_list.resize(output_msg->object_list.size());
  for (size_t i = 0; i < output_msg->object_list.size();++i) {
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

} // namespace perception
}  // namespace robosense

std::string parseConfigOption(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config") {
            if (i + 1 < argc) {
                return argv[i + 1];
            } else {
                std::cerr << "--config need a config file" << std::endl;
                return "";
            }
        }
    }
    return "modules/config/usr_config.yaml";
}

int main(int argc, char **argv) {
  // std::string config_file = std::string(PROJECT_PATH) + "/config/usr_config.yaml";
  std::string config_file = parseConfigOption(argc, argv);
  std::cout << config_file << std::endl;
  YAML::Node cfg_node;
  try {
    cfg_node = YAML::LoadFile(config_file);
  } catch (std::exception &e) {
    std::string error_msg(e.what());
    if (error_msg == "bad file") {
      std::cout << "yaml file do not exist! " << config_file;
    } else {
        std::cout << error_msg;
    }
    return false;
  }
  YAML::Node yolov8_node;
  rally::yamlSubNode(cfg_node, "yolov8_node", yolov8_node);

  robosense::perception::Yolov8NodeOptions options;
  options.Load(yolov8_node);
#if defined(USE_ROS1)
  ros::init(argc, argv, "yolov8_node");
#elif defined(USE_ROS2)
  rclcpp::init(argc, argv);
#endif
  auto node = std::make_shared<robosense::perception::Yolov8Node>(options);
#if defined(USE_ROS1)
  ros::spin();
#elif defined(USE_ROS2)
  rclcpp::spin(node->GetRos2Node());
  rclcpp::shutdown();
#endif
  return 0;
}