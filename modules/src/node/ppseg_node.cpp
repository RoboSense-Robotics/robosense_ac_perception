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
#include "perception/node/ppseg_node.hpp"
namespace robosense {
namespace perception {
PPSegNode::PPSegNode(const PPSegNodeOptions& options) {
  options_ = options;
#if defined(USE_ROS1)
  subscription_ = nh_.subscribe(options_.sub_image_topic, 10, &PPSegNode::ImageCallback, this);
  publisher_ = nh_.advertise<sensor_msgs::Image>("ppseg", 10);
#elif defined(USE_ROS2)
  const char* val = std::getenv("RMW_FASTRTPS_USE_QOS_FROM_XML");
  if (val != nullptr && std::string(val) == "1") {
    zero_copy_ = true;
  } else {
    zero_copy_ = false;
  }
  ros2_node_ptr_ = rclcpp::Node::make_shared("ppseg_node");
  if (zero_copy_) {
    subscription_zc_ = ros2_node_ptr_->create_subscription<robosense_msgs::msg::RsImage>(
      options_.sub_image_topic, 10, std::bind(&PPSegNode::ZCImageCallback, this, std::placeholders::_1));
  } else {
    subscription_ = ros2_node_ptr_->create_subscription<sensor_msgs::msg::Image>(
      options_.sub_image_topic, 10, std::bind(&PPSegNode::ImageCallback, this, std::placeholders::_1));
  }
  publisher_ = ros2_node_ptr_->create_publisher<sensor_msgs::msg::Image>("ppseg", 10);
#endif
  perception::PerceptionInterfaceOptions seg_options;
  seg_options.Load(options_.ppseg_cfg_node);
  seg_ptr_.reset(new perception::PerceptionInterface(seg_options));
  Start();
}
void PPSegNode::Start() {
  run_flag_ = true;
  if (thread_ptr_ == nullptr) {
    thread_ptr_ = std::make_unique<std::thread>(&PPSegNode::Core, this);
  }
  std::cout << Name() <<": start!" << std::endl;
}

void PPSegNode::Stop() {
  if (thread_ptr_ != nullptr) {
    run_flag_ = false;
    std::cout << Name() << ": stoped!" << std::endl;
    if (thread_ptr_->joinable()) {
      thread_ptr_->join();
    }
    thread_ptr_.reset(nullptr);
  }
}
void PPSegNode::ImageCallback(const ImageMsgsConstPtr msg) {
  try {
    // Convert ROS image message to OpenCV format
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    const auto& image = cv_ptr->image;
    perception::DetectionMsg::Ptr msg_ptr(new perception::DetectionMsg);
    msg_ptr->input_msg_ptr.reset(new perception::DetectionInputMsg);
    msg_ptr->output_msg_ptr.reset(new perception::DetectionOutputMsg);
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
void PPSegNode::ZCImageCallback(const robosense_msgs::msg::RsImage::Ptr msg) {
  try {
    cv::Mat image(cv::Size(msg->width, msg->height), CV_8UC3, reinterpret_cast<void*>(const_cast<unsigned char*>(msg->data.data())));
    perception::DetectionMsg::Ptr msg_ptr(new perception::DetectionMsg);
    msg_ptr->input_msg_ptr.reset(new perception::DetectionInputMsg);
    msg_ptr->output_msg_ptr.reset(new perception::DetectionOutputMsg);
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
    tmp_image.mat = image.clone();
    msg_ptr->input_msg_ptr->camera_data_map[options_.sub_image_topic] = tmp_image;
    msg_ptr->header = header;
    msg_queue_.push(msg_ptr);
  } catch (std::exception &e) {
    std::cout << e.what();
  }
}
#endif
void PPSegNode::Core() {
  while (run_flag_) {
    auto msg_ptr = msg_queue_.pop();
    seg_ptr_->Process(msg_ptr);
    auto out_msg_ptr = cv_bridge::CvImage(msg_ptr->header, "bgr8", msg_ptr->output_msg_ptr->mat).toImageMsg();
#if defined(USE_ROS1)
    publisher_.publish(*out_msg_ptr);
#elif defined(USE_ROS2)
    publisher_->publish(*out_msg_ptr);
#endif
  }
}

} // namespace perception
} // namespace robosense

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
  YAML::Node ppseg_node;
  rally::yamlSubNode(cfg_node, "ppseg_node", ppseg_node);

  robosense::perception::PPSegNodeOptions options;
  options.Load(ppseg_node);

#if defined(USE_ROS1)
  ros::init(argc, argv, "ppseg_node");
#elif defined(USE_ROS2)
  rclcpp::init(argc, argv);
#endif
  auto node = std::make_shared<robosense::perception::PPSegNode>(options);
#if defined(USE_ROS1)
  ros::spin();
#elif defined(USE_ROS2)
  rclcpp::spin(node->GetRos2Node());
  rclcpp::shutdown();
#endif
}