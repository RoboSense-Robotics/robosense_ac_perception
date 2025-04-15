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
#include "perception/node/promptda_node.hpp"
namespace robosense {
namespace perception {
PromptDANode::PromptDANode(const PromptDANodeOptions& options) {
  options_ = options;
#if defined(USE_ROS1)
  image_subscription_ = nh_.subscribe(options_.sub_image_topic, 100, &PromptDANode::ImageCallback, this);
  lidar_subscription_ = nh_.subscribe(options_.sub_lidar_topic, 100, &PromptDANode::LidarCallback, this);
  publisher_ = nh_.advertise<sensor_msgs::Image>("promptda", 100);
#elif defined(USE_ROS2)
  const char* val = std::getenv("RMW_FASTRTPS_USE_QOS_FROM_XML");
  if (val != nullptr && std::string(val) == "1") {
    zero_copy_ = true;
  } else {
    zero_copy_ = false;
  }
  ros2_node_ptr_ = rclcpp::Node::make_shared("promptda_node");
  if (zero_copy_) {
    image_subscription_zc_ = ros2_node_ptr_->create_subscription<robosense_msgs::msg::RsImage>(
      options_.sub_image_topic, 10, std::bind(&PromptDANode::ZCImageCallback, this, std::placeholders::_1));
    lidar_subscription_zc_ = ros2_node_ptr_->create_subscription<robosense_msgs::msg::RsPointCloud>(
      options_.sub_lidar_topic, 10, std::bind(&PromptDANode::ZCLidarCallback, this, std::placeholders::_1));
  } else {
    image_subscription_ = ros2_node_ptr_->create_subscription<sensor_msgs::msg::Image>(
      options_.sub_image_topic, 10, std::bind(&PromptDANode::ImageCallback, this, std::placeholders::_1));
    lidar_subscription_ = ros2_node_ptr_->create_subscription<sensor_msgs::msg::PointCloud2>(
        options_.sub_lidar_topic, 10, std::bind(&PromptDANode::LidarCallback, this, std::placeholders::_1));
  }
  publisher_ = ros2_node_ptr_->create_publisher<sensor_msgs::msg::Image>("promptda", 10);
#endif
  perception::PerceptionInterfaceOptions promptda_options;
  promptda_options.Load(options_.promptda_cfg_node);
  promptda_ptr_.reset(new perception::PerceptionInterface(promptda_options));
  Start();
}

void PromptDANode::Start() {
  run_flag_ = true;
  if (thread_ptr_ == nullptr) {
    thread_ptr_ = std::make_unique<std::thread>(&PromptDANode::Core, this);
  }
  std::cout << Name() << ": start!" << std::endl;
}
void PromptDANode::Stop() {
  if (thread_ptr_ != nullptr) {
    run_flag_ = false;
    std::cout << Name() << ": stoped!" << std::endl;
    if (thread_ptr_->joinable()) {
      thread_ptr_->join();
    }
    thread_ptr_.reset(nullptr);
  }
}

void PromptDANode::Core() {
  while (run_flag_) {
    auto msg_ptr = msg_queue_.pop();
    auto timestamp = int64_t(HeaderToNanoSec(msg_ptr->header)); // image msg time stamp
    //lidar
    auto cloud_ptr = FindNearestLidarMsg(timestamp);
    int circle = 0;
    while(cloud_ptr == nullptr && circle < 3) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      cloud_ptr = FindNearestLidarMsg(timestamp);
      circle++;
    }
    if (cloud_ptr == nullptr) {
      std::cout << "no match lidar msg found!" << std::endl;
    } else {
      msg_ptr->input_msg_ptr->cloud_ptr = cloud_ptr;
      std::cout << " find match lidar and process" << std::endl
                << "image time stamp: " << timestamp << std::endl
                << "lidar time stamp: " << cloud_ptr->timestamp << std::endl
                << "diff: " << (timestamp - cloud_ptr->timestamp)/1e6 << " ms" << std::endl;
      promptda_ptr_->Process(msg_ptr);
      auto out_msg_ptr = cv_bridge::CvImage(msg_ptr->header, "bgr8", msg_ptr->output_msg_ptr->mat).toImageMsg();
#if defined(USE_ROS1)
      publisher_.publish(*out_msg_ptr);
#elif defined(USE_ROS2)
      publisher_->publish(*out_msg_ptr);
#endif
    }
  }
}

void PromptDANode::ImageCallback(const ImageMsgsConstPtr msg) {
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

void PromptDANode::LidarCallback(const PointCloud2MsgsConstPtr msg) {
  int64_t timestamp = int64_t(HeaderToNanoSec(msg->header));
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

#if defined(USE_ROS2)
void PromptDANode::ZCImageCallback(const robosense_msgs::msg::RsImage::Ptr msg) {
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
void PromptDANode::ZCLidarCallback(const robosense_msgs::msg::RsPointCloud::Ptr zc_msg) {
  // convert to ros2 point msg
  PointCloud2MsgsConstPtr msg = ConvertZCPoints(zc_msg);
  int64_t timestamp = int64_t(HeaderToNanoSec(msg->header));
  std::cout << "Get lidar msg with timestamp: " << timestamp << std::endl;
  if (lidar_msg_cache_.size() >= max_cache_size_) {
    lidar_msg_cache_.pop_front();
  }
  // only xyz
  pcl::PointCloud<pcl::PointXYZ>::Ptr ori_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*msg, *ori_cloud);
  perception::PointCloud::Ptr cloud_ptr;
  cloud_ptr.reset(new perception::PointCloud());
  cloud_ptr->timestamp = timestamp;
  cloud_ptr->cloud = ori_cloud;
  lidar_msg_cache_.emplace_back(cloud_ptr);
}
#endif

perception::PointCloud::Ptr PromptDANode::FindNearestLidarMsg(const int64_t timestamp) {
  std::cout << "lidar msg cache: " <<lidar_msg_cache_.size() << std::endl;
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
  YAML::Node promptda_node;
  rally::yamlSubNode(cfg_node, "promptda_node", promptda_node);

  robosense::perception::PromptDANodeOptions options;
  options.Load(promptda_node);

#if defined(USE_ROS1)
  ros::init(argc, argv, "promptda_node");
#elif defined(USE_ROS2)
  rclcpp::init(argc, argv);
#endif
  auto node = std::make_shared<robosense::perception::PromptDANode>(options);
#if defined(USE_ROS1)
  ros::spin();
#elif defined(USE_ROS2)
  rclcpp::spin(node->GetRos2Node());
  rclcpp::shutdown();
#endif
}