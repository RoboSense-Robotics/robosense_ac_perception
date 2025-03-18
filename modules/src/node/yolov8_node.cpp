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

  robosense::Yolov8NodeOptions options;
  options.Load(yolov8_node);

  rclcpp::init(argc, argv);
  auto node = std::make_shared<robosense::Yolov8Node>(options);
  RCLCPP_INFO(rclcpp::get_logger("global_logger"), "Image Subscriber Node Started");
  rclcpp::spin(node);
  rclcpp::shutdown();
}