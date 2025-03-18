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

#include "perception/yolov8/config.h"
#include <fstream>
#include <unistd.h>  // Linux/macOS
#ifdef _WIN32
#include <direct.h> // Windows（需包含此头文件）
#endif
namespace robosense {
namespace perception {

void Yolov8DetConfig::Init(const YAML::Node &cfg_node) {
    std::cout << Name() << " init! " << std::endl;
    rally::yamlRead(cfg_node, "sub_image_topic", sub_topic);
    char buffer[256];
    if (getcwd(buffer, sizeof(buffer)) != nullptr) {
        std::cout << "current work dir is: " << buffer << std::endl;
    }
#ifdef __aarch64__
    rally::yamlRead(cfg_node, "arm_model", model_path);
#else // x86
    rally::yamlRead(cfg_node, "x86_model", model_path);
#endif
    model_path = std::string(buffer) + "/" + model_path;
    std::ifstream ifs(model_path);
    if (!ifs.is_open()) {
        std::cout << "model path error!" << std::endl;
    } else {
        std::cout << "model path is: " << model_path << std::endl;
        ifs.close();
    }
    rally::yamlRead(cfg_node, "nms_threshold", nms_threshold);
    rally::yamlRead(cfg_node, "box_conf_threshold", box_conf_threshold);
    rally::yamlRead(cfg_node, "nn_input_width", nn_input_width);
    rally::yamlRead(cfg_node, "nn_input_height", nn_input_height);
    rally::yamlRead(cfg_node, "save_img", save_img);
}

} // namespace robosense
} // namespace perception