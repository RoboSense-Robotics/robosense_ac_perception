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
#ifndef PERCEPTION_YOLOV8_YAML_H
#define PERCEPTION_YOLOV8_YAML_H

#include <iostream>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace rally {

template<typename T>
inline bool yamlRead(const YAML::Node &node, const std::string &key, T &val) {
    try {
        val = node[key].as<T>();
    } catch (std::exception &e) {
        return false;
    }
    return true;
}

inline bool yamlSubNode(const YAML::Node &node, const std::string &key, YAML::Node &ret) {
    try {
        ret = node[key];
    } catch (std::exception &e) {
        return false;
    }
    return true;
}

inline bool loadFile(const std::string &yaml_file, YAML::Node &node) {
    try {
        node = YAML::LoadFile(yaml_file);
    } catch (std::exception &e) {
        std::string error_msg(e.what());
        if (error_msg == "bad file") {
            std::cout << "yaml file do not exist! " << yaml_file;
        } else {
            std::cout << "-- YAML Load Error: ";
            std::cout << "-- In:\n\t" << yaml_file;
            std::cout << "-- What:\n\t" << error_msg;
        }
        return false;
    }

    return true;
}

}  // namespace rally

#endif  // PERCEPTION_YOLOV8_YAML_H
