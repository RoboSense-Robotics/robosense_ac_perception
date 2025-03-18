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

#ifndef PERCEPTION_PPSEG_PARAMS_H
#define PERCEPTION_PPSEG_PARAMS_H

#include "perception/ppseg/msg.h"
#include "perception/ppseg/config.h"
#include "perception/ppseg/model_attr.h"

namespace robosense {
namespace perception {

class PPSegParams {
public:
    using Ptr = std::shared_ptr<PPSegParams>;

    PPSegParams() {
        config_ptr_ = std::make_unique<PPSegConfig>();
        msg_ptr_ = std::make_unique<PPSegMsg>();
        model_attr_ptr_ = std::make_unique<PPSegModelAttr>();
    }

    PPSegConfig::Ptr config_ptr_;
    PPSegMsg::Ptr msg_ptr_;
    PPSegModelAttr::Ptr model_attr_ptr_;
private:
    static std::string Name() { return "PPSegParams"; }
};

} // namespace robosense
} // namespace perception

#endif // PERCEPTION_PPSEG_PARAMS_H
