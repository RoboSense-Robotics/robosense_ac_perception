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

#ifndef PERCEPTION_PROMPTDA_PARAMS_H
#define PERCEPTION_PROMPTDA_PARAMS_H

#include "perception/promptda/msg.h"
#include "perception/promptda/config.h"
#include "perception/promptda/model_attr.h"

namespace robosense {
namespace perception {

class PromptDAParams {
public:
    using Ptr = std::shared_ptr<PromptDAParams>;

    PromptDAParams() {
        config_ptr_ = std::make_unique<PromptDAConfig>();
        msg_ptr_ = std::make_unique<PromptDAMsg>();
        model_attr_ptr_ = std::make_unique<PromptDAModelAttr>();
    }

    PromptDAConfig::Ptr config_ptr_;
    PromptDAMsg::Ptr msg_ptr_;
    PromptDAModelAttr::Ptr model_attr_ptr_;
private:
    static std::string Name() { return "PromptDAParams"; }
};

} // namespace robosense
} // namespace perception

#endif // PERCEPTION_PROMPTDA_PARAMS_H
