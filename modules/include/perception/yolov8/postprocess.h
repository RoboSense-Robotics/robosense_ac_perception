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

#ifndef PERCEPTION_YOLOV8_POSTPROCESS_H
#define PERCEPTION_YOLOV8_POSTPROCESS_H

#include <vector>

#include "perception/yolov8/params.h"

namespace robosense {
namespace perception {

int post_process(Yolov8DetParams::Ptr params_ptr, std::vector<void*> outputs, LetterBoxInfo *letter_box, ObjectList *od_results);
void post_process_hbdnn(Yolov8DetParams::Ptr params_ptr);
} // namespace robosense
} // namespace perception
#endif // PERCEPTION_YOLOV8_POSTPROCESS_H