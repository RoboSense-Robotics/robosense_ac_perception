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

#ifndef PERCEPTION_YOLOV8_PREPROCESS_H
#define PERCEPTION_YOLOV8_PREPROCESS_H

#include "perception/common/common.h"
#include "perception/yolov8/params.h"

namespace robosense {
namespace perception {

int get_image_size(image_buffer_t* image);

int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, LetterBoxInfo* letterbox, char color);

} // namespace robosense
} // namespace perception
#endif // PERCEPTION_YOLOV8_PREPROCESS_H