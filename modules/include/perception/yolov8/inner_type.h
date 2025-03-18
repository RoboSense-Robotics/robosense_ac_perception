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

#ifndef PERCEPTION_YOLOV8_INNER_TYPE_H
#define PERCEPTION_YOLOV8_INNER_TYPE_H

#include <vector>

namespace robosense {
namespace perception {

#define OBJ_NAME_MAX_SIZE_IN 64
#define OBJ_CLASS_NUM_IN 80
constexpr int OBJ_NUMB_MAX_SIZE_IN = 128;

enum class image_format_t{
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_YUV420SP_NV21,
    IMAGE_FORMAT_YUV420SP_NV12,
};

struct image_buffer_t{
    int width;
    int height;
    int width_stride;
    int height_stride;
    image_format_t format;
    unsigned char* virt_addr;
    int size;
    int fd;
};

struct Bbox{
    int left{0};
    int top{0};
    int right{0};
    int bottom{0};
} ;

struct LetterBoxInfo {
  int x_pad{0};
  int y_pad{0};
  float scale{0};
};

struct DetObject {
  Bbox box;
  float prop{0};
  int cls_id{0};
};

struct ObjectList {
  ObjectList() {
      results.resize(OBJ_NUMB_MAX_SIZE_IN);
  }
  ~ObjectList() {
      results.clear();
  }
  int id{0};
  int count{0};
  std::vector<DetObject> results;
};
} // namespace robosense
} // namespace perception

#endif // PERCEPTION_YOLOV8_INNER_TYPE_H



