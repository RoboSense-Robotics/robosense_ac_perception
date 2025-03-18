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
#ifndef PERCEPTION_COMMON_BASIC_TYPE_H_
#define PERCEPTION_COMMON_BASIC_TYPE_H_
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <memory>
struct PointXYZIRT {
  PCL_ADD_POINT4D;
  float intensity;
  std::uint16_t ring;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}
EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        std::uint16_t, ring, ring)(double, timestamp, timestamp))

struct PointXYZIRGBT {
  PCL_ADD_POINT4D;
  float intensity;
  PCL_ADD_RGB;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRGBT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        float, rgb, rgb)(std::uint32_t, rgba, rgba)(double, timestamp,
                                                    timestamp))


namespace robosense {
namespace perception {
struct RsHeader {
  uint32_t seq{0};   // sequence ID: consecutively increasing ID
  uint64_t time{0};  // time ns
  std::string frame_id{"base_link"};
};

enum class ObjectType {
  TYPE_UNKNOWN = 0,
  TYPE_PED = 1,
};

struct PvBBox2D {
  int x{0};
  int y{0};
  uint32_t width{0};
  uint32_t height{0};
  PvBBox2D(int x_ = 0, int y_ = 0, uint32_t width_ = 0, uint32_t height_ = 0) {
    x = x_;
    y = y_;
    width = width_;
    height = height_;
  }
};

struct Vec3D {
  float x{0.f};
  float y{0.f};
  float z{0.f};

  Vec3D(float x_ = 0.f, float y_ = 0.f, float z_ = 0.f) {
    x = x_;
    y = y_;
    z = z_;
  }
};

struct Vec2D {
  float x{0.f};
  float y{0.f};

  Vec2D(float x_ = 0.f, float y_ = 0.f) {
    x = x_;
    y = y_;
  }
};

struct BoxSize {
  float length{0.f};  // 目标 box 长度（朝向方向） m
  float width{0.f};   // 目标 box 宽度（垂直朝向方向） m
  float height{0.f};  // 目标 box 高度（地面法线方向） m

  BoxSize(float _length = 0.f, float _width = 0.f, float _height = 0.f) {
    length = _length;
    width = _width;
    height = _height;
  }
};
struct ObjectInner {
  uint32_t object_id;
  ObjectType type;
  float type_confidence;
  PvBBox2D box_full;
  PvBBox2D box_visible;
  Vec3D box_center_base;
  float yaw_base;
  BoxSize box_size;
};

struct ObjectInnerArray {
  using Ptr = std::shared_ptr<ObjectInnerArray>;
  RsHeader header;
  std::vector<ObjectInner> object_list;  // 障碍物列表
};
}  // namespace perception
}  // namespace robosense
#endif