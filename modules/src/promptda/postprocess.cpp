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
#include "perception/common/input_msg.h"
#include "perception/yolov8/postprocess.h"

namespace robosense {
namespace perception {
std::vector<cv::Vec3b> InitColorMap(int num_colors) {
  std::vector<cv::Vec3b> colormap;

  // Spectral颜色映射的关键点
  std::vector<std::tuple<float, float, float>> key_colors = {
    {0.6196078431372549, 0.00392156862745098, 0.25882352941176473}, // 深红色
    {0.8352941176470589, 0.24313725490196078, 0.30980392156862746}, // 红色
    {0.9568627450980393, 0.42745098039215684, 0.2627450980392157},  // 橙色
    {0.9921568627450981, 0.6823529411764706, 0.3803921568627451},   // 黄色
    {0.996078431372549, 0.8784313725490196, 0.5450980392156862},   // 浅黄色
    {1.0, 1.0, 0.7490196078431373},                                // 白色
    {0.9019607843137255, 0.9607843137254902, 0.596078431372549},   // 浅绿色
    {0.6705882352941176, 0.8666666666666667, 0.6431372549019608},  // 绿色
    {0.4, 0.7607843137254902, 0.6470588235294118},                // 蓝绿色
    {0.19607843137254902, 0.5333333333333333, 0.7411764705882353}, // 蓝色
    {0.3686274509803922, 0.30980392156862746, 0.6352941176470588}  // 深蓝色
  };

  // 线性插值生成颜色映射
  for (int i = 0; i < num_colors; ++i) {
    float t = static_cast<float>(i) / (num_colors - 1);
    int index = static_cast<int>(t * (key_colors.size() - 1));
    float t0 = static_cast<float>(index) / (key_colors.size() - 1);
    float t1 = static_cast<float>(index + 1) / (key_colors.size() - 1);
    float alpha = (t - t0) / (t1 - t0);

    cv::Vec3b color;
    color[0] = static_cast<uint8_t>(255 * (std::get<0>(key_colors[index]) * (1 - alpha) + std::get<0>(key_colors[index + 1]) * alpha));
    color[1] = static_cast<uint8_t>(255 * (std::get<1>(key_colors[index]) * (1 - alpha) + std::get<1>(key_colors[index + 1]) * alpha));
    color[2] = static_cast<uint8_t>(255 * (std::get<2>(key_colors[index]) * (1 - alpha) + std::get<2>(key_colors[index + 1]) * alpha));

    colormap.push_back(color);
  }
  return colormap;
}

cv::Mat  ApplyColorMap(const cv::Mat& depth, const std::vector<cv::Vec3b>& colormap) {
  cv::Mat rgb_depth(depth.size(), CV_8UC3);
  // 检查内存是否连续以优化访问
  if (depth.isContinuous() && rgb_depth.isContinuous()) {
    const uint8_t* depth_data = depth.ptr<uint8_t>(0);
    cv::Vec3b* rgb_data = rgb_depth.ptr<cv::Vec3b>(0);
    for (size_t k=0; k<depth.total(); k++) {
      rgb_data[k] = colormap[depth_data[k]];
    }
  } else {
    // 非连续内存时按行遍历
    for (int i=0; i<depth.rows; i++) {
      const uint8_t* depth_row = depth.ptr<uint8_t>(i);
      cv::Vec3b* rgb_row = rgb_depth.ptr<cv::Vec3b>(i);
      for (int j=0; i< depth.cols; j++) {
        rgb_row[j] = colormap[depth_row[j]];
      }
    }
  }
  return rgb_depth;
}
} // namespace robosense
} // namespace perception