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

#ifndef PERCEPTION_PPSEG_POSTPROCESS_H
#define PERCEPTION_PPSEG_POSTPROCESS_H

#include <vector>
#include "opencv2/opencv.hpp"

namespace robosense {
namespace perception {

int GetSegmentImage(float* result, cv::Mat& result_img);
void GetSegment(int32_t* result, cv::Mat& result_img);
void GetSegment(float* result, cv::Mat& result_img);

} // namespace robosense
} // namespace perception
#endif // PERCEPTION_PPSEG_POSTPROCESS_H