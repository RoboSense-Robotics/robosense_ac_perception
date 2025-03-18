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
#include "perception/ppseg/postprocess.h"

namespace robosense {
namespace perception {
// Define the type of color
using Color = std::tuple<int, int, int>;

// //Define a structure to represent a row of the table
struct Entry {
    int id;
    const char* name;
    Color color;
};

// //Define a global table
Entry cityscapes_label[] = {
    {0, "road", Color(128, 64, 128)},
    {1, "sidewalk", Color(128, 64, 128)},
    {2, "building", Color(70, 70, 70)},
    {3, "wall", Color(102, 102, 156)},
    {4, "fence", Color(190, 153, 153)},
    {5, "pole", Color(153, 153, 153)},
    {6, "traffic light", Color(250, 170, 30)},
    {7, "traffic sign", Color(220, 220, 0)},
    {8, "vegetation", Color(107, 142, 35)},
    {9, "terrain", Color(128, 64, 128)},
    {10, "sky", Color(70, 130, 180)},
    {11, "person", Color(220, 20, 60)},
    {12, "rider", Color(255, 0, 0)},
    {13, "car", Color(0, 0, 142)},
    {14, "truck", Color(0, 0, 70)},
    {15, "bus", Color(0, 60, 100)},
    {16, "train", Color(0, 80, 100)},
    {17, "motorcycle", Color(0, 0, 230)},
    {18, "bicycle", Color(119, 11, 32)}
};

Color getColorById(int id) {
    for (const auto& entry : cityscapes_label) {
        if (entry.id == id) {
            return entry.color;
        }
    }
    return Color(0, 0, 0);
}


int GetSegmentImage(float* result, cv::Mat& result_img) {
    int height = result_img.rows;
    int width = result_img.cols;
    int num_class = 19;
    // [1,class,height,width] -> [1,3,height,width]
    for (int batch = 0; batch < 1; batch++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int maxClassIndex = 0;
                for (int c = 1; c < num_class; c++) {
                    int currentIndex = batch * (num_class * height * width) + c * (height * width) + y * width + x;
                    int maxClassPos = batch * (num_class * height * width) + maxClassIndex * (height * width) + y * width + x;
                    if (result[currentIndex] > result[maxClassPos]) {
                        maxClassIndex = c;
                    }
                }
                Color foundColor = getColorById(maxClassIndex);
                result_img.at<cv::Vec3b>(y, x)[0] = std::get<0>(foundColor);       // R
                result_img.at<cv::Vec3b>(y, x)[1] = std::get<1>(foundColor);       // G
                result_img.at<cv::Vec3b>(y, x)[2] = std::get<2>(foundColor);       // B
            }
        }
    }
    return 0;
}

void GetSegment(int32_t* result, cv::Mat& result_img) {
    int height = result_img.rows;
    int width = result_img.cols;
    // [1,class,height,width] -> [1,3,height,width]
    for (int batch = 0; batch < 1; batch++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int currentIndex = batch * (height * width) + y * width + x;
                Color foundColor = getColorById(result[currentIndex]);
                result_img.at<cv::Vec3b>(y, x)[0] = std::get<0>(foundColor);       // R
                result_img.at<cv::Vec3b>(y, x)[1] = std::get<1>(foundColor);       // G
                result_img.at<cv::Vec3b>(y, x)[2] = std::get<2>(foundColor);       // B
            }
        }
    }
}
void GetSegment(float* result, cv::Mat& result_img) {
    int height = result_img.rows;
    int width = result_img.cols;
    // [1,class,height,width] -> [1,3,height,width]
    for (int batch = 0; batch < 1; batch++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int currentIndex = batch * (height * width) + y * width + x;
                Color foundColor = getColorById(static_cast<int>(result[currentIndex]));
                result_img.at<cv::Vec3b>(y, x)[0] = std::get<0>(foundColor);       // R
                result_img.at<cv::Vec3b>(y, x)[1] = std::get<1>(foundColor);       // G
                result_img.at<cv::Vec3b>(y, x)[2] = std::get<2>(foundColor);       // B
            }
        }
    }
}

} // namespace robosense
} // namespace perception
