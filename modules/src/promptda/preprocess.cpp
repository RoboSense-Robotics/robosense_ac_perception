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
#include "perception/promptda/preprocess.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

namespace robosense {
namespace perception {
// 图像畸变校正
cv::Mat UndistortImg(const cv::Mat& img, const cv::Mat& K, const cv::Mat& D) {
  cv::Mat img_und;
  cv::undistort(img, img_und, K, D);
  return img_und;
}

// KNN 池化
cv::Mat KnnPooling(const cv::Mat& input_matrix, int pool_size, int k) {
  cv::Mat output_matrix = cv::Mat::zeros(input_matrix.size(), input_matrix.type());
  int height = input_matrix.rows;
  int width = input_matrix.cols;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (input_matrix.at<float>(i, j) == 0) {
        int top = std::max(i - pool_size / 2, 0);
        int down = std::min(i + pool_size / 2 + 1, height);
        int left = std::max(j - pool_size / 2, 0);
        int right = std::min(j + pool_size / 2 + 1, width);
        cv::Mat region = input_matrix(cv::Rect(left, top, right - left, down - top));

        std::vector<cv::Point> non_zero_points;
        cv::findNonZero(region, non_zero_points);

        if (non_zero_points.empty()) {
          output_matrix.at<float>(i, j) = 0;
        } else {
          std::vector<float> depth_values;
          std::vector<cv::Point2f> point_indices;
          for (const auto& point : non_zero_points) {
            depth_values.push_back(region.at<float>(point));
            point_indices.emplace_back(point.x + left, point.y + top);
          }

          std::vector<float> distances;
          for (const auto& point : point_indices) {
            distances.push_back(cv::norm(point - cv::Point2f(j, i)));
          }

          std::vector<int> indices(distances.size());
          std::iota(indices.begin(), indices.end(), 0);
          std::sort(indices.begin(), indices.end(), [&distances](int a, int b) {
              return distances[a] < distances[b];
          });

          float res_depth = 0;
          for (int l = 0; l < std::min(k, static_cast<int>(indices.size())); ++l) {
              res_depth += depth_values[indices[l]];
          }
          res_depth /= std::min(k, static_cast<int>(indices.size()));
          output_matrix.at<float>(i, j) = res_depth;
          }
      } else {
        output_matrix.at<float>(i, j) = input_matrix.at<float>(i, j);
      }
    }
  }
  return output_matrix;
}

cv::Mat KnnPoolingOptimized(const cv::Mat& input_matrix, int pool_size, int k) {
  CV_Assert(input_matrix.type() == CV_32F);
  cv::Mat output_matrix = cv::Mat::zeros(input_matrix.size(), CV_32F);
  const int height = input_matrix.rows;
  const int width = input_matrix.cols;
  const int pool_radius = pool_size / 2;

  #pragma omp parallel for
  for (int i = 0; i < height; ++i) {
      const float* input_row = input_matrix.ptr<float>(i);
      float* output_row = output_matrix.ptr<float>(i);
      for (int j = 0; j < width; ++j) {
          if (input_row[j] == 0.0f) {
              // 计算池化区域边界
              const int top = std::max(i - pool_radius, 0);
              const int bottom = std::min(i + pool_radius + 1, height);
              const int left = std::max(j - pool_radius, 0);
              const int right = std::min(j + pool_radius + 1, width);

              // 手动遍历区域并收集非零点
              std::vector<float> depth_values;
              std::vector<float> sq_distances;
              for (int y = top; y < bottom; ++y) {
                  const float* region_row = input_matrix.ptr<float>(y);
                  for (int x = left; x < right; ++x) {
                      const float val = region_row[x];
                      if (val != 0.0f) {
                          const float dx = x - j;
                          const float dy = y - i;
                          sq_distances.push_back(dx * dx + dy * dy);
                          depth_values.push_back(val);
                      }
                  }
              }

              const int num_points = depth_values.size();
              if (num_points == 0) {
                  output_row[j] = 0.0f;
              } else {
                  // 使用优先队列优化Top-K选择
                  std::priority_queue<std::pair<float, int>> max_heap;
                  for (int idx = 0; idx < num_points; ++idx) {
                      if (max_heap.size() < k) {
                          max_heap.emplace(sq_distances[idx], idx);
                      } else if (sq_distances[idx] < max_heap.top().first) {
                          max_heap.pop();
                          max_heap.emplace(sq_distances[idx], idx);
                      }
                  }

                  // 提取前K个最小距离的深度值
                  float sum = 0.0f;
                  const int effective_k = std::min(k, static_cast<int>(max_heap.size()));
                  for (int l = 0; l < effective_k; ++l) {
                      sum += depth_values[max_heap.top().second];
                      max_heap.pop();
                  }
                  output_row[j] = sum / effective_k;
              }
          } else {
              output_row[j] = input_row[j];
          }
      }
  }
  return output_matrix;
}

// 点云从激光雷达坐标系到图像坐标系的投影
cv::Mat ProjectLidar(const perception::PointCloud::Ptr& points, const Eigen::Matrix4f& ego2sensor_rt, const Eigen::Matrix3f& intrinsic) {
  auto points_lidar = *(points->cloud);
  std::vector<Eigen::Vector4f> points_lidar_homogeneous;
  for (const auto& point : points_lidar) {
    points_lidar_homogeneous.emplace_back(point.x, point.y, point.z, 1);
  }

  std::vector<Eigen::Vector4f> points_camera_homogeneous;
  for (const auto& point : points_lidar_homogeneous) {
    points_camera_homogeneous.push_back(ego2sensor_rt * point);
  }

  std::vector<Eigen::Vector3f> points_camera;
  std::vector<bool> valid;
  for (const auto& point : points_camera_homogeneous) {
    points_camera.emplace_back(point.x(), point.y(), point.z());
    valid.push_back(point.z() > 0.01);
  }

  for (auto& point : points_camera) {
    point /= point.z();
  }

  std::vector<Eigen::Vector3f> points_img;
  for (size_t i = 0; i < points_camera.size(); ++i) {
    if (valid[i]) {
      Eigen::Vector3f img_point = intrinsic * points_camera[i];
      img_point.z() = points_camera_homogeneous[i].z();
      points_img.push_back(img_point);
    }
  }

  cv::Mat result(points_img.size(), 3, CV_32F);
  for (size_t i = 0; i < points_img.size(); ++i) {
    result.at<float>(i, 0) = points_img[i].x();
    result.at<float>(i, 1) = points_img[i].y();
    result.at<float>(i, 2) = points_img[i].z();
  }
  return result;
}

cv::Mat ProjectLidarOptimized(const perception::PointCloud::Ptr& points, const Eigen::Matrix4f& ego2sensor_rt, const Eigen::Matrix3f& intrinsic) {
  const auto& points_lidar = points->cloud->points;
  const size_t num_points = points_lidar.size();
  std::vector<Eigen::Vector3f> points_img;
  points_img.reserve(num_points); // 预分配内存

  // 合并所有操作到单次遍历
  #pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    const auto& pt = points_lidar[i];
    // 齐次坐标转换 + 外参变换
    Eigen::Vector4f pt_cam_homo = ego2sensor_rt * Eigen::Vector4f(pt.x, pt.y, pt.z, 1.0f);

    // 过滤无效点
    if (pt_cam_homo.z() <= 0.01f) continue;

    // 归一化并投影到图像坐标系
    const float inv_z = 1.0f / pt_cam_homo.z();
    Eigen::Vector3f pt_norm(pt_cam_homo.x() * inv_z, pt_cam_homo.y() * inv_z, 1.0f);
    Eigen::Vector3f img_pt = intrinsic * pt_norm;
    img_pt.z() = pt_cam_homo.z(); // 保留深度值

    // 写入结果
    #pragma omp critical
    points_img.emplace_back(img_pt);
  }

  // 直接构造 cv::Mat 避免逐点填充
  cv::Mat result(points_img.size(), 3, CV_32F);
  if (!points_img.empty()) {
    std::memcpy(result.data, points_img[0].data(), points_img.size() * sizeof(Eigen::Vector3f));
  }
  return result;
}

void createColormapLUT(cv::Mat& colormap_lut)
{
  int lut_size = 256;  // You can adjust the LUT size for more granularity
  colormap_lut = cv::Mat(lut_size, 1, CV_8UC1);

  // Fill the LUT with values from 0 to 255
  for (int i = 0; i < lut_size; ++i)
  {
    colormap_lut.at<uchar>(i, 0) = static_cast<uchar>(i);
  }

  // Apply the colormap to the LUT
  cv::applyColorMap(colormap_lut, colormap_lut, cv::COLORMAP_JET);
}

void ProjectLidarOptimized2(const perception::PointCloud::Ptr& points, const Eigen::Matrix4f& ego2sensor_rt, const Eigen::Matrix3f& intrinsic, const cv::Mat& ori_img) {
  const auto& points_lidar = points->cloud->points;
  const size_t num_points = points_lidar.size();
  std::vector<Eigen::Vector3f> points_img;
  points_img.reserve(num_points); // 预分配内存
  float min_z = 10000;
  float max_z = 0;
  // 合并所有操作到单次遍历
  #pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    const auto& pt = points_lidar[i];
    // 齐次坐标转换 + 外参变换
    Eigen::Vector4f pt_cam_homo = ego2sensor_rt * Eigen::Vector4f(pt.x, pt.y, pt.z, 1.0f);

    // 过滤无效点
    if (pt_cam_homo.z() <= 0.01f) continue;

    // 归一化并投影到图像坐标系
    const float inv_z = 1.0f / pt_cam_homo.z();
    Eigen::Vector3f pt_norm(pt_cam_homo.x() * inv_z, pt_cam_homo.y() * inv_z, 1.0f);
    Eigen::Vector3f img_pt = intrinsic * pt_norm;
    img_pt.z() = pt_cam_homo.z(); // 保留深度值
    min_z = img_pt.z() < min_z ? img_pt.z() : min_z;
    max_z = img_pt.z() > max_z ? img_pt.z() : max_z;
    // 写入结果
    #pragma omp critical
    points_img.emplace_back(img_pt);
  }

  cv::Mat img = ori_img.clone();
  cv::Mat result(points_img.size(), 3, CV_32F);
  cv::Mat colormap_lut;
  createColormapLUT(colormap_lut);
  for (size_t i = 0; i < points_img.size(); ++i) {
    auto x = static_cast<int>(points_img[i].x()) * 7;
    auto y = static_cast<int>(points_img[i].y()) * 7;
    if (x >= 0 && x < 1918 && y >= 0 && y < 882) {
      float normalized_depth = (points_img[i].z() - min_z) / (max_z - min_z);
      auto colormap_index = static_cast<int>(normalized_depth * 255);
      cv::Vec3b color = colormap_lut.at<cv::Vec3b>(colormap_index, 0);
      cv::Scalar point_color(color[0], color[1], color[2]);  // BGR format
      // cv::Scalar point_color(255, 0, 0);  // BGR format
      cv::circle(img, cv::Point(x, y), 1, point_color, -1);
    }
  }
  auto time_stamp = points->timestamp;
  cv::imwrite(std::string("./results2/proj_")+std::to_string(time_stamp) + std::string(".png"), img);
}
} // namespace robosense
} // namespace perception