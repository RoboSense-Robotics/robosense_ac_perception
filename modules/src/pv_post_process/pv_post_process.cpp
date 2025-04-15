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
#include "perception/pv_post_process/pv_post_process.hpp"
#include <unistd.h>  // Linux/macOS

namespace robosense {
namespace perception {
bool PvPostProcess::Init(const YAML::Node& config) {
  char buffer[256];
  if (getcwd(buffer, sizeof(buffer)) != nullptr) {
      std::cout << "current work dir is: " << buffer << std::endl;
  }
  std::string calib_file;
  if (rally::yamlRead(config, "calib_file", calib_file)) {
    std::cout << " calib_file: " << calib_file << std::endl;
  } else {
    std::cout << " please set calib_file in yaml file! ";
    return false;
  }
#if defined(USE_ROS1)
    calib_file = std::string(PROJECT_PATH) + "/../" + calib_file;
#elif defined(USE_ROS2)
    calib_file = std::string(buffer) + "/" + calib_file;
#endif
  SetCalibration(calib_file);
  mask_image_ = cv::Mat::zeros(image_height_, image_width_, CV_8UC1);
  rally::yamlRead(config, "debug_image", debug_image_);
  rally::yamlRead(config, "debug_lidar", debug_lidar_);
  rally::yamlRead(config, "use_cover_mask", mask_cover_ped_region_);
  cover_record_.init("cover");
  segment_record_.init("segment");
  set_record_.init("set");
  return true;
}
void PvPostProcess::SetCalibration(const std::string& calib_file) {
  lidar_calib_ = std::make_shared<LidarCalib>();
  camera_calib_ = std::make_shared<CameraCalib>();
  YAML::Node calib_node;
  calib_node = YAML::LoadFile(calib_file);
  // lidar
  const YAML::Node& lidar_cfg = calib_node["Sensor"]["Lidar"];
  const YAML::Node& lidar2imu_cfg = lidar_cfg["extrinsic"];
  Eigen::Quaternionf q_lidar2imu(lidar2imu_cfg["quaternion"]["w"].as<float>(),
                                 lidar2imu_cfg["quaternion"]["x"].as<float>(),
                                 lidar2imu_cfg["quaternion"]["y"].as<float>(),
                                 lidar2imu_cfg["quaternion"]["z"].as<float>());
  Eigen::Translation3f t_lidar2imu(
      lidar2imu_cfg["translation"]["x"].as<float>(),
      lidar2imu_cfg["translation"]["y"].as<float>(),
      lidar2imu_cfg["translation"]["z"].as<float>());
  lidar_calib_->lidar2imu_trans = t_lidar2imu * q_lidar2imu;
  std::cout << "lidar2imu_trans:\n"
            << lidar_calib_->lidar2imu_trans.matrix() << std::endl;

  const YAML::Node& camera_cfg = calib_node["Sensor"]["Camera"];
  const YAML::Node& cam2imu_cfg = camera_cfg["extrinsic"];
  Eigen::Quaternionf q_cam2imu(cam2imu_cfg["quaternion"]["w"].as<float>(),
                               cam2imu_cfg["quaternion"]["x"].as<float>(),
                               cam2imu_cfg["quaternion"]["y"].as<float>(),
                               cam2imu_cfg["quaternion"]["z"].as<float>());
  Eigen::Translation3f t_cam2imu(cam2imu_cfg["translation"]["x"].as<float>(),
                                 cam2imu_cfg["translation"]["y"].as<float>(),
                                 cam2imu_cfg["translation"]["z"].as<float>());
  auto cam2imu = t_cam2imu * q_cam2imu;

  lidar_calib_->lidar2cam_trans =
      cam2imu.inverse() * lidar_calib_->lidar2imu_trans;
  std::cout << "lidar2cam_trans:\n"
            << lidar_calib_->lidar2cam_trans.matrix() << std::endl;

  // camera
  std::vector<float> coeffs;
  rally::yamlRead(camera_cfg["intrinsic"], "dist_coeff", coeffs);
  camera_calib_->D = cv::Mat(coeffs).clone();
  std::cout << "D:\n" << camera_calib_->D << std::endl;
  camera_calib_->K = cv::Mat::eye(3, 3, CV_32F);
  for (int i = 0; i < 9; ++i) {
    camera_calib_->K.at<float>(i / 3, i % 3) =
        camera_cfg["intrinsic"]["int_matrix"][i].as<float>();
  }
  image_width_ = camera_cfg["intrinsic"]["image_size"][0].as<int>();
  image_height_ = camera_cfg["intrinsic"]["image_size"][1].as<int>();
  std::cout << "K:\n"
            << camera_calib_->K << " image size " << image_width_ << " "
            << image_height_ << " " << std::endl;
}
void PvPostProcess::GetCoverMask(const ObjectInnerArray::Ptr& object_array) {
  mask_image_ = 0;
  int count = 0;
  if (!mask_cover_ped_region_) {
    std::sort(object_array->object_list.begin(),
              object_array->object_list.end(),
              [&](const ObjectInner& obj1, const ObjectInner& obj2) {
                return obj1.box_full.height < obj2.box_full.height;
              });
  }
  for (const auto& object : object_array->object_list) {
    ++count;
    // edge object
    if (object.box_full.x < 100 || object.box_full.x > mask_image_.cols - 100 ||
        object.box_full.width < 10 || object.box_full.y < 100 ||
        object.box_full.y > mask_image_.rows - 100) {
      continue;
    }
    int into_thresh = 0.5 * object.box_full.width * 0.5;
    cv::Rect rect(object.box_full.x + into_thresh, object.box_full.y,
                  object.box_full.width - into_thresh * 2,
                  object.box_full.height);
    mask_image_(rect) = count;
  }
  for (size_t i = 0; i < object_array->object_list.size(); ++i) {
    if (object_array->object_list[i].type != ObjectType::TYPE_PED &&
        mask_cover_ped_region_ == false) {
      continue;
    }
    const auto& box_1 = object_array->object_list[i].box_full;
    int cur_x_min = box_1.x, cur_x_max = box_1.x + box_1.width;
    int cur_y_min = box_1.y, cur_y_max = box_1.y + box_1.height;
    for (size_t j = i + 1; j < object_array->object_list.size(); ++j) {
      if (object_array->object_list[j].type != ObjectType::TYPE_PED &&
          mask_cover_ped_region_ == false) {
        continue;
      }
      const auto& box_2 = object_array->object_list[j].box_full;
      int next_x_min = box_2.x, next_x_max = box_2.x + box_2.width;
      int next_y_min = box_2.y, next_y_max = box_2.y + box_2.height;
      // no cover
      if (cur_x_min >= next_x_max || cur_x_max <= next_x_min ||
          cur_y_min >= next_y_max || cur_y_max <= next_y_min) {
        continue;
      }
      // get cover rect
      int over_x_min = std::max(cur_x_min, next_x_min);
      int over_x_max = std::min(cur_x_max, next_x_max);
      int over_y_min = std::max(cur_y_min, next_y_min);
      int over_y_max = std::min(cur_y_max, next_y_max);
      cv::Rect over_rect(over_x_min, over_y_min, over_x_max - over_x_min,
                         over_y_max - over_y_min);
      mask_image_(over_rect) = 0;
    }
  }
}
void PvPostProcess::SegmentPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_ptr,
    const ObjectInnerArray::Ptr& object_array,
    std::vector<std::vector<int>>& segment_index) {
  segment_index.resize(object_array->object_list.size());
  std::vector<cv::Point3f> tmp_cam;
  std::vector<cv::Point2f> tmp_cv;
  tmp_cam.reserve(cloud_ptr->points.size());
  tmp_cv.reserve(cloud_ptr->points.size());
  for (size_t i = 0; i < cloud_ptr->points.size(); ++i) {
    const auto& p = cloud_ptr->points[i];
    Eigen::Vector3f point(p.x, p.y, p.z);
    Eigen::Vector3f point_cam = lidar_calib_->lidar2cam_trans * point;
    tmp_cam.emplace_back(cv::Point3f(point_cam[0], point_cam[1], point_cam[2]));
  }
  cv::projectPoints(tmp_cam, cv::Mat::zeros(3, 1, CV_64F),
                    cv::Mat::zeros(3, 1, CV_64F), camera_calib_->K,
                    camera_calib_->D, tmp_cv);
  for (size_t i = 0; i < cloud_ptr->points.size(); ++i) {
    const auto& p = cloud_ptr->points[i];
    if (p.z > 1.5) {
      continue;
    }
    const auto& voxel = tmp_cv[i];
    // edge
    if (isnan(voxel.x) || isnan(voxel.y) || voxel.x <= 1 ||
        voxel.x >= mask_image_.cols - 2 || voxel.y <= 1 ||
        voxel.y >= mask_image_.rows - 2) {
      continue;
    }
    // no obj
    if (mask_image_.at<uchar>(voxel.y, voxel.x) == 0) {
      continue;
    }
    // nearest
    if (mask_image_.at<uchar>(voxel.y - 1, voxel.x - 1) == 0 ||
        mask_image_.at<uchar>(voxel.y + 1, voxel.x + 1) == 0 ||
        mask_image_.at<uchar>(voxel.y - 1, voxel.x + 1) == 0 ||
        mask_image_.at<uchar>(voxel.y + 1, voxel.x - 1) == 0) {
      continue;
    }
    segment_index[mask_image_.at<uchar>(voxel.y, voxel.x) - 1].emplace_back(i);
  }
}
void PvPostProcess::Set3DInfo(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_ptr,
    const ObjectInnerArray::Ptr& object_array,
    std::vector<std::vector<int>>& segment_index,
    ObjectInnerArray::Ptr& out_object,
    std::vector<std::vector<int>>& segment_result) {
  out_object->object_list.clear();
  for (size_t i = 0; i < object_array->object_list.size(); ++i) {
    auto& obj = object_array->object_list[i];
    if (obj.type != ObjectType::TYPE_PED || segment_index[i].empty()) {
      continue;
    }
    std::sort(segment_index[i].begin(), segment_index[i].end(),
              [&](int a, int b) {
                return cloud_ptr->points[a].x < cloud_ptr->points[b].x;
              });
    size_t max_cluster_point_num = 0;
    std::pair<int, int> cluster = {0, 0};  // [)
    float pre_x = 0, start_x = 0;
    size_t start = 0;
    for (size_t j = 0; j < segment_index[i].size(); ++j) {
      float cur_x = cloud_ptr->points[segment_index[i][j]].x;
      // first
      if (pre_x <= 0.1) {
        start_x = cur_x;
        pre_x = cur_x;
      }
      // into cluster
      else if (std::abs(cur_x - pre_x) < 0.2 &&
               std::abs(cur_x - start_x) < 0.8) {
        pre_x = cur_x;
      }
      // new cluster
      else {
        if (j - start > max_cluster_point_num * 1.5) {
          cluster = std::make_pair(start, j);
          max_cluster_point_num = j - start;
        }
        pre_x = cur_x;
        start_x = cur_x;
        start = j;
      }
    }
    if (segment_index[i].size() - start > max_cluster_point_num * 1.5) {
      cluster = std::make_pair(start, segment_index[i].size());
      max_cluster_point_num = segment_index[i].size() - start;
    }
    if (cluster.second - cluster.first < 10) {
      continue;
    }
    float first_depth = cloud_ptr->points[segment_index[i][0]].x;
    if (first_depth < 2 && cluster.second - cluster.first < 300) {
      continue;
    }
    // get center
    Eigen::Vector3f center(0, 0, 0);
    float max_height = 0;
    for (int j = cluster.first; j < cluster.second; ++j) {
      const auto& p = cloud_ptr->points[segment_index[i][j]];
      center += Eigen::Vector3f(p.x, p.y, p.z);
      segment_result[i].emplace_back(j);
      max_height = std::max(max_height, p.z);
    }
    center = center / (cluster.second - cluster.first);
    obj.box_center_base.x = center[0];
    obj.box_center_base.y = center[1];
    obj.box_center_base.z = max_height - 1.7 * 0.5;
    obj.box_size.width = 0.7;
    obj.box_size.length = 0.7;
    obj.box_size.height = 1.7;
    out_object->object_list.emplace_back(obj);
  }
}
void PvPostProcess::MatchAndSmooth(ObjectInnerArray::Ptr& out_object) {
  std::vector<bool> match_label(history_obj_.size(), false);
  std::vector<bool> delete_label(out_object->object_list.size(), true);
  for (int i = 0; i < out_object->object_list.size(); ++i) {
    auto& cur_obj = out_object->object_list[i];
    for (int j = 0; j < history_obj_.size(); ++j) {
      auto& pre_obj = history_obj_[j];
      if (match_label[j]) {
        continue;
      }
      float dist =
          std::hypot(cur_obj.box_center_base.x - pre_obj.box_center_base.x,
                     cur_obj.box_center_base.y - pre_obj.box_center_base.y);
      if (dist < 0.25) {
        float cur_weight = dist / 0.25 * 0.8;
        cur_obj.box_center_base.x =
            cur_weight * cur_obj.box_center_base.x +
            (1 - cur_weight) * pre_obj.box_center_base.x;
        cur_obj.box_center_base.y =
            cur_weight * cur_obj.box_center_base.y +
            (1 - cur_weight) * pre_obj.box_center_base.y;
        cur_obj.box_center_base.z =
            0.2 * cur_obj.box_center_base.z + 0.8 * pre_obj.box_center_base.z;
        match_label[j] = true;
        delete_label[i] = false;
        break;
      }
    }
  }
  history_obj_ = out_object->object_list;
  for (int i = out_object->object_list.size() - 1; i >= 0; --i) {
    if (delete_label[i])
      out_object->object_list.erase(out_object->object_list.begin() + i);
  }
}
void PvPostProcess::Perception(MsgInterface::Ptr msg_ptr) {
  const auto& cloud_ptr = msg_ptr->input_msg_ptr->cloud_ptr;
  const auto& object_ptr = msg_ptr->input_msg_ptr->input_object;
  cover_record_.tic();
  GetCoverMask(object_ptr);
  cover_record_.toc();
  segment_record_.tic();
  std::vector<std::vector<int>> segment_index;
  SegmentPointCloud(cloud_ptr, object_ptr, segment_index);
  std::vector<std::vector<int>> segment_result(segment_index.size());
  segment_record_.toc();
  set_record_.tic();
  Set3DInfo(cloud_ptr, object_ptr, segment_index,
            msg_ptr->output_msg_ptr->output_object, segment_result);
  set_record_.toc();
  MatchAndSmooth(msg_ptr->output_msg_ptr->output_object);
  if (debug_image_) {
    msg_ptr->output_msg_ptr->debug_image.setTo(cv::Scalar(0, 0, 0));
    for (int i = 0; i < mask_image_.rows; ++i) {
      for (int j = 0; j < mask_image_.cols; ++j) {
        if (mask_image_.at<uchar>(i, j) != 0) {
          msg_ptr->output_msg_ptr->debug_image.at<cv::Vec3b>(i, j) =
              cv::Vec3b(255, 255, 255);
        }
      }
    }
  }
  if (debug_lidar_) {
    if (msg_ptr->output_msg_ptr->debug_cloud_ptr == nullptr) {
      msg_ptr->output_msg_ptr->debug_cloud_ptr =
          std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    msg_ptr->output_msg_ptr->debug_cloud_ptr->points.clear();
    for (const auto& segment : segment_index) {
      for (const auto& index : segment) {
        msg_ptr->output_msg_ptr->debug_cloud_ptr->points.emplace_back(
            cloud_ptr->points[index]);
      }
    }
  }
}

}  // namespace perception
}  // namespace robosense