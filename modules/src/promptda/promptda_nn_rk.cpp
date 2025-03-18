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
#include "perception/promptda/promptda_nn_rk.h"

namespace robosense {
namespace perception {

void PromptDANN::Init(const YAML::Node& cfg_node) {
    std::cout << Name() << ": init" << std::endl;
    params_ptr_->config_ptr_->Init(cfg_node);
    InitInfer();
    InitCalibration(params_ptr_->config_ptr_->calib_file);
    InitPreProcess();
    colormap_ = InitColorMap(256);
    image_preprocess_time_record_.init("image preprocess");
    lidar_preprocess_time_record_.init("lidar preprocess");
    knn_pooling_time_record_.init("knn pooling");
    preprocess_time_record_.init("preprocess");
    infer_time_record_.init("infer");
    postprocess_time_record_.init("postprocess");
}

bool PromptDANN::LoadEngine(const std::string& engineFile, rknn_context& context) {
  std::ifstream file(engineFile, std::ios::binary);
  if (!file.is_open()) {
      std::cerr << "Could not open engine file: " << engineFile << std::endl;
      return false;
  }

  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer;
  buffer.resize(fileSize);
  file.read(buffer.data(), fileSize);
  file.close();

  int ret = rknn_init(&context, buffer.data(), fileSize, 0, NULL);

  if (ret < 0) {
    return false;
  }
  return true;
}

void PromptDANN::InitCalibration(const std::string& calib_file) {
  lidar_calib_ = std::make_shared<LidarCalib>();
  camera_calib_ = std::make_shared<CameraCalib>();
  std::cout << "calib_file:" << calib_file << std::endl;
  auto calib_node = YAML::LoadFile(calib_file);
  // lidar
  const YAML::Node& lidar_cfg = calib_node["Sensor"]["Lidar"];
  const YAML::Node& lidar2imu_cfg = lidar_cfg["extrinsic"];
  Eigen::Quaternionf q_lidar2imu(
      lidar2imu_cfg["quaternion"]["w"].as<float>(),
      lidar2imu_cfg["quaternion"]["x"].as<float>(),
      lidar2imu_cfg["quaternion"]["y"].as<float>(),
      lidar2imu_cfg["quaternion"]["z"].as<float>());
  Eigen::Translation3f t_lidar2imu(
      lidar2imu_cfg["translation"]["x"].as<float>(),
      lidar2imu_cfg["translation"]["y"].as<float>(),
      lidar2imu_cfg["translation"]["z"].as<float>());
  lidar_calib_->lidar2imu_trans = t_lidar2imu * q_lidar2imu;
  std::cout << "lidar2imu_trans:\n" << lidar_calib_->lidar2imu_trans.matrix()
            << std::endl;
  // camera
  const YAML::Node& camera_cfg = calib_node["Sensor"]["Camera"];
  const YAML::Node& cam2imu_cfg = camera_cfg["extrinsic"];
  Eigen::Quaternionf q_cam2imu(
    cam2imu_cfg["quaternion"]["w"].as<float>(),
    cam2imu_cfg["quaternion"]["x"].as<float>(),
    cam2imu_cfg["quaternion"]["y"].as<float>(),
    cam2imu_cfg["quaternion"]["z"].as<float>());
  Eigen::Translation3f t_cam2imu(
    cam2imu_cfg["translation"]["x"].as<float>(),
    cam2imu_cfg["translation"]["y"].as<float>(),
    cam2imu_cfg["translation"]["z"].as<float>());
  auto cam2imu = t_cam2imu * q_cam2imu;

  lidar_calib_->lidar2cam_trans = cam2imu.inverse() * lidar_calib_->lidar2imu_trans;
  std::cout << "lidar2cam_trans:\n" << lidar_calib_->lidar2cam_trans.matrix()
            << std::endl;

  std::vector<float> coeffs;
  rally::yamlRead(camera_cfg["intrinsic"], "dist_coeff", coeffs);
  camera_calib_->D = cv::Mat(coeffs).clone();
  std::cout << "D:\n" << camera_calib_->D << std::endl;
  camera_calib_->K = cv::Mat::eye(3, 3, CV_32F);
  for (int i = 0; i < 9; ++i) {
    camera_calib_->K.at<float>(i / 3, i % 3) =
        camera_cfg["intrinsic"]["int_matrix"][i].as<float>();
  }
  std::cout << "K:\n" << camera_calib_->K << std::endl;
}

void PromptDANN::InitInfer() {
  std::cout << Name() << ": init infer" << std::endl;
  std::cout << "engine path: " << params_ptr_->config_ptr_->s1_model_path << std::endl;
  std::cout << "engine path: " << params_ptr_->config_ptr_->s2_model_path << std::endl;
  if(!LoadEngine(params_ptr_->config_ptr_->s1_model_path, s1_context_)) {
    std::cout << "Failed to deserialize engine." << std::endl;
    exit(-1);
  }; 
  if(!LoadEngine(params_ptr_->config_ptr_->s2_model_path, s2_context_)) {
    std::cout << "Failed to deserialize engine." << std::endl;
    exit(-1);
  }; 
  std::unordered_map<int, rknn_core_mask> map = {
    {0, RKNN_NPU_CORE_0},
    {1, RKNN_NPU_CORE_1},
    {2, RKNN_NPU_CORE_2},
    {3, RKNN_NPU_CORE_AUTO}
  };
  // rknn_set_core_mask(context_, map[params_ptr_->config_ptr_->core]);
  InitMem();
}

void PromptDANN::InitMem() {
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;

  int num_input = 0;
  int num_output = 0;

  { // s1
    BASE_RKNN_CHECK(rknn_query(s1_context_, RKNN_QUERY_IN_OUT_NUM, &s1_num_io_tensors_, sizeof(s1_num_io_tensors_)));
    // input
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    input_attrs.resize(s1_num_io_tensors_.n_input);
    depth_blobs_.resize(s1_num_io_tensors_.n_input);
    s1_inputs_.resize(s1_num_io_tensors_.n_input);
    for (size_t i = 0; i < s1_num_io_tensors_.n_input; ++i) {
      input_attrs[i].index = i;
      BASE_RKNN_CHECK(rknn_query(s1_context_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr)));
      DumpTensorAttr(input_attrs[i]);
      s1_inputs_[i].size = input_attrs[i].n_elems * sizeof(float);
      s1_inputs_[i].buf = malloc(s1_inputs_[i].size);
      s1_inputs_[i].index = i;
      s1_inputs_[i].type = RKNN_TENSOR_FLOAT32;
      s1_inputs_[i].fmt = RKNN_TENSOR_NHWC;
    }
    // output
    std::cout << s1_num_io_tensors_.n_output << std::endl;
    output_attrs.resize(s1_num_io_tensors_.n_output);
    s1_outputs_.resize(s1_num_io_tensors_.n_output);
    for (size_t i = 0; i < s1_num_io_tensors_.n_output; ++i) {
      output_attrs[i].index = i;
      BASE_RKNN_CHECK(rknn_query(s1_context_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr)));
      DumpTensorAttr(output_attrs[i]);
      s1_outputs_[i].index = i;
      s1_outputs_[i].is_prealloc = true;
      s1_outputs_[i].want_float = true;
      if (s1_outputs_[i].want_float) {
        s1_outputs_[i].size = output_attrs[i].n_elems * sizeof(float);
      } else {
        s1_outputs_[i].size = output_attrs[i].size;
      }
      s1_outputs_[i].buf = malloc(s1_outputs_[i].size);
      internal_attr_.push_back(output_attrs[i]);
      printf("i: %d s1_outputs[i].buf: %p\n", i, s1_outputs_[i].buf);
    }
  }

  { //s2
    BASE_RKNN_CHECK(rknn_query(s2_context_, RKNN_QUERY_IN_OUT_NUM, &s2_num_io_tensors_, sizeof(s2_num_io_tensors_)));
    // input
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    input_attrs.resize(s2_num_io_tensors_.n_input);
    s2_inputs_.resize(s2_num_io_tensors_.n_input);
    for (size_t i = 0; i < s2_num_io_tensors_.n_input; ++i) {
      input_attrs[i].index = i;
      BASE_RKNN_CHECK(rknn_query(s2_context_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr)));
      DumpTensorAttr(input_attrs[i]);
      s2_inputs_[i].size = input_attrs[i].n_elems * sizeof(float);
      s2_inputs_[i].buf = malloc(s2_inputs_[i].size);
      s2_inputs_[i].index = i;
      s2_inputs_[i].type = RKNN_TENSOR_FLOAT32;
      s2_inputs_[i].fmt = RKNN_TENSOR_NHWC;

      // if (strcmp(input_attrs[i].name, "input_image") == 0) {
      //   s2_inputs_[i].fmt = RKNN_TENSOR_NHWC;
      // } else {
      //   s2_inputs_[i].fmt = RKNN_TENSOR_NCHW;
      //   // s2_inputs_[i].buf = s1_outputs_[i-1].buf;
      //   // printf("i: %d s1_outputs[i-1].buf %p\n", i,  s2_inputs_[i].buf);
      // }
    }
    // output
    output_attrs.resize(s2_num_io_tensors_.n_output);
    infer_msg->nn_outputs.resize(s2_num_io_tensors_.n_output);
    s2_outputs_.resize(s2_num_io_tensors_.n_output);
    // model_attr->n_output = s1_num_io_tensors_.n_output;
    for (size_t i = 0; i < s2_num_io_tensors_.n_output; ++i) {
      output_attrs[i].index = i;
      BASE_RKNN_CHECK(rknn_query(s2_context_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr)));
      DumpTensorAttr(output_attrs[i]);
      s2_outputs_[i].index = i;
      s2_outputs_[i].is_prealloc = true;
      s2_outputs_[i].want_float = true;
      if (s2_outputs_[i].want_float) {
        s2_outputs_[i].size = output_attrs[i].n_elems * sizeof(float);
      } else {
        s2_outputs_[i].size = output_attrs[i].size;
      }
      infer_msg->nn_outputs[i] = malloc(s2_outputs_[i].size);
      s2_outputs_[i].buf = infer_msg->nn_outputs[i];
    }
  }
}

void PromptDANN::InitPreProcess() {
  crop_start_y_ = 140;
  crop_end_y_ = 980;
  crop_start_x_ = 0;
  crop_end_x_ = 1904;
  img_scale_factor_ = 0.5;

  lidar2cam_ = lidar_calib_->lidar2cam_trans.matrix();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      lidar_project_k_(i, j) = camera_calib_->K.at<float>(i, j);
    }
  }

  lidar_project_scale_factor_ = 1 / 7.0;

  lidar_project_k_(0, 2) = lidar_project_k_(0, 2) - 1.0f * crop_start_x_;
  lidar_project_k_(1, 2) = lidar_project_k_(1, 2) - 1.0f * crop_start_y_;
  lidar_project_k_ = lidar_project_k_ * lidar_project_scale_factor_;
  lidar_project_k_(2, 2) = 1.0f;

  lidar_w_ = static_cast<int>((crop_end_x_ - crop_start_x_) * lidar_project_scale_factor_);
  lidar_h_ = static_cast<int>((crop_end_y_ - crop_start_y_) * lidar_project_scale_factor_);

  knn_pooling_scale_ = 18;
  knn_k_ = 4;
}

void PromptDANN::PreProcess(const Image &image, const PointCloud::Ptr cloud_ptr) {
  auto& raw_image = image.mat;

  // 图像畸变校正和裁剪
  image_preprocess_time_record_.tic();
  auto undis_img = UndistortImg(raw_image, camera_calib_->K, camera_calib_->D);
  auto crop_image = undis_img(cv::Rect(crop_start_x_, crop_start_y_, crop_end_x_ - crop_start_x_, crop_end_y_ - crop_start_y_));
  if (img_scale_factor_ != 1) {
    cv::resize(crop_image, crop_image, cv::Size(), img_scale_factor_, img_scale_factor_, cv::INTER_NEAREST);
  }
  image_preprocess_time_record_.toc();

  lidar_preprocess_time_record_.tic();
  cv::Mat pc_img = ProjectLidar(cloud_ptr, lidar2cam_, lidar_project_k_);
  cv::Mat sparse_depth = cv::Mat::zeros(lidar_h_, lidar_w_, CV_32F);
  for (int i = 0; i < pc_img.rows; ++i) {
      int x = static_cast<int>(pc_img.at<float>(i, 0));
      int y = static_cast<int>(pc_img.at<float>(i, 1));
      if (x >= 0 && x < lidar_w_ && y >= 0 && y < lidar_h_) {
          sparse_depth.at<float>(y, x) = pc_img.at<float>(i, 2);
      }
  }

  // 低像素深度图 KNN 补全
  knn_pooling_time_record_.tic();
  cv::Mat dense_depth = KnnPooling(sparse_depth, knn_pooling_scale_, knn_k_);
  knn_pooling_time_record_.toc();
  lidar_preprocess_time_record_.toc();

  // nchw
  cv::minMaxLoc(dense_depth, &input_min_, &input_max_);
  cv::normalize(dense_depth, depth_blobs_[0], 0, 1, cv::NORM_MINMAX, CV_32FC1);
  for (auto i = 1; i < s1_inputs_.size(); ++i) {
    auto scale = std::pow(2.0, i);
    cv::resize(depth_blobs_[0], depth_blobs_[i], cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_LINEAR);
  }
  crop_image.convertTo(image_blob_, CV_32F, 1.0 / 255.0);

  for (auto i = 0; i < s1_inputs_.size(); ++i) {
    memcpy(s1_inputs_[i].buf, depth_blobs_[i].data, s1_inputs_[i].size);
  }
  memcpy(s2_inputs_[0].buf, image_blob_.data, s2_inputs_[0].size);
}

void PromptDANN::Perception(const DetectionMsg::Ptr &msg_ptr) {
  std::cout << Name() << ": perception" << std::endl;
  const auto &config_ptr = params_ptr_->config_ptr_;

  const std::string& topic = config_ptr->sub_topic;
  preprocess_time_record_.tic();
  PreProcess(msg_ptr->input_msg_ptr->camera_data_map.at(topic), msg_ptr->input_msg_ptr->cloud_ptr);
  preprocess_time_record_.toc();

  infer_time_record_.tic();

  BASE_RKNN_CHECK(rknn_inputs_set(s1_context_, s1_num_io_tensors_.n_input, s1_inputs_.data()));
  BASE_RKNN_CHECK(rknn_run(s1_context_, nullptr));
  BASE_RKNN_CHECK(rknn_outputs_get(s1_context_, s1_num_io_tensors_.n_output, s1_outputs_.data(), NULL))

  for (auto i = 0; i < s1_outputs_.size(); i++) {
    rknn_nchw_2_nhwc(s1_outputs_[i].buf, s2_inputs_[i+1].buf, 
      internal_attr_[i].dims[0], internal_attr_[i].dims[1],
      internal_attr_[i].dims[2], internal_attr_[i].dims[3]);
  }

  BASE_RKNN_CHECK(rknn_inputs_set(s2_context_, s2_num_io_tensors_.n_input, s2_inputs_.data()));
  BASE_RKNN_CHECK(rknn_run(s2_context_, nullptr));
  BASE_RKNN_CHECK(rknn_outputs_get(s2_context_, s2_num_io_tensors_.n_output, s2_outputs_.data(), NULL))

  infer_time_record_.toc();

  postprocess_time_record_.tic();
  PostProcess(msg_ptr, msg_ptr->input_msg_ptr->camera_data_map.at(topic), std::make_pair(config_ptr->clip_min, config_ptr->clip_max));
  postprocess_time_record_.toc();
}

void PromptDANN::PostProcess(const DetectionMsg::Ptr &msg_ptr, const Image& image, std::pair<float, float> limit_depth) {
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &config_ptr = params_ptr_->config_ptr_;
  auto timestamp = image.timestamp;
  double outputs_min, outputs_max;
  cv::Mat out_depth(cv::Size(config_ptr->nn_input_width, config_ptr->nn_input_height), CV_32FC1, reinterpret_cast<float*>(infer_msg->nn_outputs[0]));
  cv::normalize(out_depth, out_depth, input_min_, input_max_, cv::NORM_MINMAX, CV_32FC1);
  if (limit_depth.first == 0 && limit_depth.second == 0) {
    cv::minMaxLoc(out_depth, &outputs_min, &outputs_max);
  } else {
    outputs_min = limit_depth.first;
    outputs_max = limit_depth.second;
    out_depth = cv::max(out_depth, outputs_min);
    out_depth = cv::min(out_depth, outputs_max);
  }
  cv::normalize(out_depth, out_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::Mat rgb_depth;
  rgb_depth = ApplyColorMap(out_depth, colormap_);
  cv::cvtColor(rgb_depth, rgb_depth, cv::COLOR_RGB2BGR);
  msg_ptr->output_msg_ptr->mat = rgb_depth;
  if (params_ptr_->config_ptr_->save_img) {
    std::string save_path = "./results/" + std::to_string(timestamp) + "_depth.jpg";
    cv::imwrite(save_path, rgb_depth);
  }
}

} // namespace robosense
} // namespace perception

