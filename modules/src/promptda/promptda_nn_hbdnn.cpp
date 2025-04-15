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
#include "perception/promptda/promptda_nn_hbdnn.h"

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

bool PromptDANN::LoadEngine(const std::string& engineFile) {
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

  const void* data = static_cast<const void*>(buffer.data());
  int32_t size = buffer.size();
  BASE_HB_CHECK(
    hbDNNInitializeFromDDR(&mPackedDnnHandle_, &data, &size, 1),
    "hbDNNInitializeFromDDR failed");
  const char **model_name_list;
  int model_count = 0;
  BASE_HB_CHECK(
    hbDNNGetModelNameList(&model_name_list, &model_count, mPackedDnnHandle_),
    "hbDNNGetModelNameList failed");
  if (model_count > 1) {
    std::cout << "This model file have more than 1 model, only use model 0.";
  }
  const char *model_name = model_name_list[0];
  std::cout << "model name: " << model_name << std::endl;
  BASE_HB_CHECK(
    hbDNNGetModelHandle(&mDnnHandle_, mPackedDnnHandle_, model_name),
    "hbDNNGetModelHandle failed");
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
  std::cout << "engine path: " << params_ptr_->config_ptr_->model_path << std::endl;
  if(!LoadEngine(params_ptr_->config_ptr_->model_path)) {
    std::cout << "Failed to deserialize engine." << std::endl;
    exit(-1);
  };
  InitMem();
}

void PromptDANN::InitMem() {
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;

  int num_input = 0;
  int num_output = 0;
  BASE_HB_CHECK(
    hbDNNGetInputCount(&num_input, mDnnHandle_),
    "hbDNNGetInputCount failed");
  infer_msg->inputs.resize(num_input);
  inputs_.resize(num_input);
  for (int i = 0; i < num_input; ++i) {
    const char* name;
    hbDNNTensorProperties& input_prop = inputs_[i].properties;
    BASE_HB_CHECK(
      hbDNNGetInputName(&name, mDnnHandle_, i),
     "hbDNNGetInputName failed");
    BASE_HB_CHECK(
      hbDNNGetInputTensorProperties(&input_prop, mDnnHandle_, i),
      "hbDNNGetInputTensorProperties failed");
    input_prop.alignedShape = input_prop.validShape;
    BASE_HB_CHECK(
      hbSysAllocCachedMem(&inputs_[i].sysMem[0], input_prop.alignedByteSize),
      "HBDNN Malloc failed");
    infer_msg->inputs[i] = inputs_[i].sysMem[0].virAddr;
    DumpTensorAttr(name, input_prop, inputs_[i]);
  }

  BASE_HB_CHECK(
    hbDNNGetOutputCount(&num_output, mDnnHandle_),
    "hbDNNGetOutputCount failed");
  infer_msg->nn_outputs.resize(num_output);
  outputs_.resize(num_output);
  model_attr->n_output = num_output;
  for (int i = 0; i < num_output; ++i) {
    const char* name;
    BASE_HB_CHECK(
      hbDNNGetOutputName(&name, mDnnHandle_, i),
      "hbDNNGetOutputName failed");
    hbDNNTensorProperties &output_prop = outputs_[i].properties;
    BASE_HB_CHECK(
      hbDNNGetOutputTensorProperties(&output_prop, mDnnHandle_, i),
      "hbDNNGetOutputTensorProperties failed");
    if (output_prop.quantiType == SCALE) {
      model_attr->output_attrs.push_back(TensorAttr(output_prop.scale.scaleData));
    } else {
      model_attr->output_attrs.push_back(TensorAttr(nullptr));
    }
    BASE_HB_CHECK(
      hbSysAllocCachedMem(&outputs_[i].sysMem[0], output_prop.alignedByteSize),
      "HBDNN Malloc failed");
    infer_msg->nn_outputs[i] = outputs_[i].sysMem[0].virAddr;
    DumpTensorAttr(name, output_prop, outputs_[i]);
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
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;

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
  // image_blob_ = cv::dnn::blobFromImage(crop_image, 1.0 / 255, crop_image.size(), cv::Scalar(0, 0, 0), false, false);
  cv::minMaxLoc(dense_depth, &input_min_, &input_max_);
  cv::normalize(dense_depth, depth_blob_, 0, 1, cv::NORM_MINMAX, CV_32FC1);

  memcpy(infer_msg->inputs[1], depth_blob_.data, dense_depth.total()*sizeof(float));
  // Convert BGR888 to YUV420SP
  const int& infer_width = config_ptr->nn_input_width;
  const int& infer_height = config_ptr->nn_input_height;
  cv::Mat img_nv12;
  cv::Mat yuv_mat;
  cv::cvtColor(crop_image, yuv_mat, cv::COLOR_RGB2YUV_I420);
  uint8_t *yuv = yuv_mat.ptr<uint8_t>();
  img_nv12 = cv::Mat(infer_height * 3 / 2, infer_width, CV_8UC1);
  uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
  int uv_height = infer_height / 2;
  int uv_width = infer_width / 2;
  int y_size = infer_height * infer_width;
  memcpy(ynv12, yuv, y_size);
  uint8_t *nv12 = ynv12 + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;
  for (int i = 0; i < uv_width * uv_height; i++) {
      *nv12++ = *u_data++;
      *nv12++ = *v_data++;
  }
  memcpy(infer_msg->inputs[0], ynv12, int(3 * infer_height * infer_width / 2));
}

void PromptDANN::Infer() {
  BASE_HB_CHECK(hbSysFlushMem(&inputs_[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN), "hbSysFlushMem failed");
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  hbDNNTensor *output = outputs_.data();
  hbDNNTaskHandle_t mTaskHandle = nullptr;
  BASE_HB_CHECK(hbDNNInfer(&mTaskHandle,
                            &output,
                            inputs_.data(),
                            mDnnHandle_,
                            &infer_ctrl_param), "hbDNNInfer failed");
  BASE_HB_CHECK(hbDNNWaitTaskDone(mTaskHandle, 0), "hbDNNWaitTaskDone failed");
  for (auto i = 0; i < outputs_.size(); i++) {
    BASE_HB_CHECK(hbSysFlushMem(&outputs_[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE), "hbSysFlushMem failed");
  }
  BASE_HB_CHECK(hbDNNReleaseTask(mTaskHandle), "hbDNNReleaseTask failed");
}

void PromptDANN::Perception(const DetectionMsg::Ptr &msg_ptr) {
  std::cout << Name() << ": perception" << std::endl;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;

  const std::string& topic = config_ptr->sub_topic;
  preprocess_time_record_.tic();
  PreProcess(msg_ptr->input_msg_ptr->camera_data_map.at(topic), msg_ptr->input_msg_ptr->cloud_ptr);
  preprocess_time_record_.toc();

  infer_time_record_.tic();
  Infer();
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
    std::string save_path = "./results/" +  std::to_string(timestamp) + "_depth.jpg";
    cv::imwrite(save_path, rgb_depth);
  }
}

} // namespace robosense
} // namespace perception

