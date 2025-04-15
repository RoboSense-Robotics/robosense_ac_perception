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
#include "perception/promptda/promptda_nn_trt.h"
#include "perception/common/trt_utils.h"

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
    project_lidar_time_record_.init("project lidar");
    sparse_time_record_.init("get sparse");
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

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trt_logger_));
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), fileSize));

  if (!engine_) {
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

void PromptDANN::CheckNNAttr(nvinfer1::Dims dims) {
  const auto &config_ptr = params_ptr_->config_ptr_;
}

void PromptDANN::InitInfer() {
  std::cout << Name() << ": init infer" << std::endl;
  std::cout << "engine path: " << params_ptr_->config_ptr_->model_path << std::endl;
  if(!LoadEngine(params_ptr_->config_ptr_->model_path)) {
    std::cout << "Failed to deserialize engine." << std::endl;
    exit(-1);
  };
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    std::cout << "Failed to create execution context." << std::endl;
    exit(-1);
  }
  cudaStreamCreate(&stream_);
  InitMem();
  context_->enqueueV3(stream_);
  if (use_cuda_graph_) {
    BASE_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    context_->enqueueV3(stream_);
    auto ret = cudaStreamEndCapture(stream_, &graph_);
    if (ret == cudaSuccess) {
#if CUDA_VERSION < 12010
      BASE_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#else
      BASE_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#endif
      BASE_CUDA_CHECK(cudaGraphDestroy(graph_));
    } else {
      std::cout << "Failed to capture cuda graph." << std::endl;
      exit(-1);
    }
  }
}

void PromptDANN::InitMem() {
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;
  model_attr->is_quant = false;
  auto num_io_tensors = engine_->getNbIOTensors();
  int num_input = 0;
  int num_output = 0;
  for (auto i=0; i < num_io_tensors; i++) {
    auto name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      num_input ++;
    } else {
      num_output ++;
    }
  }
  infer_msg->gpu_inputs.resize(num_input);
  model_attr->n_output = num_output;
  infer_msg->gpu_outputs.resize(num_output);
  infer_msg->nn_outputs.resize(num_output);
  std::cout << "num_input: " << num_input << " num_output: " << num_output <<std::endl;
  for (auto i=0; i < num_io_tensors; i++) {
    auto name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      auto dims = context_->getTensorShape(name);
      auto data_size = dataTypeSize(engine_->getTensorDataType(name));
      auto size = data_size;
      for (auto j=0; j<dims.nbDims; j++) {
        size *= dims.d[j];
      }
      DumpTensorAttr(name, dims, data_size);
      CheckNNAttr(dims);
      BASE_CUDA_CHECK(cudaMalloc(&infer_msg->gpu_inputs[i], size));
      context_->setTensorAddress(name, infer_msg->gpu_inputs[i]);
    } else {
      auto dims = context_->getTensorShape(name);
      auto data_size = dataTypeSize(engine_->getTensorDataType(name));
      auto size = data_size;
      auto out_tensor_attr = TensorAttr(dims.nbDims);
      for (auto j=0; j<dims.nbDims; j++) {
        size *= dims.d[j];
        out_tensor_attr.setDims(j, dims.d[j]);
        out_tensor_attr.setSize(size);
      }
      DumpTensorAttr(name, dims, data_size);
      model_attr->output_attrs.push_back(out_tensor_attr);
      int output_index = i - num_input;
      infer_msg->nn_outputs[output_index] = malloc(size);
      BASE_CUDA_CHECK(cudaMalloc(&infer_msg->gpu_outputs[output_index], size));
      context_->setTensorAddress(name, infer_msg->gpu_outputs[output_index]);
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
  project_lidar_time_record_.tic();
#if __aarch64__
  cv::Mat pc_img = ProjectLidar(cloud_ptr, lidar2cam_, lidar_project_k_);
#else
  cv::Mat pc_img = ProjectLidarOptimized(cloud_ptr, lidar2cam_, lidar_project_k_);
#endif
  project_lidar_time_record_.toc();
  sparse_time_record_.tic();
  cv::Mat sparse_depth = cv::Mat::zeros(lidar_h_, lidar_w_, CV_32F);
  for (int i = 0; i < pc_img.rows; ++i) {
      int x = static_cast<int>(pc_img.at<float>(i, 0));
      int y = static_cast<int>(pc_img.at<float>(i, 1));
      if (x >= 0 && x < lidar_w_ && y >= 0 && y < lidar_h_) {
          sparse_depth.at<float>(y, x) = pc_img.at<float>(i, 2);
      }
  }
  sparse_time_record_.toc();

  // 低像素深度图 KNN 补全
  knn_pooling_time_record_.tic();
#if __aarch64__
  cv::Mat dense_depth = KnnPooling(sparse_depth, knn_pooling_scale_, knn_k_);
#else
  cv::Mat dense_depth = KnnPoolingOptimized(sparse_depth, knn_pooling_scale_, knn_k_);
#endif
  knn_pooling_time_record_.toc();
  lidar_preprocess_time_record_.toc();

  image_blob_ = cv::dnn::blobFromImage(crop_image, 1.0 / 255, crop_image.size(), cv::Scalar(0, 0, 0), false, false);
  cv::minMaxLoc(dense_depth, &input_min_, &input_max_);
  cv::normalize(dense_depth, depth_blob_, 0, 1, cv::NORM_MINMAX, CV_32FC1);
  // depth_blob_ = cv::dnn::blobFromImage(dense_depth, 1.0, dense_depth.size(), cv::Scalar(0, 0, 0), false, false);
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
  BASE_CUDA_CHECK(cudaMemcpyAsync(infer_msg->gpu_inputs[0], image_blob_.data,
    image_blob_.total() * sizeof(float), cudaMemcpyHostToDevice, stream_));
  BASE_CUDA_CHECK(cudaMemcpyAsync(infer_msg->gpu_inputs[1], depth_blob_.data,
    depth_blob_.total() * sizeof(float), cudaMemcpyHostToDevice, stream_));


  if (use_cuda_graph_) {
    BASE_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
  } else {
    if(!context_->enqueueV3(stream_)) {
      std::cout << "Failed to forward." << std::endl;
      exit(-1);
    }
  }

  for (int i=0; i< model_attr->n_output; i++) {
    BASE_CUDA_CHECK(cudaMemcpyAsync(infer_msg->nn_outputs[i], infer_msg->gpu_outputs[i],
      model_attr->output_attrs[i].data_size, cudaMemcpyDeviceToHost, stream_));
  }
  BASE_CUDA_CHECK(cudaStreamSynchronize(stream_));
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

