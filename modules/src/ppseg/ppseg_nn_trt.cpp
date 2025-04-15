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
#include "perception/ppseg/ppseg_nn_trt.h"

namespace robosense {
namespace perception {

void PPSegNN::Init(const YAML::Node& cfg_node) {
    std::cout << Name() << ": init" << std::endl;
    params_ptr_->config_ptr_->Init(cfg_node);
    InitInfer();
    InitCalibration(params_ptr_->config_ptr_->calib_file);
    preprocess_time_record_.init("preprocess");
    infer_time_record_.init("infer");
    postprocess_time_record_.init("postprocess");
}

bool PPSegNN::LoadEngine(const std::string& engineFile) {
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

void PPSegNN::InitCalibration(const std::string& calib_file) {
  camera_calib_ = std::make_shared<CameraCalib>();
  std::cout << "calib_file:" << calib_file << std::endl;
  auto calib_node = YAML::LoadFile(calib_file);
  const YAML::Node& camera_cfg = calib_node["Sensor"]["Camera"];
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

void PPSegNN::CheckNNAttr(nvinfer1::Dims dims) {
  const auto &config_ptr = params_ptr_->config_ptr_;
  if (config_ptr->nn_input_width != dims.d[3] || config_ptr->nn_input_height != dims.d[2]) {
    std::cout << "input width or height is not match" << std::endl;
    exit(-1);
  }
}

void PPSegNN::InitInfer() {
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
}

void PPSegNN::InitMem() {
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;
  model_attr->is_quant = false; // tensorrt 模型为fp32模型
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

  infer_msg->result = cv::Mat::zeros(config_ptr->nn_input_height, config_ptr->nn_input_width, CV_8UC3);
}

std::shared_ptr<cv::Mat> PPSegNN::PreProcess(Image &image) {
  const auto &raw_image = image.mat;
  const auto &config_ptr = params_ptr_->config_ptr_;
  int crop_x = (raw_image.cols - config_ptr->nn_input_width) / 2;
  int crop_y = (raw_image.rows - config_ptr->nn_input_height) / 2;
  // 0.unidist 1.resize 2.nhwc2nchw and bgr2rgb and normalize
  cv::Mat img_und;
  cv::undistort(raw_image, img_und, camera_calib_->K, camera_calib_->D);
  img_und = img_und(cv::Rect(crop_x, crop_y, config_ptr->nn_input_width,
                              config_ptr->nn_input_height));
  image.mat = img_und;
  auto blob = cv::dnn::blobFromImage(img_und, 1.0 / 255.0, cv::Size(),
                cv::Scalar(0, 0, 0), false, false);
  return std::make_shared<cv::Mat>(blob);
}

void PPSegNN::Perception(const DetectionMsg::Ptr &msg_ptr) {
    std::cout << Name() << ": perception" << std::endl;
    const auto &infer_msg = params_ptr_->msg_ptr_;
    const auto &config_ptr = params_ptr_->config_ptr_;
    const auto &model_attr = params_ptr_->model_attr_ptr_;

    const std::string& topic = config_ptr->sub_topic;
    preprocess_time_record_.tic();
    auto input_ptr = PreProcess(msg_ptr->input_msg_ptr->camera_data_map.at(topic));
    preprocess_time_record_.toc();

    infer_time_record_.tic();
    BASE_CUDA_CHECK(cudaMemcpyAsync(infer_msg->gpu_inputs[0], input_ptr->data,
      input_ptr->total() * sizeof(float), cudaMemcpyHostToDevice, stream_));
    if(!context_->enqueueV3(stream_)) {
      std::cout << "Failed to forward." << std::endl;
      exit(-1);
    }
    for (int i=0; i< model_attr->n_output; i++) {
      BASE_CUDA_CHECK(cudaMemcpyAsync(infer_msg->nn_outputs[i], infer_msg->gpu_outputs[i],
        model_attr->output_attrs[i].data_size, cudaMemcpyDeviceToHost, stream_));
    }
    BASE_CUDA_CHECK(cudaStreamSynchronize(stream_));
    infer_time_record_.toc();

    postprocess_time_record_.tic();
    PostProcess(msg_ptr, msg_ptr->input_msg_ptr->camera_data_map.at(topic));
    postprocess_time_record_.toc();
}

void PPSegNN::PostProcess(const DetectionMsg::Ptr &msg_ptr, const Image& image) {
  const auto &infer_msg = params_ptr_->msg_ptr_;
  auto& out = infer_msg->result;
  auto dest = static_cast<int32_t*>(infer_msg->nn_outputs[0]);

  uint64_t timestamp = image.timestamp;
  const cv::Mat& raw_img = image.mat;
  cv::cvtColor(raw_img, raw_img, cv::COLOR_RGB2BGR);
  GetSegment(dest, out);
  cv::Mat ori_sized_mask;
  cv::cvtColor(out, ori_sized_mask, cv::COLOR_BGR2RGB);
  if (params_ptr_->config_ptr_->save_img) {
    std::string save_path = "./results/" + std::to_string(timestamp) + "_mask.jpg";
    cv::imwrite(save_path, ori_sized_mask);
  }
  cv::addWeighted(ori_sized_mask, 0.5, raw_img, 0.5, 0.0, ori_sized_mask);
  msg_ptr->output_msg_ptr->mat = ori_sized_mask;
  if (params_ptr_->config_ptr_->save_img) {
    std::string save_path2 = "./results/" + std::to_string(timestamp) + ".jpg";
    cv::imwrite(save_path2, ori_sized_mask);
  }
}

} // namespace robosense
} // namespace perception

