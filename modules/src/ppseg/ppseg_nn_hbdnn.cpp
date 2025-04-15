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
#include "perception/ppseg/ppseg_nn_hbdnn.h"

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

void PPSegNN::InitInfer() {
  std::cout << Name() << ": init infer" << std::endl;
  std::cout << "engine path: " << params_ptr_->config_ptr_->model_path << std::endl;
  if(!LoadEngine(params_ptr_->config_ptr_->model_path)) {
    std::cout << "Failed to deserialize engine." << std::endl;
    exit(-1);
  };
  InitMem();
}

void PPSegNN::InitMem() {
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
  infer_msg->result = cv::Mat::zeros(config_ptr->nn_input_height, config_ptr->nn_input_width, CV_8UC3);
}

void PPSegNN::PreProcess(Image &image) {
  const auto &raw_image = image.mat;
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;

  int crop_x = (raw_image.cols - config_ptr->nn_input_width) / 2;
  int crop_y = (raw_image.rows - config_ptr->nn_input_height) / 2;
  // 0.unidist 1.resize 2. normalize
  cv::Mat img_und;
  cv::undistort(raw_image, img_und, camera_calib_->K, camera_calib_->D);
  img_und = img_und(cv::Rect(crop_x, crop_y, config_ptr->nn_input_width,
                              config_ptr->nn_input_height));
  image.mat = img_und;
  const int& infer_width = config_ptr->nn_input_width;
  const int& infer_height = config_ptr->nn_input_height;
  // Convert BGR888 to YUV420SP
  cv::Mat img_nv12;
  cv::Mat yuv_mat;
  cv::cvtColor(img_und, yuv_mat, cv::COLOR_RGB2YUV_I420);
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
  auto dest = infer_msg->inputs[0];
  memcpy(dest, ynv12, int(3 * infer_height * infer_width / 2));
}

void PPSegNN::Infer() {
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

void PPSegNN::Perception(const DetectionMsg::Ptr &msg_ptr) {
  std::cout << Name() << ": perception" << std::endl;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;
  const std::string& topic = config_ptr->sub_topic;
  preprocess_time_record_.tic();
  PreProcess(msg_ptr->input_msg_ptr->camera_data_map.at(topic));
  preprocess_time_record_.toc();

  infer_time_record_.tic();
  Infer();
  infer_time_record_.toc();

  postprocess_time_record_.tic();
  PostProcess(msg_ptr, msg_ptr->input_msg_ptr->camera_data_map.at(topic));
  postprocess_time_record_.toc();
}

void PPSegNN::PostProcess(const DetectionMsg::Ptr &msg_ptr, const Image& image) {
  const auto &infer_msg = params_ptr_->msg_ptr_;
  auto& out = infer_msg->result;
  auto dest = reinterpret_cast<int32_t*>(infer_msg->nn_outputs[0]);
  uint64_t timestamp = image.timestamp;
  const cv::Mat& raw_img = image.mat;
  cv::cvtColor(raw_img, raw_img, cv::COLOR_RGB2BGR);
  GetSegment(dest, out);
  cv::Mat ori_sized_mask;
  cv::cvtColor(out, ori_sized_mask, cv::COLOR_RGB2BGR);
  if (params_ptr_->config_ptr_->save_img) {
    std::string save_ori = "./results/" + std::to_string(timestamp) + "_ori.jpg";
    cv::imwrite(save_ori, raw_img);
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

