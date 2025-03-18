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
#include "perception/ppseg/ppseg_nn_rk.h"

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

	int ret = rknn_init(&context_, buffer.data(), fileSize, 0, NULL);

	if (ret < 0) {
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

void PPSegNN::InitInfer() {
  std::cout << Name() << ": init infer" << std::endl;
	std::cout << "engine path: " << params_ptr_->config_ptr_->model_path << std::endl;
	if(!LoadEngine(params_ptr_->config_ptr_->model_path)) {
		std::cout << "Failed to deserialize engine." << std::endl;
		exit(-1);
	};
  InitMem();
}

void PPSegNN::CheckNNAttr(rknn_tensor_attr tenso_attr) {
  const auto &config_ptr = params_ptr_->config_ptr_;
  if (config_ptr->nn_input_width != tenso_attr.dims[2] || config_ptr->nn_input_height != tenso_attr.dims[1]) {
    std::cout << "input width or height is not match, please check usr_config.yaml and your model!" << std::endl;
    exit(-1);
  }
}

void PPSegNN::InitMem() {
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;

  int num_input = 0;
  int num_output = 0;
  BASE_RKNN_CHECK(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &num_io_tensors_, sizeof(num_io_tensors_)));

  // input
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  input_attrs_.resize(num_io_tensors_.n_input);
  infer_msg->inputs.resize(num_io_tensors_.n_input);
  inputs_.resize(num_io_tensors_.n_input);
  for (size_t i = 0; i < num_io_tensors_.n_input; ++i) {
    input_attrs_[i].index = i;
    BASE_RKNN_CHECK(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr)));
    DumpTensorAttr(input_attrs_[i]);
    CheckNNAttr(input_attrs_[i]);
    inputs_[i].size = input_attrs_[i].n_elems * sizeof(float);
    infer_msg->inputs[i] = malloc(inputs_[i].size);
    inputs_[i].index = i;
    inputs_[i].type = RKNN_TENSOR_FLOAT32;
    inputs_[i].fmt = RKNN_TENSOR_NHWC;
    inputs_[i].buf = infer_msg->inputs[i];
  }
  // output
  output_attrs_.resize(num_io_tensors_.n_output);
  infer_msg->nn_outputs.resize(num_io_tensors_.n_output);
  outputs_.resize(num_io_tensors_.n_output);
  model_attr->n_output = num_io_tensors_.n_output;
  for (size_t i = 0; i < num_io_tensors_.n_output; ++i) {
    output_attrs_[i].index = i;
    BASE_RKNN_CHECK(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr)));
    DumpTensorAttr(output_attrs_[i]);

    outputs_[i].index = i;
    outputs_[i].is_prealloc = true;
    outputs_[i].want_float = 1;
    outputs_[i].size = output_attrs_[i].n_elems * sizeof(float);
    infer_msg->nn_outputs[i] = malloc(outputs_[i].size);
    std::cout << outputs_[i].size << std::endl;
    outputs_[i].buf = infer_msg->nn_outputs[i];
  }

  infer_msg->result = cv::Mat::zeros(config_ptr->nn_input_height, config_ptr->nn_input_width, CV_8UC3);
}

void PPSegNN::PreProcess(Image &image) {
  const auto &raw_image = image.mat;
  const auto &config_ptr = params_ptr_->config_ptr_;
  int crop_x = (raw_image.cols - config_ptr->nn_input_width) / 2;
  int crop_y = (raw_image.rows - config_ptr->nn_input_height) / 2;
  // 0.unidist 1.resize 2. normalize
  cv::Mat img_und;
  cv::undistort(raw_image, img_und, camera_calib_->K, camera_calib_->D);
  img_und = img_und(cv::Rect(crop_x, crop_y, config_ptr->nn_input_width,
                              config_ptr->nn_input_height));
  image.mat = img_und;
  img_und.convertTo(image_blob_, CV_32F, 1.0 / 255.0);
  memcpy(inputs_[0].buf, image_blob_.data, inputs_[0].size);
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
  BASE_RKNN_CHECK(rknn_inputs_set(context_, num_io_tensors_.n_input, inputs_.data()));
  BASE_RKNN_CHECK(rknn_run(context_, nullptr));
  BASE_RKNN_CHECK(rknn_outputs_get(context_, num_io_tensors_.n_output, outputs_.data(), NULL))
  infer_time_record_.toc();

  postprocess_time_record_.tic();
  PostProcess(msg_ptr, msg_ptr->input_msg_ptr->camera_data_map.at(topic));
  postprocess_time_record_.toc();
}

void PPSegNN::PostProcess(const DetectionMsg::Ptr &msg_ptr, const Image& image) {
  const auto &infer_msg = params_ptr_->msg_ptr_;
  auto& out = infer_msg->result;
  auto dest = reinterpret_cast<float*>(infer_msg->nn_outputs[0]);
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

