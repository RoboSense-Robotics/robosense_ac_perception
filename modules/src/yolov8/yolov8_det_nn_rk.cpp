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
#include "perception/yolov8/yolov8_det_nn_rk.h"

namespace robosense {
namespace perception {

void Yolov8DetNN::Init(const YAML::Node& cfg_node) {
    std::cout << Name() << ": init" << std::endl;
    params_ptr_->config_ptr_->Init(cfg_node);
    InitInfer();
    preprocess_time_record_.init("preprocess");
    infer_time_record_.init("infer");
    postprocess_time_record_.init("postprocess");
}

bool Yolov8DetNN::LoadEngine(const std::string& engineFile) {
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

void Yolov8DetNN::InitInfer() {
  std::cout << Name() << ": init infer" << std::endl;
  std::cout << "engine path: " << params_ptr_->config_ptr_->model_path << std::endl;
  if(!LoadEngine(params_ptr_->config_ptr_->model_path)) {
    std::cout << "Failed to deserialize engine." << std::endl;
    exit(-1);
  };
  InitMem();
}

void Yolov8DetNN::InitMem() {
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
    infer_msg->inputs[i] = malloc(input_attrs_[i].size);
    inputs_[i].index = i;
    inputs_[i].type = RKNN_TENSOR_UINT8;
    inputs_[i].fmt = RKNN_TENSOR_NHWC;
    inputs_[i].size = input_attrs_[i].size;
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

    model_attr->is_quant = (output_attrs_[i].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs_[i].type == RKNN_TENSOR_INT8);
    auto tenso_attr = TensorAttr(output_attrs_[i].n_dims, output_attrs_[i].zp, output_attrs_[i].scale);
    for (uint32_t j=0; j < output_attrs_[i].n_dims; j++) {
      tenso_attr.setDims(j, output_attrs_[i].dims[j]);
    }
    model_attr->output_attrs.push_back(tenso_attr);

    outputs_[i].index = i;
    outputs_[i].is_prealloc = true;
    outputs_[i].size = output_attrs_[i].size;
    outputs_[i].want_float = !model_attr->is_quant;
    infer_msg->nn_outputs[i] = malloc(output_attrs_[i].size);
    outputs_[i].buf = infer_msg->nn_outputs[i];
  }
}

void Yolov8DetNN::PreProcess(const Image &image) {
  const auto &raw_image = image.mat;
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;

  int bg_color = 114;
  image_buffer_t dst_img;
  image_buffer_t src_img;
  memset(&dst_img, 0, sizeof(image_buffer_t));
  memset(&src_img, 0, sizeof(image_buffer_t));
  src_img.width = raw_image.size().width;
  src_img.height = raw_image.size().height;
  src_img.format = image_format_t::IMAGE_FORMAT_RGB888;
  src_img.size = get_image_size(&src_img);
  src_img.virt_addr = (unsigned char *)raw_image.data;

  dst_img.width = config_ptr->nn_input_width;
  dst_img.height = config_ptr->nn_input_height;
  dst_img.format = image_format_t::IMAGE_FORMAT_RGB888;
  dst_img.size = get_image_size(&dst_img);
  dst_img.virt_addr = (unsigned char *)params_ptr_->msg_ptr_->inputs[0];

  convert_image_with_letterbox(&src_img, &dst_img, &(params_ptr_->msg_ptr_->lb), bg_color);
}

void Yolov8DetNN::Perception(const DetectionMsg::Ptr &msg_ptr) {
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
  PostProcess();
  // save
  DrawImg(msg_ptr, msg_ptr->input_msg_ptr->camera_data_map.at(topic), params_ptr_->msg_ptr_->od_results);
  DetectResultToObjectInner(params_ptr_->msg_ptr_->od_results,msg_ptr->input_msg_ptr->camera_data_map.at(topic).timestamp, msg_ptr->output_msg_ptr->object_list_ptr);
  postprocess_time_record_.toc();

}

void Yolov8DetNN::PostProcess() {
  post_process(params_ptr_, params_ptr_->msg_ptr_->nn_outputs, &(params_ptr_->msg_ptr_->lb), &(params_ptr_->msg_ptr_->od_results));
  // std::cout<< "POST PROCESS" << std::endl;
  // auto& od_results = params_ptr_->msg_ptr_->od_results;
  // auto count = od_results.count;
  // for (int i=0; i<count; i++) {
  //     auto& res = od_results.results[i];
  //     std::cout << "i: "         << i             << " "
  //           << "cls id is: " << res.cls_id    << " "
  //           << "prop is: "   << res.prop      << " "
  //           << "box is: "    << res.box.left  << " "
  //                            << res.box.top   << " "
  //                            << res.box.right << " "
  //                            << res.box.bottom << std::endl;
  // }
  // std::cout<< "POST END";
}
void Yolov8DetNN::DetectResultToObjectInner(const ObjectList& od_results,uint64_t timestamp, perception::ObjectInnerArray::Ptr& out_msg) {
    // header
    out_msg->header.time = timestamp;
    // list
    out_msg->object_list.resize(od_results.count);
    for (int i = 0; i < od_results.count;++i){
        auto& res = od_results.results[i];
        auto &obj = out_msg->object_list[i];
        obj.object_id = i;
        if(res.cls_id==0){
          obj.type = perception::ObjectType::TYPE_PED;
        }else{
          obj.type = perception::ObjectType::TYPE_UNKNOWN;
        }
        obj.type_confidence = res.prop;
        obj.box_full.x = res.box.left;
        obj.box_full.y = res.box.top;
        obj.box_full.height=res.box.bottom-res.box.top;
        obj.box_full.width=res.box.right-res.box.left;
    }
}
void Yolov8DetNN::DrawImg(const DetectionMsg::Ptr &msg_ptr, const Image& image, const ObjectList& od_results) {
  cv::Mat raw_img;
  cv::cvtColor(image.mat, raw_img, cv::COLOR_RGB2BGR);
  auto count = od_results.count;

  for (int i = 0; i < count; ++i) {
    auto& res = od_results.results[i];
    // draw 2d
    if (res.cls_id == 0) {
      cv::rectangle(raw_img, cv::Point2i(int(res.box.left), int(res.box.top)),
                    cv::Point2i(int(res.box.right), int(res.box.bottom)), cv::Scalar(0, 255, 0), 5);
      // std::stringstream ss;
      // ss.precision(2);
      // ss.setf(std::ios::fixed);
      // ss << std::to_string(res.cls_id) << "|" << res.prop;
      // cv::putText(raw_img, ss.str(), cv::Point2i(res.box.left + 10, res.box.top + 30),
          // 0, 1, cv::Scalar(0, 0, 255), 1);
    }
  }
  msg_ptr->output_msg_ptr->mat = raw_img;
  if (params_ptr_->config_ptr_->save_img) {
    std::string save_path = "./results/" + std::to_string(image.timestamp) + ".jpg";
    cv::imwrite(save_path, raw_img);
  }
}

} // namespace robosense
} // namespace perception

