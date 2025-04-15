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
#include "perception/yolov8/yolov8_det_nn_hbdnn.h"

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
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;

  int num_input = 0;
  int num_output = 0;
  BASE_HB_CHECK(
    hbDNNGetInputCount(&num_input, mDnnHandle_),
    "hbDNNGetInputCount failed");
  // std::vector<hbDNNTensorProperties> input_attrs;
  // input_attrs.resize(num_input);
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
  // std::vector<hbDNNTensorProperties> output_attrs;
  // output_attrs.resize(num_output);
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

void Yolov8DetNN::PreProcess(const Image &image) {
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;

  int bg_color = 127;
  float y_scale = 1.0;
  float x_scale = 1.0;
  int x_shift = 0;
  int y_shift = 0;
  cv::Mat resize_img;
  const auto &raw_image = image.mat;
  int ori_width = raw_image.cols;
  int ori_height = raw_image.rows;

  const int& infer_width = config_ptr->nn_input_width;
  const int& infer_height = config_ptr->nn_input_height;

  x_scale = std::min(1.0 * infer_height / ori_height, 1.0 * infer_width / ori_width);
  y_scale = x_scale;
  if (x_scale <= 0 || y_scale <= 0) {
      throw std::runtime_error("Invalid scale factor.");
  }

  int new_w = ori_width * x_scale;
  x_shift = (infer_width - new_w) / 2;
  int x_other = infer_width - new_w - x_shift;

  int new_h = ori_height * y_scale;
  y_shift = (infer_height - new_h) / 2;
  int y_other = infer_height - new_h - y_shift;

  cv::Size targetSize(new_w, new_h);
  cv::resize(raw_image, resize_img, targetSize);
  cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(bg_color, bg_color, bg_color));
  infer_msg->lb.scale = x_scale;
  infer_msg->lb.x_pad = x_shift;
  infer_msg->lb.y_pad = y_shift;

  // Convert BGR888 to YUV420SP
  cv::Mat img_nv12;
  cv::Mat yuv_mat;
  cv::cvtColor(resize_img, yuv_mat, cv::COLOR_RGB2YUV_I420);
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

void Yolov8DetNN::Infer() {
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
  Infer();
  infer_time_record_.toc();

  postprocess_time_record_.tic();
  PostProcess();
  // save
  DrawImg(msg_ptr, msg_ptr->input_msg_ptr->camera_data_map.at(topic), infer_msg->od_results);
  DetectResultToObjectInner(infer_msg->od_results, msg_ptr->input_msg_ptr->camera_data_map.at(topic).timestamp, msg_ptr->output_msg_ptr->object_list_ptr);
  postprocess_time_record_.toc();

}

void Yolov8DetNN::PostProcess() {
  post_process_hbdnn(params_ptr_);
  // auto& results = params_ptr_->msg_ptr_->od_results.results;
  // for (size_t i=0; i < results.size(); i++) {
  //     auto& res = results[i];
  //     std::cout << "i: "         << i             << " "
  //           << "cls id is: " << res.cls_id    << " "
  //           // << "cls name is: " << res.cls_name    << " "
  //           << "score is: "   << res.prop      << " "
  //           << "box is: "    << res.box.left << " "
  //                             << res.box.top   << " "
  //                             << res.box.right << " "
  //                             << res.box.bottom << std::endl;
  // }
  // std::cout << "POST END" << std::endl;
}
void Yolov8DetNN::DetectResultToObjectInner(const ObjectList& od_results, uint64_t timestamp, perception::ObjectInnerArray::Ptr& out_msg) {
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

