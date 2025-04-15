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
#include "perception/yolov8/yolov8_det_nn_trt.h"
#include "perception/common/trt_utils.h"

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

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trt_logger_));
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), fileSize));

  if (!engine_) {
      return false;
  }
  return true;
}

void Yolov8DetNN::CheckNNAttr(nvinfer1::Dims dims) {
  const auto &config_ptr = params_ptr_->config_ptr_;
  if (config_ptr->nn_input_width != dims.d[3] || config_ptr->nn_input_height != dims.d[2]) {
    std::cout << "input width or height is not match" << std::endl;
    exit(-1);
  }
}

void Yolov8DetNN::InitInfer() {
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

void Yolov8DetNN::InitMem() {
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;
  const auto &model_attr = params_ptr_->model_attr_ptr_;
  model_attr->is_quant = false; // tensorrt 模型为fp16模型
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
}

std::shared_ptr<cv::Mat> Yolov8DetNN::PreProcess(const Image &image) {
  const auto &raw_image = image.mat;
  const auto &config_ptr = params_ptr_->config_ptr_;
  const auto &infer_msg = params_ptr_->msg_ptr_;

  // 1.resize  2.letterbox 3.nhwc2nchw and bgr2rgb and normalize
  float x_scale = std::min(1.0 * config_ptr->nn_input_height / raw_image.rows, 1.0 * config_ptr->nn_input_width / raw_image.cols);
  float y_scale = x_scale;
  if (x_scale <= 0 || y_scale <= 0) {
      throw std::runtime_error("Invalid scale factor.");
  }

  int new_w = raw_image.cols * x_scale;
  int x_shift = (config_ptr->nn_input_width - new_w) / 2;
  int x_other = config_ptr->nn_input_width - new_w - x_shift;

  int new_h = raw_image.rows * y_scale;
  int y_shift = (config_ptr->nn_input_height - new_h) / 2;
  int y_other = config_ptr->nn_input_height - new_h - y_shift;
  infer_msg->lb.scale = x_scale;
  infer_msg->lb.x_pad = x_shift;
  infer_msg->lb.y_pad = y_shift;
  cv::Mat resize_img;
  cv::resize(raw_image, resize_img, cv::Size(new_w, new_h), cv::INTER_NEAREST);
  cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
  auto blob = cv::dnn::blobFromImage(resize_img, 1.0 / 255.0, resize_img.size(), cv::Scalar(0, 0, 0), false, false);
  return std::make_shared<cv::Mat>(blob);
}

void Yolov8DetNN::Perception(const DetectionMsg::Ptr &msg_ptr) {
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
    for (int i = 0; i < od_results.count; ++i){
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

