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
#ifndef PERCEPTION_COMMON_HBDNN_UTILS_H_
#define PERCEPTION_COMMON_HBDNN_UTILS_H_
#include <unordered_map>
#include "hb_dnn.h"

#define BASE_HB_CHECK(error_code, err_msg) { HBAssert((error_code), (err_msg), __FILE__, __LINE__); }
inline void HBAssert(int32_t error_code, const char* err_msg, const char* file, int line, bool abort = true) {
    if (error_code != 0) {
        std::cout << "HBassert: " << error_code << " " << err_msg << " " << file
                << " " << line << std::endl;
        // std::cout << std::string(hbDNNGetErrorDesc(error_code)) << std::endl;
        if (abort) {
            exit(-1);
        }
    }
}
std::unordered_map<int32_t, std::string> hbDNNTensorLayoutMapToString = {
    {HB_DNN_LAYOUT_NHWC, "HB_DNN_LAYOUT_NHWC"},
    {HB_DNN_LAYOUT_NCHW, "HB_DNN_LAYOUT_NCHW"},
    {HB_DNN_LAYOUT_NONE, "HB_DNN_LAYOUT_NONE"}
};
std::unordered_map<int32_t, std::string> hbDNNDataTypeMapToString = {
    {HB_DNN_IMG_TYPE_Y, "HB_DNN_IMG_TYPE_Y"},
    {HB_DNN_IMG_TYPE_NV12, "HB_DNN_IMG_TYPE_NV12"},
    {HB_DNN_IMG_TYPE_NV12_SEPARATE, "HB_DNN_IMG_TYPE_NV12_SEPARATE"},
    {HB_DNN_IMG_TYPE_YUV444, "HB_DNN_IMG_TYPE_YUV444"},
    {HB_DNN_IMG_TYPE_RGB, "HB_DNN_IMG_TYPE_RGB"},
    {HB_DNN_IMG_TYPE_BGR, "HB_DNN_IMG_TYPE_BGR"},
    {HB_DNN_TENSOR_TYPE_S4, "HB_DNN_TENSOR_TYPE_S4"},
    {HB_DNN_TENSOR_TYPE_U4, "HB_DNN_TENSOR_TYPE_U4"},
    {HB_DNN_TENSOR_TYPE_S8, "HB_DNN_TENSOR_TYPE_S8"},
    {HB_DNN_TENSOR_TYPE_U8, "HB_DNN_TENSOR_TYPE_U8"},
    {HB_DNN_TENSOR_TYPE_F16, "HB_DNN_TENSOR_TYPE_F16"},
    {HB_DNN_TENSOR_TYPE_S16, "HB_DNN_TENSOR_TYPE_S16"},
    {HB_DNN_TENSOR_TYPE_U16, "HB_DNN_TENSOR_TYPE_U16"},
    {HB_DNN_TENSOR_TYPE_F32, "HB_DNN_TENSOR_TYPE_F32"},
    {HB_DNN_TENSOR_TYPE_S32, "HB_DNN_TENSOR_TYPE_S32"},
    {HB_DNN_TENSOR_TYPE_U32, "HB_DNN_TENSOR_TYPE_U32"},
    {HB_DNN_TENSOR_TYPE_F64, "HB_DNN_TENSOR_TYPE_F64"},
    {HB_DNN_TENSOR_TYPE_S64, "HB_DNN_TENSOR_TYPE_S64"},
    {HB_DNN_TENSOR_TYPE_U64, "HB_DNN_TENSOR_TYPE_U64"},
    {HB_DNN_TENSOR_TYPE_MAX, "HB_DNN_TENSOR_TYPE_MA"}
};
std::unordered_map<int32_t, std::string> hbDNNQuantiTypeMapToString = {
    {NONE, "NONE"},
    {SHIFT,"SHIFT"},
    {SCALE,"SCALE"}
};
void DumpTensorAttr(const std::string& name, const hbDNNTensorProperties& prop, const hbDNNTensor&tensor) {
    std::stringstream dims;
    for (int i = 0; i < prop.alignedShape.numDimensions - 1; ++i) {
    dims << prop.alignedShape.dimensionSize[i] << "x";
    }
    dims << prop.alignedShape.dimensionSize[prop.alignedShape.numDimensions - 1];
    std::cout << "name "     << name
              << " n_dims="  << prop.validShape.numDimensions
              << " dims="    << dims.str()
              << " fmt="     << hbDNNTensorLayoutMapToString[prop.tensorLayout]
              << " type="    << hbDNNDataTypeMapToString[prop.tensorType]
              << " qnt_type="<< hbDNNQuantiTypeMapToString[prop.quantiType]
            //   << " scale="   << *(prop.scale.scaleData)
              << " memsize=" << tensor.sysMem[0].memSize
              << " addr="    << tensor.sysMem[0].virAddr;
            //   << std::endl;
    // if (prop.quantiType == SCALE) {
    //     std::cout << "scale=" << *(prop.scale.scaleData);
    // }
    std::cout << std::endl;
}


#endif // PERCEPTION_COMMON_HBDNN_UTILS_H_