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
#ifndef PERCEPTION_COMMON_UTILS_H_
#define PERCEPTION_COMMON_UTILS_H_

#include<vector>
#include<cassert>
constexpr size_t MAX_DIMS_IN = 16;
struct TensorAttr {
    TensorAttr() {
        dims.resize(MAX_DIMS_IN);
    }
    TensorAttr(uint32_t n) : n_dims(n), zp(0), scale(1.0) {
      dims.resize(MAX_DIMS_IN);
  }
    TensorAttr(uint32_t n, int32_t zp_, float scale_) : n_dims(n), zp(zp_), scale(scale_) {
        dims.resize(MAX_DIMS_IN);
    }

    TensorAttr(float* scale_point) : scale_pointer(scale_point) {}

    void setDims(uint32_t index, uint32_t num) {
      assert(index < MAX_DIMS_IN);
      dims[index] = num;
    }

    void setSize(uint32_t size) {
      data_size = size;
    }
    ~TensorAttr() {
        dims.clear();
    }
    uint32_t n_dims;
    std::vector<uint32_t> dims;
    uint32_t data_size;
    int32_t zp;
    float scale;
    float* scale_pointer;
};

#endif // PERCEPTION_COMMON_UTILS_H_
