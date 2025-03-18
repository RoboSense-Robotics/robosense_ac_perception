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
#ifndef PERCEPTION_COMMON_TRT_UTILS_H_
#define PERCEPTION_COMMON_TRT_UTILS_H_

#include "NvInfer.h"

class TrtLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) throw() {
      if (severity != Severity::kINFO && severity != Severity::kVERBOSE) {
        std::cout << msg << std::endl;
      }
    }
  };

inline size_t dataTypeSize(nvinfer1::DataType dataType) {
    switch (dataType) {
      case nvinfer1::DataType::kINT32:
      case nvinfer1::DataType::kFLOAT:
        return 4U;
      case nvinfer1::DataType::kHALF:
        return 2U;
      case nvinfer1::DataType::kBOOL:
      case nvinfer1::DataType::kUINT8:
      case nvinfer1::DataType::kINT8:
        return 1U;
    }
    return 0;
  }

  #define BASE_CUDA_CHECK(condition) \
    { GPUAssert((condition), __FILE__, __LINE__); }

  inline void GPUAssert(cudaError_t code, const char* file, int line,
                        bool abort = true) {
    if (code != cudaSuccess) {
      std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file
                  << " " << line << std::endl;
      if (abort) {
        exit(code);
      }
    }
  }
#endif // PERCEPTION_COMMON_TRT_UTILS_H_
