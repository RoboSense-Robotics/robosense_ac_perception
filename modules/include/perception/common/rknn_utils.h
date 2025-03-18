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
#ifndef PERCEPTION_COMMON_RKNN_UTILS_H_
#define PERCEPTION_COMMON_RKNN_UTILS_H_

#include "rknn_api.h"

#define BASE_RKNN_CHECK(condition) \
  { RKNNAssert((condition), __FILE__, __LINE__); }

inline void RKNNAssert(int code, const char* file, int line,
                      bool abort = true) {
  if (code != RKNN_SUCC) {
    std::cout << "RKNNAssert: " << " RKNN failed ret = " << code << " in " << file
                << " " << line << std::endl;
    if (abort) {
      exit(code);
    }
  }
}

void DumpTensorAttr(rknn_tensor_attr tenso_attr) {
  std::stringstream dims;
  dims << "[";
  for (size_t i = 0; i < tenso_attr.n_dims-1; ++i) {
    dims << tenso_attr.dims[i] << ", ";
  }
  dims << tenso_attr.dims[tenso_attr.n_dims-1] << "]";
  std::cout << "index="    << tenso_attr.index
              << " name="    << tenso_attr.name
              << " n_dims="  << tenso_attr.n_dims
              << " dims="    << dims.str()
              << " n_elems=" << tenso_attr.n_elems
              << " size="    << tenso_attr.size
              << " fmt="     << get_format_string(tenso_attr.fmt)
              << " type="    << get_type_string(tenso_attr.type)
              << " qnt_type="<< get_qnt_type_string(tenso_attr.qnt_type)
              << " zp="      << tenso_attr.zp
              << " scale="   << tenso_attr.scale
              << std::endl;
}

void rknn_nchw_2_nhwc(void* nchw, void* nhwc, int N, int C, int H, int W)
{
    auto nchw_f = reinterpret_cast<float*>(nchw);
    auto nhwc_f = reinterpret_cast<float*>(nhwc);
    for (int ni = 0; ni < N; ni++)
    {
        for (int hi = 0; hi < H; hi++)
        {
            for (int wi = 0; wi < W; wi++)
            {
                for (int ci = 0; ci < C; ci++)
                {
                    nhwc_f[ni * H * W * C + hi * W * C + wi * C + ci] = nchw_f[ni * C * H * W + ci * H * W + hi * W + wi];
                }
            }
        }
    }
}
#endif // PERCEPTION_COMMON_RKNN_UTILS_H_
