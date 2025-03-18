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
#ifndef PERCEPTION_COMMON_TIMERECORD_H_
#define PERCEPTION_COMMON_TIMERECORD_H_

#include <iostream>
#include <chrono>

namespace robosense {
namespace perception {
class TimeRecord {
 public:
  void init(std::string title) { title_ = title; }

  void tic() { start_point_ = std::chrono::high_resolution_clock::now(); }

  void toc() {
    // if (global_cnt < 0) {
    //   global_cnt++;
    //   return;
    // }
    // 1. cur time
    end_point_ = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds time_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_point_ -
                                                             start_point_);
    float time_micro = time_nano.count() / 1e6;

    // 2. max min
    max_val = std::max(time_micro, max_val);
    min_val = std::min(time_micro, min_val);

    // 3. mean
    sum_val += time_micro;
    cnt += 1;
    mean_val = sum_val / cnt;

    // 4. std

    std::cout << title_ << " " << cnt << ": min=" << min_val
          << " ms, max=" << max_val << " ms, "
          << "mean=" << mean_val << " ms, cur=" << time_micro << " ms\n";
  }

 public:
  std::string title_;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_point_,
      end_point_;

  int cnt = 0, global_cnt = 0;
  float max_val = std::numeric_limits<float>::min();
  float min_val = std::numeric_limits<float>::max();
  float sum_val = 0.;
  float mean_val = 0.;
};
}  // namespace perception
}  // namespace robosense
#endif  // PERCEPTION_COMMON_TIMERECORD_H_