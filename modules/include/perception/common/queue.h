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
#ifndef PERCEPTION_SYNCQUEUE_H
#define PERCEPTION_SYNCQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
template <typename T>
class SyncQueue {
public:
  SyncQueue() {
    max_size_ = 30;
  }
  SyncQueue(size_t max_size) : max_size_(max_size) {}
  ~SyncQueue() {
    while (!queue_.empty()) {
      queue_.pop_front();
    }
  }

  void push(const T& value) {
    std::unique_lock<std::mutex> lock(mtx_);
    queue_.push_back(value);
    // std::cout << "queue size : " << queue_.size() << std::endl;
    if (queue_.size() > max_size_) {
      queue_.pop_front();
    }
    cv_.notify_one();
  }

  T pop() {
    T value;
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    if (!queue_.empty()) {
      value = queue_.front();
      queue_.pop_front();
    }
    return value;
  }

  bool empty() const {
    std::unique_lock<std::mutex> lock(mtx_);
    return queue_.empty();
  }

  inline size_t size() {
    std::unique_lock<std::mutex> lock(mtx_);
    return queue_.size();
  }

private:
  std::deque<T> queue_;
  size_t max_size_;
  std::mutex mtx_;
  std::condition_variable cv_;
};
#endif // PERCEPTION_SYNCQUEUE_H
