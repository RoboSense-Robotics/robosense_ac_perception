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
#ifndef PERCEPTION_COMMON_FACTORY_H_
#define PERCEPTION_COMMON_FACTORY_H_
#include <memory>
#include <unordered_map>
template <typename T>
class Factory {
public:
  using Creator = std::function<std::unique_ptr<T>()>;

  static void RegisterClass(const std::string& class_name, Creator creator) {
    GetRegistry().emplace(class_name, std::move(creator));
  }

  static std::unique_ptr<T> Create(const std::string& class_name) {
      auto it = GetRegistry().find(class_name);

      if (it != GetRegistry().end()) {
          return it->second();
      }
      for (const auto& pair: GetRegistry()) {
        std::cout << "Already registered Class:" << pair.first << std::endl;
      }
      std::cout << class_name << " not register!" << std::endl;
      return nullptr;
  }

private:
  static std::unordered_map<std::string, Creator>& GetRegistry() {
      static std::unordered_map<std::string, Creator> registry;
      return registry;
  }
};

#define REGISTER_CLASS(ParaentClass, ClassName) \
    static bool ClassName##_registered_ = [] { \
        Factory<ParaentClass>::RegisterClass(#ClassName, [] { return std::make_unique<ClassName>(); }); \
        return true; \
    }()
#endif  // PERCEPTION_COMMON_FACTORY_H_
