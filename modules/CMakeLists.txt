cmake_minimum_required(VERSION 3.8)
project(robosense_ac_perception)

# 通用警告设置
if(MSVC)
    # MSVC编译器设置
    add_compile_options(/W4)  # 相当于GCC的-Wall，开启所有常规警告
    # add_compile_options(/WX)  # 可选：将警告视为错误
elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC/Clang编译器设置
    add_compile_options(-Wall -Wextra)
    # add_compile_options(-Werror)  # 可选：将警告视为错误
endif()
add_compile_definitions(NOMINMAX)
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(perception_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(CMAKE_BUILD_TYPE Release)

include(cmake/check_platform.cmake)
include(cmake/yolov8.cmake)
include(cmake/pv_post_process.cmake)
include(cmake/ppseg.cmake)
include(cmake/promptda.cmake)

# yolov8 node
add_executable(yolov8_node src/node/yolov8_node.cpp)
target_include_directories(yolov8_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(yolov8_node
  yolov8
)

ament_target_dependencies(yolov8_node
  rclcpp
  std_msgs
  perception_msgs
)

install(TARGETS yolov8_node
  DESTINATION lib/${PROJECT_NAME})

#post process
add_executable(pv_post_process_node src/node/pv_post_process_node.cpp)
target_include_directories(pv_post_process_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(pv_post_process_node
  pv_post_process
)
ament_target_dependencies(pv_post_process_node
  rclcpp
  std_msgs
  perception_msgs
  PCL
  pcl_conversions
  visualization_msgs
  cv_bridge
)
install(TARGETS pv_post_process_node
  DESTINATION lib/${PROJECT_NAME})

# ppseg_node
add_executable(ppseg_node src/node/ppseg_node.cpp)
target_include_directories(ppseg_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(ppseg_node
  ppseg
)

ament_target_dependencies(ppseg_node
  rclcpp
  std_msgs
  perception_msgs
)

install(TARGETS ppseg_node
  DESTINATION lib/${PROJECT_NAME})

# promptda node
add_executable(promptda_node src/node/promptda_node.cpp)
target_include_directories(promptda_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(promptda_node
  promptda
)

ament_target_dependencies(promptda_node
  rclcpp
  std_msgs
  perception_msgs
)

install(TARGETS promptda_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
 launch
 rviz2_config
 config
 DESTINATION share/${PROJECT_NAME})

ament_package()
