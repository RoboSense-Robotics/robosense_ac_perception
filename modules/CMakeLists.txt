cmake_minimum_required(VERSION 3.8)
project(robosense_ac_perception)
set(CMAKE_BUILD_TYPE Release)
option(ENABLE_DEBUG_SYMBOL "DEBUG SYMBOL" OFF)
if (ENABLE_DEBUG_SYMBOL)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -g")
endif()
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

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include(cmake/check_platform.cmake)
include(cmake/find_inferlib.cmake)
include(cmake/yolov8.cmake)
include(cmake/pv_post_process.cmake)
include(cmake/ppseg.cmake)
include(cmake/promptda.cmake)

find_package(catkin QUIET)  # ROS1 核心包
find_package(ament_cmake QUIET)  # ROS2 核心包

if(catkin_FOUND)
  set(ROS_VERSION "ROS1")
elseif(ament_cmake_FOUND)
  set(ROS_VERSION "ROS2")
endif()

if(ROS_VERSION STREQUAL "ROS1")
  message(STATUS "Building with ROS1")
  add_definitions(-DUSE_ROS1)  # 定义ROS1宏
  add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}")
  message("PROJECT_PATH: ${PROJECT_SOURCE_DIR}")
  find_package(catkin REQUIRED COMPONENTS
    sensor_msgs
    roscpp
    std_msgs
    pcl_conversions
    cv_bridge
    perception_msgs
    visualization_msgs
  )
  include_directories(${catkin_INCLUDE_DIRS})
  catkin_package()
elseif(ROS_VERSION STREQUAL "ROS2")
  message(STATUS "Building with ROS2")
  add_definitions(-DUSE_ROS2)  # 定义ROS2宏

  # find dependencies
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(cv_bridge REQUIRED)

  find_package(perception_msgs REQUIRED)
  find_package(robosense_msgs REQUIRED)
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
endif()

# yolov8 node
add_executable(yolov8_node src/node/yolov8_node.cpp)
target_include_directories(yolov8_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(ROS_VERSION STREQUAL "ROS1")
    target_link_libraries(yolov8_node yolov8 ${catkin_LIBRARIES})
elseif(ROS_VERSION STREQUAL "ROS2")
  target_link_libraries(yolov8_node yolov8)
  ament_target_dependencies(yolov8_node
    rclcpp
    std_msgs
    cv_bridge
    perception_msgs
    robosense_msgs
  )
  install(TARGETS yolov8_node
    DESTINATION lib/${PROJECT_NAME})
endif()

# post process
add_executable(pv_post_process_node src/node/pv_post_process_node.cpp)
target_include_directories(pv_post_process_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(ROS_VERSION STREQUAL "ROS1")
  target_link_libraries(pv_post_process_node pv_post_process ${catkin_LIBRARIES})
elseif(ROS_VERSION STREQUAL "ROS2")
  target_link_libraries(pv_post_process_node pv_post_process)
  ament_target_dependencies(pv_post_process_node
    rclcpp
    std_msgs
    perception_msgs
    robosense_msgs
    PCL
    pcl_conversions
    visualization_msgs
    cv_bridge
  )
  install(TARGETS pv_post_process_node
    DESTINATION lib/${PROJECT_NAME})
endif()

# ppseg_node
add_executable(ppseg_node src/node/ppseg_node.cpp)
target_include_directories(ppseg_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(ROS_VERSION STREQUAL "ROS1")
  target_link_libraries(ppseg_node ppseg ${catkin_LIBRARIES})
elseif(ROS_VERSION STREQUAL "ROS2")
  target_link_libraries(ppseg_node ppseg)
  ament_target_dependencies(ppseg_node
    rclcpp
    std_msgs
    cv_bridge
    robosense_msgs
  )
  install(TARGETS ppseg_node
          DESTINATION lib/${PROJECT_NAME})
endif()

# promptda node
add_executable(promptda_node src/node/promptda_node.cpp)
target_include_directories(promptda_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(ROS_VERSION STREQUAL "ROS1")
  target_link_libraries(promptda_node promptda ${catkin_LIBRARIES})
elseif(ROS_VERSION STREQUAL "ROS2")
  target_link_libraries(promptda_node promptda)
  ament_target_dependencies(promptda_node
    rclcpp
    std_msgs
    cv_bridge
    robosense_msgs
  )
  install(TARGETS promptda_node
    DESTINATION lib/${PROJECT_NAME})
endif()

#install other files
if(ROS_VERSION STREQUAL "ROS1")
#
elseif(ROS_VERSION STREQUAL "ROS2")
  install(DIRECTORY
  launch
  rviz2_config
  config
  DESTINATION share/${PROJECT_NAME})
endif()

if(ROS_VERSION STREQUAL "ROS2")
  ament_package()
endif()
