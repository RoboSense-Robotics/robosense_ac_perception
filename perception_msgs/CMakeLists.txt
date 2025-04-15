cmake_minimum_required(VERSION 3.8)
project(perception_msgs)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(catkin QUIET)  # ROS1 核心包
find_package(ament_cmake QUIET)  # ROS2 核心包

if(catkin_FOUND)
  set(ROS_VERSION "ROS1")
elseif(ament_cmake_FOUND)
  set(ROS_VERSION "ROS2")
endif()

if (${ROS_VERSION} STREQUAL "ROS1")
  message("Building with ROS1")
  find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
  )
  add_message_files(
    FILES
    "BBox2D.msg"
    "ObjectInner.msg"
    "Size3D.msg"
    "Vec2D.msg"
    "Vec3D.msg"
    "ObjectInnerArray.msg"
  )
  generate_messages(
    DEPENDENCIES
    std_msgs
  )
  # catkin_package()

  catkin_package(
  #   # INCLUDE_DIRS include
  # #   #  LIBRARIES perception_ros_msgs
    CATKIN_DEPENDS std_msgs message_runtime
  # #   #  DEPENDS system_lib
  )

elseif(${ROS_VERSION} STREQUAL "ROS2")
  message("Building with ROS2")
  # find dependencies
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(rosidl_default_generators REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/BBox2D.msg"
    "msg/ObjectInner.msg"
    "msg/Size3D.msg"
    "msg/Vec2D.msg"
    "msg/Vec3D.msg"
    "msg/ObjectInnerArray.msg"
    DEPENDENCIES builtin_interfaces std_msgs
  )

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

  ament_package()
endif()