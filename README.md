# robosense_ac_perception

[README](README.md) | [中文文档](README_CN.md)

## 1. Introduction

`robosense_ac_perception` provides perception capabilities such as object detection, semantic segmentation, dense depth estimation, etc.

## 2. Prerequisites

This project supports both ROS1 and ROS2. It has been tested on the Noetic version of ROS1 and the Humble version of ROS2.

For usage on the x86 platform, an Nvidia GPU (e.g., 3060) is required, along with the installation of `CUDA` and `TensorRT` (recommended version 8.5.2.2).

For the ARM platform, it has been adapted to the following platforms:

* RK3588 platform based on RKNN version 2.3.0
* JETSON Orin based on Jetpack 6.2
* RDK X5 based on OS Version 3.1.1

### 2.1 Install ROS1 or ROS2

Please follow the official instructions to install ROS:

* [ROS1 Installation](http://wiki.ros.org/noetic/Installation/Ubuntu)

* [ROS2 Installation](https://docs.ros.org/en/humble/Installation.html)

## 3. Installation & Deployment

### 3.1 Pulling the Code

Create a new folder or enter your existing workspace, and execute the following commands to pull the code into the workspace.

```bash
cd WORKSPACE_PATH/src
mkdir ac_studio && cd ac_studio
git clone https://github.com/RoboSense-Robotics/robosense_ac_perception.git -b main
```

### 3.2 Model Download

```shell
cd robosense_ac_perception
bash download_model.sh
```

### 3.3 Installing Dependencies

Use the `rosdep` tool to install the dependencies required for compiling `robosense_ac_perception`.

```bash
cd ac_studio
rosdep install --from-paths robosense_ac_perception --ignore-src -r -y
```
Please follow the [instructions](https://github.com/RoboSense-Robotics/robosense_ac_ros2_sdk_infra/blob/main/modules/ac_driver/README.md) to compile the required robosense_msgs for ROS2. Compilation is not needed for ROS1.


### 3.4 Compile robosense_ac_perception

#### For ROS1 Environment

Execute the following commands in the workspace to compile and install `robosense_ac_perception`.

```bash
cd WORKSPACE_PATH
catkin build perception_msgs
catkin build robosense_ac_perception
```

After compilation and installation, it's recommended to refresh the workspace's `bash profile` to ensure component functionality.

```bash
source devel/setup.bash
```

#### For ROS2 Environment

Execute the following command in the workspace to compile and install `robosense_ac_perception`.

```bash
cd WORKSPACE_PATH
colcon build --symlink-install --parallel-workers 8 --packages-select perception_msgs robosense_ac_perception
```

After compilation and installation, it's recommended to refresh the workspace's `bash profile` to ensure component functionality.

```bash
source install/setup.bash
```

## 4. Running
### 4.1 Acquiring Data
Data can be obtained either by connecting to an Active Camera online or by playing offline data packets for testing.
#### 4.1.1 Running the Active Camera

Refer to the [documentation](https://github.com/RoboSense-Robotics/robosense_ac_ros2_sdk_infra/tree/main/modules/ac_driver) to run the Super Sensor node for real-time data acquisition. Follow the documentation instructions to set the corresponding environment variables in the terminal to enable zero-copy mode or non-zero-copy mode for data acquisition.

#### 4.1.2 Offline Data Playback
Use the `ros` command to play back data packets, for example:

```bash
# ROS1
rosbag play BAG_PATH
# ROS2
ros2 bag play BAG_PATH
```

### 4.2 Running Nodes

#### 4.2.1 Running Object Detection Node

Run the `yolov8_node` node using the `ros` command:

```bash
# ROS1
roslaunch modules/launch/yolov8.launch
# ROS2
ros2 launch robosense_ac_perception start_yolov8.py
```
#### 4.2.2 Running Semantic Segmentation Node

Run the `ppseg_node` node using the `ros` command:

```bash
# ROS1
roslaunch modules/launch/ppseg.launch
# ROS2
ros2 launch robosense_ac_perception start_ppseg.py
```

#### 4.2.3 Running Dense Depth Estimation Node

Run the `promptda_node` node using the `ros` command:

```bash
# ROS1
roslaunch modules/launch/promptda.launch
# ROS2
ros2 launch robosense_ac_perception start_promptda.py
```

## 5. FAQ

[Create New Issue](https://github.com/RoboSense-Robotics/robosense_ac_perception/issues/new)

## 6. Open Source License

[The Apache License, version 2.0.](https://www.apache.org/licenses/LICENSE-2.0)