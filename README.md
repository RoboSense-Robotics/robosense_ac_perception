# robosense_ac_perception

[README](README.md) | [中文文档](README_CN.md)

## 1. Introduction

`robosense_ac_perception` provides perception capabilities such as object detection, semantic segmentation, dense depth estimation, etc.

## 2. Prerequisites

This project is developed and tested based on `ros2 humble`, and it depends on the libraries of the `ros2 humble` version.

For usage on the x86 platform, an Nvidia GPU (e.g., 3060) is required, along with the installation of `CUDA` and `TensorRT` (recommended version 8.5.2.2). For the arm platform, the rk3588 hardware platform is needed, ensuring that the rknn version is 2.3.0.
### 2.1 Install ROS2

Follow the instructions in the [official guide](https://docs.ros.org/en/humble/Installation.html) tailored for your operating system.

## 3. Installation & Deployment

### 3.1 Pulling the Code

Create a new folder or enter your existing `ros2` workspace, and execute the following commands to pull the code into the workspace. For information on creating a workspace, refer to the [official documentation](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html).

```bash
cd WORKSPACE_PATH
mkdir ac_studio && cd ac_studio
# http
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

Compile the pcl_conversions and pcl_msg modules:
```bash
# If pcl is not installed, install it first
sudo apt install libpcl-dev
git clone https://github.com/ros-perception/pcl_msgs.git -b ros2
git clone https://github.com/ros-perception/perception_pcl.git -b humble
# Enter the workspace
cd WORKSPACE_PATH
colcon build --symlink-install
```

### 3.4 Compile robosense_ac_perception

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
Data can be obtained either by connecting to a Super Sensor online or by playing offline data packets for testing.
#### 4.1.1 Running the Super Sensor

Refer to the [documentation](https://github.com/RoboSense-Robotics/robosense_ac_ros2_sdk_infra/tree/main/modules/ros_metas) to run the Super Sensor node for real-time data acquisition.

#### 4.1.2 Offline Data Playback
Use the `ros2 bag` command to play back data packets, for example:

```bash
ros2 bag play BAG_PATH
```

### 4.2 Running Nodes

#### 4.2.1 Running Object Detection Node

Run the `yolov8_node` node using the `ros2 launch` command:

```bash
ros2 launch robosense_ac_perception start_yolov8.py
```
#### 4.2.2 Running Semantic Segmentation Node

Run the `ppseg_node` node using the `ros2 launch` command:

```bash
ros2 launch robosense_ac_perception start_ppseg.py
```

#### 4.2.3 Running Dense Depth Estimation Node

Run the `promptda_node` node using the `ros2 launch` command:

```bash
ros2 launch robosense_ac_perception start_promptda.py
```

## 5. FAQ

[Create New Issue](https://github.com/RoboSense-Robotics/robosense_ac_perception/issues/new)

## 6. Open Source License

[The Apache License, version 2.0.](https://www.apache.org/licenses/LICENSE-2.0)