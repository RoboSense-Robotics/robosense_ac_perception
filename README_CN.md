# robosense_ac_perception

[README](README.md) | [中文文档](README_CN.md)

## 1. 简介

`robosense_ac_perception` 提供感知功能，例如目标检测、语义分割、稠密深度估计等

## 2. 前置依赖

此项目既支持 ROS1 也支持 ROS2，其中 ROS1 基于 noteic 版本测试通过，ROS2 基于 humble 版本测试通过。

如果在 x86 平台使用，依赖 Nvidia 显卡（比如3060），请确保安装了 `CUDA` 和 `TensorRT`（推荐安装版本为 8.5.2.2）;

如果在 arm 平台使用，已经适配以下平台，

* RK3588 基于 RKNN 版本 2.3.0
* JETSON Orin 基于 Jetpack 6.2
* RDK X5 基于 OS Version 3.1.1

### 2.1 安装 ROS1 或 ROS2

请依照官方说明安装 ROS

* [ROS1 安装](http://wiki.ros.org/noetic/Installation/Ubuntu)

* [ROS2 安装](https://docs.ros.org/en/humble/Installation.html)

## 3. 安装部署

### 3.1 代码拉取

您可以创建一个新的文件夹或进入您现有的工作空间，执行以下命令将代码拉取到工作空间内。

```bash
cd WORKSPACE_PATH/src
mkdir ac_studio && cd ac_studio
git clone https://github.com/RoboSense-Robotics/robosense_ac_perception.git -b main
```
### 3.2 模型下载

```shell
cd robosense_ac_perception
bash download_model.sh
```

### 3.3 安装依赖

可以通过 `rosdep` 工具安装 `robosense_ac_perception` 编译所需的依赖

```bash
cd ac_studio
rosdep install --from-paths robosense_ac_perception --ignore-src -r -y
```
请先按照[说明](https://github.com/RoboSense-Robotics/robosense_ac_ros2_sdk_infra/blob/main/modules/ac_driver/README.md)编译 ROS2 所需 robosense_msgs，ROS1 则无需编译。
### 3.4 编译 robosense_ac_perception

####  基于 ROS1 环境

在工作空间下执行以下命令来编译安装 `robosense_ac_perception`

```
cd WORKSPACE_PATH
catkin build perception_msgs
catkin build robosense_ac_perception
```

编译安装完成后，推荐刷新一下工作空间的 `bash profile`，确保组件功能正常

```
source devel/setup.bash
```

#### 基于 ROS2 环境

在工作空间下执行以下命令来编译安装 `robosense_ac_perception`

```bash
cd WORKSPACE_PATH
colcon build --symlink-install --parallel-workers 8 --packages-select perception_msgs robosense_ac_perception
```

编译安装完成后，推荐刷新一下工作空间的 `bash profile`，确保组件功能正常

```bash
source install/setup.bash
```

## 4. 运行
### 4.1 获取数据
可以连接 Active Camera 在线获取数据，或者离线播放数据包进行测试。
#### 4.1.1 运行 Active Camera

参考 [文档](https://github.com/RoboSense-Robotics/robosense_ac_ros2_sdk_infra/tree/main/modules/ac_driver) 运行超级传感器节点，实时获取数据，按照文档说明在终端设置对应环境变量，启用零拷贝模式或非零拷贝模式获取数据。

#### 4.1.2 离线播放数据
使用 `ros` 命令播放数据包，例如：

``` bash
# ros1
rosbag play BAG_PATH
# ros2
ros2 bag play BAG_PATH
```

### 4.2 运行节点

#### 4.2.1 运行目标检测节点

通过 `ros` 命令运行 `yolov8_node` 节点

```bash
# ros1
roslaunch modules/launch/yolov8.launch
# ros2
ros2 launch robosense_ac_perception start_yolov8.py
```
#### 4.2.2 运行语义分割节点

通过 `ros` 命令运行 `ppseg_node` 节点

```bash
# ros1
roslaunch modules/launch/ppseg.launch
# ros2
ros2 launch robosense_ac_perception start_ppseg.py
```

#### 4.2.3 运行稠密深度估计节点

通过 `ros` 命令运行 `promptda_node` 节点

```bash
# ros1
roslaunch modules/launch/promptda.launch
# ros2
ros2 launch robosense_ac_perception start_promptda.py
```


## 5. FAQ

[Create New Issue](https://github.com/RoboSense-Robotics/robosense_ac_perception/issues/new)

## 6. 开源许可

[The Apache License, version 2.0.](https://www.apache.org/licenses/LICENSE-2.0)