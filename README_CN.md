# robosense_ac_perception

[README](README.md) | [中文文档](README_CN.md)

## 1. 简介

`robosense_ac_perception` 提供感知功能，例如目标检测、语义分割、稠密深度估计等

## 2. 前置依赖

此项目基于 `ros2 humble` 进行开发测试，并依赖于 `ros2 humble` 版本的依赖库。

如果在 x86 平台使用，依赖 Nvidia 显卡（比如3060），请确保安装了 `CUDA` 和 `TensorRT`（推荐安装版本为 8.5.2.2）; 如果在 arm 平台使用，依赖 rk3588 硬件平台，确保rknn版本为 2.3.0
### 2.1 安装 ros2

根据您的操作系统选择 [官方教程](https://docs.ros.org/en/humble/Installation.html) 中的指定内容进行执行

## 3. 安装部署

### 3.1 代码拉取

您可以创建一个新的文件夹或进入您现有的 `ros2` 工作空间，执行以下命令将代码拉取到工作空间内，关于工作空间的创建，请参考[官方文档](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)。

```bash
cd WORKSPACE_PATH
mkdir ac_studio && cd ac_studio
# http
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

编译 pcl_conversions 和 pcl_msg 模块
```bash
#如果没有安装pcl，先安装
sudo apt install libpcl-dev
git clone https://github.com/ros-perception/pcl_msgs.git -b ros2
git clone https://github.com/ros-perception/perception_pcl.git -b humble
# 进入工作空间
cd WORKSPACE_PATH
colcon build --symlink-install
```

### 3.4 编译 robosense_ac_perception

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
可以连接超级传感器在线获取数据，或者离线播放数据包进行测试。
#### 4.1.1 运行超级传感器

参考 [文档](https://github.com/RoboSense-Robotics/robosense_ac_ros2_sdk_infra/tree/main/modules/ac_driver) 运行超级传感器节点，实时获取数据

#### 4.1.2 离线播放数据
使用 `ros2 bag` 命令播放数据包，例如：

``` bash
ros2 bag play BAG_PATH
```

### 4.2 运行节点

#### 4.2.1 运行目标检测节点

通过 `ros2 launch` 命令可以运行 `yolov8_node` 节点

```bash
ros2 launch robosense_ac_perception start_yolov8.py
```
#### 4.2.2 运行语义分割节点

通过 `ros2 launch` 命令可以运行 `ppseg_node` 节点

```bash
ros2 launch robosense_ac_perception start_ppseg.py
```

#### 4.2.3 运行稠密深度估计节点

通过 `ros2 launch` 命令可以运行 `promptda_node` 节点

```bash
ros2 launch robosense_ac_perception start_promptda.py
```


## 5. FAQ

[Create New Issue](https://github.com/RoboSense-Robotics/robosense_ac_perception/issues/new)

## 6. 开源许可

[The Apache License, version 2.0.](https://www.apache.org/licenses/LICENSE-2.0)