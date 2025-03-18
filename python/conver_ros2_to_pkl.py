import rosbag
import numpy as np
import pickle
import os
from scipy.spatial.transform import Rotation
import sensor_msgs.point_cloud2 as pc2
import cv2
import yaml
import argparse

def xyzquat2transformat(xyz, quat):
    mat = np.eye(4)  # 创建一个 4x4 单位矩阵
    mat[:3, 3] = xyz  # 将前 3 个元素（平移向量）赋值给最后一列
    R = Rotation.from_quat(quat).as_matrix()  # 将四元数转换成旋转矩阵
    mat[:3, :3] = R  # 将旋转矩阵赋值给 4x4 矩阵的前 3x3 部分
    return mat

def binary_to_rgb_image(binary_data, width, height):
    # 确保二进制数据长度是3的倍数，因为每个像素有3个字节 (R, G, B)
    if len(binary_data) != width * height * 3:
        raise ValueError("二进制数据的长度与给定的图像尺寸不匹配")

    # 将二进制数据转换为一个 NumPy 数组，数据类型是 uint8
    rgb_values = np.frombuffer(binary_data, dtype=np.uint8)

    # 将数组重新形状为 (height, width, 3)，符合 OpenCV 的 BGR 格式
    rgb_values = rgb_values.reshape((height, width, 3))

    # OpenCV 默认是 BGR 格式，如果需要显示 RGB 图片，可以直接使用此格式
    return rgb_values


def parse_arguments():
    parser = argparse.ArgumentParser(description="start convert rosbag2 to pkl")
    parser.add_argument("--ros2path", type=str, required=True, help="ros2 path")
    parser.add_argument("--pklpath", type=str, required=True, help="output pkl path")
    parser.add_argument("--ros1path", type=str, required=True, help="output ros1 bag path")
    parser.add_argument("--calib", type=str, required=True, help="calib file path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    ros2path = args.ros2path
    ros1path = args.ros1path
    pklpath = args.pklpath
    calib_path = args.calib
    cm = 'rosbags-convert ' + ' --src ' + ros2path + ' --dst ' + ros1path
    print(cm)
    os.system(cm)

    # read calib
    with open(calib_path, 'r') as file:
        config = yaml.safe_load(file)

    camera_k = np.array(config['Sensor']['Camera']['intrinsic']['int_matrix']).reshape(3, 3)
    camera_d = np.array(config['Sensor']['Camera']['intrinsic']['dist_coeff'])
    image_width, image_height = config['Sensor']['Camera']['intrinsic']['image_size']
    camera2imu = config['Sensor']['Camera']['extrinsic']
    camera2imu_xyz = np.array(
        [camera2imu['translation']['x'], camera2imu['translation']['y'], camera2imu['translation']['z']])
    camera2imu_quat = np.array(
        [camera2imu['quaternion']['x'], camera2imu['quaternion']['y'], camera2imu['quaternion']['z'],
         camera2imu['quaternion']['w']])
    camera2imu_rt = xyzquat2transformat(camera2imu_xyz, camera2imu_quat)

    lidar2imu = config['Sensor']['Lidar']['extrinsic']
    lidar2imu_xyz = np.array(
        [lidar2imu['translation']['x'], lidar2imu['translation']['y'], lidar2imu['translation']['z']])
    lidar2imu_quat = np.array([lidar2imu['quaternion']['x'], lidar2imu['quaternion']['y'], lidar2imu['quaternion']['z'],
                               lidar2imu['quaternion']['w']])
    lidar2imu_rt = xyzquat2transformat(lidar2imu_xyz, lidar2imu_quat)

    imu2lidar_rt = np.linalg.inv(lidar2imu_rt)
    camera2lidar_rt = np.dot(imu2lidar_rt, camera2imu_rt)
    lidar2camera_rt = np.linalg.inv(camera2lidar_rt)

    bag = rosbag.Bag(ros1path, 'r')
    img_ts = []
    img_data = []
    lidar_ts = []
    lidar_data = []
    for topic, msg, t in bag.read_messages():
        ts = int(str(msg.header.stamp))
        if topic == '/rs_lidar/points':
            pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pc_list = list(pc_data)
            pc_array = np.array(pc_list)  # 将数据转换为 numpy 数组
            lidar_ts.append(ts)
            lidar_data.append(pc_array)

        if topic == '/rs_camera/rgb':
            image_encoded = np.frombuffer(msg.data, dtype='u1')
            image_decoded = binary_to_rgb_image(image_encoded, image_width, image_height)
            img_ts.append(ts)
            img_data.append(image_decoded)

    camera_info = {'intrinsic': camera_k, 'ego2sensor_rt': lidar2camera_rt, 'd': camera_d}
    lidar_ts = np.array(lidar_ts)
    img_ts = np.array(img_ts)
    img_indices = [np.abs(img_ts - x).argmin() for x in lidar_ts]
    pc_img = []
    pc_ts_img_ts = []
    for pc, pc_ts, img_idx in zip(lidar_data, lidar_ts, img_indices):
        pc_img.append([pc, img_data[img_idx]])
        pc_ts_img_ts.append([pc_ts, img_ts[img_idx]])
    res = {'pc_img': pc_img, 'camera_info': camera_info, 'pc_ts_img_ts': pc_ts_img_ts}
    with open(pklpath,
              'wb') as f:
        pickle.dump(res, f)
