import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.interpolate import CubicSpline
import cv2


def xyzquat2transformat(xyz, quat):
    mat = np.eye(4)  # 创建一个 4x4 单位矩阵
    mat[:3, 3] = xyz  # 将前 3 个元素（平移向量）赋值给最后一列
    R = Rotation.from_quat(quat).as_matrix()  # 将四元数转换成旋转矩阵
    mat[:3, :3] = R  # 将旋转矩阵赋值给 4x4 矩阵的前 3x3 部分
    return mat


def undistort_img(img, K, D):
    if len(D) == 4:

        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D,
                                                           np.eye(3, dtype=np.float32), K,
                                                           (img.shape[1], img.shape[0]), cv2.CV_32FC1)

        img_und = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST)

    else:
        img_und = cv2.undistort(img, K, D)

    return img_und

def ego2img(points_lidar, ego2sensor_rt, intrinsic):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)

    ego2camera = ego2sensor_rt
    points_camera_homogeneous = points_lidar_homogeneous @ ego2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.01, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = intrinsic
    points_img = points_camera @ camera2img.T
    points_img[:, 2] = points_camera_homogeneous[:, 2]
    return points_img[valid]

def visualize_depth(depth: np.ndarray,
                    depth_min=None,
                    depth_max=None,
                    percentile=2,
                    ret_minmax=False,
                    cmap='Spectral'):
    depth=depth
    if depth_min is None: depth_min = np.percentile(depth, percentile)
    if depth_max is None: depth_max = np.percentile(depth, 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np

def unproject_depth(depth, 
                    ixt, 
                    depth_min=0.01, 
                    depth_max=None, 
                    color=None, 
                    ext=None, 
                    conf=None, 
                    ret_pcd=False, 
                    clip_box=None):
    height, width = depth.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = depth.reshape(-1)
    mask = np.ones_like(xx, dtype=np.bool_)
    if depth_min is not None:
        mask &= zz >= depth_min
    if depth_max is not None:
        mask &= zz <= depth_max
    if conf is not None:
        mask &= conf.reshape(-1) == 2
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    pcd = np.stack([xx, yy, np.ones_like(xx)], axis=1)
    pcd = pcd * zz[:, None]
    pcd = np.dot(pcd, np.linalg.inv(ixt).T)
    if ext is not None:
        pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
        pcd = np.dot(pcd, np.linalg.inv(ext).T)
    new_mask = np.ones_like(pcd[:, 0]).astype(np.bool_)
    if clip_box is not None:
        assert len(clip_box) == 6
        for i, val in enumerate(clip_box):
            if val is None:
                continue
            if i == 0: new_mask &= (pcd[:, 0] <= val)
            elif i == 1: new_mask &= (pcd[:, 1] <= val)
            elif i == 2: new_mask &= (pcd[:, 2] <= val)
            elif i == 3: new_mask &= (pcd[:, 0] >= val)
            elif i == 4: new_mask &= (pcd[:, 1] >= val)
            elif i == 5: new_mask &= (pcd[:, 2] >= val)
    if color is not None:
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.
        if ret_pcd:
            points = pcd
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3][new_mask])
            pcd.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3)[mask][new_mask])
        else:
            return pcd[:, :3][new_mask], color.reshape(-1, 3)[mask][new_mask]
    else:
        if ret_pcd:
            points = pcd
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd[:, :3][new_mask])
        else:
            return pcd[:, :3][new_mask]
    return pcd

def smooth_min_max(min_vals, 
                   max_vals, 
                   interval: int = 60):
    '''
    Slerp interpolate and smooth min and max values
    Args: 
        min_vals: list[float]
        max_vals: list[float]
    Returns: 
        min_vals_smooth: list[float]
        max_vals_smooth: list[float]
    '''

    key_frames = list(range(0, len(min_vals), interval))
    if key_frames[-1] != len(min_vals) - 1:
        key_frames.append(len(min_vals) - 1)

    key_frame_indices = np.array(key_frames)
    min_key_vals = np.array([min_vals[i] for i in key_frames])
    max_key_vals = np.array([max_vals[i] for i in key_frames])

    # Use CubicSpline for smooth interpolation
    min_spline = CubicSpline(
        key_frame_indices, min_key_vals, bc_type='natural')
    max_spline = CubicSpline(
        key_frame_indices, max_key_vals, bc_type='natural')

    x_full = np.arange(len(min_vals))
    min_vals_smooth = min_spline(x_full)
    max_vals_smooth = max_spline(x_full)
    # plt.plot(min_vals, label='min_vals')
    # plt.plot(min_vals_smooth, label='min_vals_smooth')
    # plt.legend()
    # plt.savefig('min_vals.png')
    return min_vals_smooth, max_vals_smooth

if __name__ == '__main__':
    depth = np.random.rand(100, 100)
    visualize_depth(depth)