import cv2
import torch
from promptda.promptda import PromptDA
import pickle
from tqdm import tqdm
import numpy as np
from promptda.utils.depth_utils import undistort_img, ego2img
import argparse


def knn_pooling(input_matrix, pool_size, k):
    # 输入矩阵的高度和宽度
    height, width = input_matrix.shape
    # 池化窗口的高和宽
    indices = np.indices((height, width)).transpose(1, 2, 0)
    # 创建输出矩阵
    output_matrix = np.zeros((height, width))
    # 在输入矩阵上滑动池化窗口
    for i in range(height):
        for j in range(width):
            if input_matrix[i, j] == 0:
                # 当前池化窗口的区域
                top = max(i - (pool_size) // 2, 0)
                down = min(i + (pool_size) // 2 + 1, height)
                left = max(j - (pool_size) // 2, 0)
                right = min(j + (pool_size) // 2 + 1, width)
                region = input_matrix[top:down, left:right]
                pool_indices = indices[top:down, left:right, :]
                non_zero_in_region_i, non_zero_in_region_j, = np.nonzero(region)
                if len(non_zero_in_region_i) == 0:
                    output_matrix[i, j] = 0
                else:
                    depth_cab = region[non_zero_in_region_i, non_zero_in_region_j]
                    pool_indices_cab = pool_indices[non_zero_in_region_i, non_zero_in_region_j, :]
                    dis_indices_cab = np.linalg.norm(pool_indices_cab - np.array([i, j])[None, :], axis=1)
                    top_k_indices = np.argsort(dis_indices_cab)[:k]
                    res_depth = depth_cab[top_k_indices].mean()
                    output_matrix[i, j] = res_depth
            else:
                output_matrix[i, j] = input_matrix[i, j]

    return output_matrix

def parse_arguments():
    parser = argparse.ArgumentParser(description="run PromptDa")
    parser.add_argument("--model_res_save_path", type=str, required=True, help="model res save path")
    parser.add_argument("--input_info_path", type=str, required=True, help="input info path")
    parser.add_argument("--ckpt_path", type=str, required=True, help="ckpt path", default='./assets/model_big.ckpt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    model = PromptDA(encoder='vitl', ckpt_path=args.ckpt_path).to('cuda').eval()
    model_res_save_path = args.model_res_save_path
    input_info_path = args.input_info_path
    model_res = []
    with open(input_info_path, 'rb') as f:
        input_info = pickle.load(f)

    # model input config
    origin_H = 1080
    origin_W = 1920
    crop_start_y = 140
    crop_end_y = 140 + 840
    crop_start_x = 0
    crop_end_x = 1904
    img_scale_factor = 1
    lidar_project_scale_factor = 1 / 7
    knn_pooling_scale = 18
    knn_k = 4

    depth_imgs = input_info['pc_img']
    camera_info = input_info['camera_info']
    camera_k = camera_info['intrinsic']
    camera_d = camera_info['d']
    ego2sensor_rt = camera_info['ego2sensor_rt']

    lidar_project_k = camera_k.copy()
    lidar_project_k[0, 2] = lidar_project_k[0, 2] - crop_start_x
    lidar_project_k[1, 2] = lidar_project_k[1, 2] - crop_start_y
    lidar_project_k = lidar_project_k * lidar_project_scale_factor
    lidar_project_k[2, 2] = 1

    for depth_img in tqdm(depth_imgs):

        img, pc = depth_img[1], depth_img[0]
        img = undistort_img(img, camera_k, camera_d)
        img = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :]
        lidar_w, lidar_h = int(img.shape[1] * lidar_project_scale_factor), int(
            img.shape[0] * lidar_project_scale_factor)

        if img_scale_factor != 1:
            img = cv2.resize(img, None, fx=img_scale_factor, fy=img_scale_factor, interpolation=cv2.INTER_LINEAR)

        input_img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        input_img = torch.from_numpy(input_img).unsqueeze(0).cuda() / 255
        pc_img = ego2img(pc, ego2sensor_rt, lidar_project_k)
        pc_x, pc_y, pc_z = pc_img[:, 0], pc_img[:, 1], pc_img[:, 2]
        pc_x, pc_y = pc_x.astype(np.int32), pc_y.astype(np.int32)
        pc_mask = (pc_x >= 0) & (pc_x < lidar_w) & (pc_y >= 0) & (pc_y < lidar_h)
        pc_x, pc_y, pc_z = pc_x[pc_mask], pc_y[pc_mask], pc_z[pc_mask]
        sorted_flat_xy_index = pc_y * lidar_w + pc_x
        _, unique_index = np.unique(sorted_flat_xy_index, return_index=True)
        pc_x, pc_y, pc_z = pc_x[unique_index], pc_y[unique_index], pc_z[unique_index]
        sparse_depth = np.zeros(
            (lidar_h, lidar_w))
        sparse_depth[pc_y, pc_x] = pc_z
        dense_depth = knn_pooling(sparse_depth, knn_pooling_scale, knn_k)
        dense_depth = torch.from_numpy(dense_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        depth = model.predict(input_img, dense_depth)  # HxW, depth in meters
        model_res.append(np.squeeze(depth.detach().cpu().numpy()))

    with open(model_res_save_path, 'wb') as file:
        pickle.dump(model_res, file)
