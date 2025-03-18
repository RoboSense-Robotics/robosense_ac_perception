import imageio
import pickle
import open3d as o3d
import numpy as np
import cv2
from tqdm import tqdm
from promptda.utils.depth_utils import unproject_depth, ego2img, undistort_img
from promptda.utils.io_wrapper import visualize_depth
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="generate video")
    parser.add_argument("--input_info_path", type=str, required=True, help="input info path")
    parser.add_argument("--videoout_path", type=str, required=True, help="video out path")
    parser.add_argument("--model_res_save_path", type=str, required=True, help="model res save path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    videoout = args.videoout_path
    depth_path = args.model_res_save_path
    input_info_path = args.input_info_path

    # 创建o3d无窗口渲染器
    width, height = int(800 / 800 * 1904), int(600 / 800 * 1904)
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # 白色背景
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"  # 适用于点云
    mat.point_size = 2.0  # 设置点大小
    crop_image_pc_height = 200

    with open(depth_path, 'rb') as f:
        depth_res = pickle.load(f)
    with open(input_info_path, 'rb') as f:
        input_info = pickle.load(f)
    origin_H = 1080
    origin_W = 1920
    start_y = 140
    end_y = 140 + 840
    start_x = 0
    end_x = 1904
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
    lidar_project_k[0, 2] = lidar_project_k[0, 2] - start_x
    lidar_project_k[1, 2] = lidar_project_k[1, 2] - start_y
    lidar_project_k = lidar_project_k * lidar_project_scale_factor
    lidar_project_k[2, 2] = 1
    camera_k_after_crop = camera_k.copy()
    camera_k_after_crop[0, 2] = camera_k_after_crop[0, 2] - start_x
    camera_k_after_crop[1, 2] = camera_k_after_crop[1, 2] - start_y

    # 自适应全局深度显示
    min_depth_show=999
    max_depth_show=-1
    for tmp_depth in depth_res:
        min_depth_show=min(min_depth_show,tmp_depth.min())
        max_depth_show=max(max_depth_show,tmp_depth.max())
    print('min_depth_show', min_depth_show, 'max_depth_show', max_depth_show)

    print('min_depth_show', min_depth_show, 'max_depth_show', max_depth_show)

    writer = imageio.get_writer(videoout, fps=10, format='FFMPEG', quality=10, codec='libx264')

    camera_move_x = []
    while len(camera_move_x) < len(depth_imgs):
        tmp_limit = np.random.randint(20, 50)
        camera_move_x.extend([x / tmp_limit for x in range(-tmp_limit, tmp_limit, 1)])
        camera_move_x.extend([x / tmp_limit for x in range(tmp_limit, -tmp_limit, -1)])
    camera_move_x = camera_move_x[:len(depth_imgs)]

    camera_move_y = []
    while len(camera_move_y) < len(depth_imgs):
        tmp_limit = np.random.randint(20, 50)
        camera_move_y.extend([x / tmp_limit / 2 for x in range(-tmp_limit, tmp_limit, 1)])
        camera_move_y.extend([x / tmp_limit / 2 for x in range(tmp_limit, -tmp_limit, -1)])
    camera_move_y = camera_move_y[:len(depth_imgs)]

    for depth_img, depth, diff_x, diff_y in tqdm(zip(depth_imgs, depth_res, camera_move_x, camera_move_y)):
        img, pc_input = depth_img[1], depth_img[0]
        img = undistort_img(img, camera_k, camera_d)
        img = img[start_y:end_y, start_x:end_x, :]

        # 3d点云渲染图
        points, color = unproject_depth(depth, camera_k_after_crop, color=img)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(color)
        camera_position = [0 + diff_x, 0 + diff_y, -1]  # 相机位置
        target = [0, 0, 1]  # 观察目标点
        up_vector = [0, -1, 0]  # 相机的上方向
        renderer.scene.camera.look_at(target, camera_position, up_vector)
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pointcloud", pcd, mat)
        image_pc = renderer.render_to_image()
        image_pc = np.asarray(image_pc)
        image_pc = cv2.cvtColor(image_pc, cv2.COLOR_RGBA2RGB)

        # 深度图
        depth_show = visualize_depth(depth, depth_min=min_depth_show, depth_max=max_depth_show)


        pc_img_origin = ego2img(pc_input, ego2sensor_rt, camera_k_after_crop)
        pc_x_origin, pc_y_origin, pc_z_origin = pc_img_origin[:, 0], pc_img_origin[:, 1], pc_img_origin[:, 2]
        pc_x_origin, pc_y_origin = pc_x_origin.astype(np.int32), pc_y_origin.astype(np.int32)
        pc_z_origin_min = pc_z_origin.min()
        pc_z_origin_max = pc_z_origin.max()
        pc_depth = (pc_z_origin - pc_z_origin_min) / (pc_z_origin_max - pc_z_origin_min) * 255
        pc_depth = pc_depth.astype(int)
        radius = 2  # 半径，表示点的大小
        thickness = -1
        for y, x, d in zip(pc_y_origin, pc_x_origin, pc_depth):
            cv2.circle(img, (int(x * img_scale_factor), int(y * img_scale_factor)), radius,
                       (0, int(d), 0), thickness)

        image_pc = image_pc[crop_image_pc_height:-crop_image_pc_height, :, :]
        video_img = cv2.vconcat([img, depth_show, image_pc])
        writer.append_data(video_img)

    writer.close()
