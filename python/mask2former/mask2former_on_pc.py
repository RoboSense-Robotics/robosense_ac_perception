import numpy as np
import cv2
import pickle
import get_dataset_colormap
import imageio
from tqdm import tqdm
import argparse


def undistort_img(img, K, D):
    if len(D) == 4:

        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D,
                                                           np.eye(3, dtype=np.float32), K,
                                                           (img.shape[1], img.shape[0]), cv2.CV_32FC1)

        img_und = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST)

    else:
        img_und = cv2.undistort(img, K, D)

    return img_und


def parse_arguments():
    parser = argparse.ArgumentParser(description="run mask2former")
    parser.add_argument("--input_info_path", type=str, required=True, help="input info path")
    parser.add_argument("--videoout_path", type=str, required=True, help="video out path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    input_info_path = args.input_info_path
    videoout = args.videoout_path
    with open(input_info_path, 'rb') as f:
        input_info = pickle.load(f)
    input_dic_depth_imgs = input_info['pc_img']
    camera_info = input_info['camera_info']
    camera_k = camera_info['intrinsic']
    camera_d = camera_info['d']
    writer = imageio.get_writer(videoout, fps=10, format='FFMPEG', quality=10, codec='libx264')
    from mmseg.apis import MMSegInferencer

    seg_inferencer = MMSegInferencer(
        model='mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024', device='cuda')

    for input_dic_depth_img in tqdm(input_dic_depth_imgs):
        img_origin, _ = input_dic_depth_img[1], input_dic_depth_img[0]
        img_origin = undistort_img(img_origin, camera_k, camera_d)
        res = seg_inferencer(img_origin, return_datasamples=True)
        res = res.pred_sem_seg.data.cpu().numpy()
        seg_map = np.squeeze(res, axis=0)
        seg_img = get_dataset_colormap.label_to_color_image(
            seg_map, get_dataset_colormap.get_cityscapes_name()).astype(np.uint8)
        overlay = img_origin * 0.5 + seg_img * 0.5
        overlay = overlay.astype(np.uint8)
        overlay = cv2.hconcat([img_origin, seg_img])
        writer.append_data(overlay)

    writer.close()
