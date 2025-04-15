# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

# execute forward inference for caffe/onnx model to calculate their accuracy

import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from horizon_tc_ui import HB_ONNXRuntime

model_input_height = 1024
model_input_width = 1792
def preprocess_numpy(image_file):
    image = cv2.imread(image_file)
    height, width, channels = image.shape
    img = cv2.resize(image, (model_input_width, model_input_height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, ::-1]
    # img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :]
    return image, img

def plot_image(origin_image, onnx_output):
    def get_pallete():
        pallete = [210, 0, 0, 200, 200, 0, 0, 200, 0, 0, 0, 255]
        return pallete
    print(onnx_output[0].dtype)
    onnx_output = onnx_output[0].astype(np.uint8)
    onnx_output = np.squeeze(onnx_output)

    image_shape = origin_image.shape[:2][::-1]

    onnx_output = np.expand_dims(onnx_output, axis=2)
    onnx_output = cv2.resize(onnx_output,
                             image_shape,
                             interpolation=cv2.INTER_NEAREST)
    out_img = Image.fromarray(onnx_output)
    out_img.putpalette(get_pallete())
    plt.imshow(origin_image)
    plt.imshow(out_img, alpha=0.6)
    fig_name = 'demo.jpg'
    print(f"save predicting image with name {fig_name} ")
    plt.savefig(fig_name)


def postprocess(model_output, origin_image):
    # pred_result = np.argmax(model_output[0], axis=-1)
    # origin_image = np.squeeze(origin_image, axis=0)
    plot_image(origin_image, model_output)

def save_to_bin(image_data):
    # for i in tqdm(range(len(a))):
    #     meta_save_name = meta_base_save_name + "/" + a[i]["file_name"].split(
    #         "/")[-1].split(".")[0] + "_traj_labels_meta.bin"
    save_name = "test_image_data.bin"
    print(image_data.dtype, image_data.shape)
    with open(save_name, "w") as sf:
        image_data.tofile(sf)
    print("Save meta end!")

    # for i in range(1, 11):
    #     print(test_image_data[len(test_image_data) - i])
    # print(test_image_data.dtype)
    # test_image_data.tofile("test_image_data.bin")
def load_from_bin(bin_name):
    with open(bin_name, "rb") as f:
        data = f.read()
        test_image_data = np.frombuffer(data, dtype=np.uint8)
        for i in range(10):
            print(test_image_data[i])
        test_image_data = test_image_data.reshape(-1, 224, 960, 3)
        # test_image_data = test_image_data.reshape(-1, 3, 224, 960)
        print(test_image_data.shape)
    return test_image_data

def inference(sess, image_name, input_layout):
    if input_layout is None:
        logging.warning(f"input_layout not provided. Using {sess.layout[0]}")
        input_layout = sess.layout[0]
    origin_image, image_data = preprocess_numpy(image_name)
    input_name = sess.input_names[0]
    output_names = sess.output_names
    # save_to_bin(image_data)
    # image_data = load_from_bin("0000066.yuv444.bin")
    output = sess.run(output_names, {input_name: image_data})
    postprocess(output, origin_image)
    palette = [[128, 64, 128],[128, 64, 128],[70, 70, 70],[102, 102, 156],[190, 153, 153],[153, 153, 153],[250, 170, 30],[220, 220, 0],
        [107, 142, 35],[128, 64, 128],[70, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],
        [0, 80, 100],[0, 0, 230],[119, 11, 32]]
    # palette = [[210, 0, 0], [200, 200, 0], [0, 200, 0], [0, 0, 255]]
    seg = output[0][0]
    print(seg.shape)
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    save_path = "result_ptq.png"
    cv2.imwrite(save_path, color_seg[...,::-1])
    print(f"Result saved as {save_path}")

def main():
    model_file = "/open_explorer/ppseg_ws/ppseg_1024_1792_nv12/ppseg_1024_1792_nv12_quantized_model.onnx"
    image_file = "/open_explorer/ppseg_ws/data/raw_data/test.png"
    sess = HB_ONNXRuntime(model_file=model_file)
    sess.set_dim_param(0, 0, '?')
    inference(sess, image_name=image_file, input_layout="NCHW")


if __name__ == '__main__':
    main()
