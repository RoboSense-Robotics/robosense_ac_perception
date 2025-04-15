import numpy as np
import cv2
import os
from horizon_tc_ui.data.transformer import (BGR2NV12Transformer,
                                            HWC2CHWTransformer,
                                            NV12ToYUV444Transformer,BGR2RGBTransformer,
                                            ResizeTransformer, YUVTransformer)

model_input_height = 1024
model_input_width = 1792
def calibration_transformers():
    """
    step：
        1、resize to 224 * 960
        2、HWC to CHW
        3、BGR to RGB
    """
    transformers = [
        ResizeTransformer((model_input_height, model_input_width)),
        HWC2CHWTransformer(),
        BGR2RGBTransformer(data_format="CHW"),
    ]
    return transformers


def infer_transformers(input_layout="NHWC"):
    """
    step：
        1、resize to 224 * 960
        2、bgr to nv12
        3、nv12 to yuv444
    """
    transformers = [
        ResizeTransformer((model_input_height, model_input_width)),
        BGR2NV12Transformer(data_format="HWC", cvt_mode="opencv"),
        NV12ToYUV444Transformer(target_size=(model_input_height,
                                             model_input_width),
                                yuv444_output_layout=input_layout[1:]),
    ]
    return transformers


def main(images_dir):
    transformers = calibration_transformers()
    images_list = os.listdir(images_dir)
    out_dir = "calibration_data"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    for image_name in images_list:
        image_file = os.path.join(images_dir, image_name)
        image = cv2.imread(image_file)
        image = [image]
        for trans in transformers:
            image = trans(image)
        image[0].astype(np.float32).tofile(os.path.join(out_dir, f"{image_name}.rgb"))

if __name__ == '__main__':
    main("data")
