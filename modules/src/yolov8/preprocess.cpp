/************************************************************************
Copyright 2025 RoboSense Technology Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
************************************************************************/

#include "perception/common/input_msg.h"
#include "perception/yolov8/preprocess.h"

namespace robosense {
namespace perception {

int get_image_size(image_buffer_t* image) {
    if (image == NULL) {
        return 0;
    }
    switch (image->format)
    {
    case image_format_t::IMAGE_FORMAT_GRAY8:
        return image->width * image->height;
    case image_format_t::IMAGE_FORMAT_RGB888:
        return image->width * image->height * 3;
    case image_format_t::IMAGE_FORMAT_RGBA8888:
        return image->width * image->height * 4;
    case image_format_t::IMAGE_FORMAT_YUV420SP_NV12:
    case image_format_t::IMAGE_FORMAT_YUV420SP_NV21:
        return image->width * image->height * 3 / 2;
    default:
        return 0;
        break;
    }
}


static int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {
    if (dst == NULL) {
        printf("dst buffer is null\n");
        return -1;
    }

    float x_ratio = (float)crop_width / (float)dst_box_width;
    float y_ratio = (float)crop_height / (float)dst_box_height;

    // printf("src_width=%d src_height=%d crop_x=%d crop_y=%d crop_width=%d crop_height=%d\n",
    //     src_width, src_height, crop_x, crop_y, crop_width, crop_height);
    // printf("dst_width=%d dst_height=%d dst_box_x=%d dst_box_y=%d dst_box_width=%d dst_box_height=%d\n",
    //     dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    // printf("channel=%d x_ratio=%f y_ratio=%f\n", channel, x_ratio, y_ratio);

    // 从原图指定区域取数据，双线性缩放到目标指定区域
    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            int src_x = (int)(dst_x_offset * x_ratio) + crop_x;
            int src_y = (int)(dst_y_offset * y_ratio) + crop_y;

            float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
            float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

            int index1 = src_y * src_width * channel + src_x * channel;
            int index2 = index1 + src_width * channel;    // down
            if (src_y == src_height - 1) {
                // 如果到图像最下边缘，变成选择上面的像素
                index2 = index1 - src_width * channel;
            }
            int index3 = index1 + 1 * channel;            // right
            int index4 = index2 + 1 * channel;            // down right
            if (src_x == src_width - 1) {
                // 如果到图像最右边缘，变成选择左边的像素
                index3 = index1 - 1 * channel;
                index4 = index2 - 1 * channel;
            }

            // printf("dst_x=%d dst_y=%d dst_x_offset=%d dst_y_offset=%d src_x=%d src_y=%d x_diff=%f y_diff=%f src index=%d %d %d %d\n",
            //     dst_x, dst_y, dst_x_offset, dst_y_offset,
            //     src_x, src_y, x_diff, y_diff,
            //     index1, index2, index3, index4);

            for (int c = 0; c < channel; c++) {
                unsigned char A = src[index1+c];
                unsigned char B = src[index3+c];
                unsigned char C = src[index2+c];
                unsigned char D = src[index4+c];

                unsigned char pixel = (unsigned char)(
                    A * (1 - x_diff) * (1 - y_diff) +
                    B * x_diff * (1 - y_diff) +
                    C * y_diff * (1 - x_diff) +
                    D * x_diff * y_diff
                );

                dst[(dst_y * dst_width  + dst_x) * channel + c] = pixel;
            }
        }
    }

    return 0;
}

static int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {

    unsigned char* src_y = src;
    unsigned char* src_uv = src + src_width * src_height;

    unsigned char* dst_y = dst;
    unsigned char* dst_uv = dst + dst_width * dst_height;

    crop_and_scale_image_c(1, src_y, src_width, src_height, crop_x, crop_y, crop_width, crop_height,
        dst_y, dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    
    crop_and_scale_image_c(2, src_uv, src_width / 2, src_height / 2, crop_x / 2, crop_y / 2, crop_width / 2, crop_height / 2,
        dst_uv, dst_width / 2, dst_height / 2, dst_box_x, dst_box_y, dst_box_width, dst_box_height);

    return 0;
}

static int convert_image_cpu(image_buffer_t *src, image_buffer_t *dst, Bbox *src_box, Bbox *dst_box, char color) {
    // int ret;
    if (dst->virt_addr == NULL) {
        return -1;
    }
    if (src->virt_addr == NULL) {
        return -1;
    }
    if (src->format != dst->format) {
        return -1;
    }

    int src_box_x = 0;
    int src_box_y = 0;
    int src_box_w = src->width;
    int src_box_h = src->height;
    if (src_box != NULL) {
        src_box_x = src_box->left;
        src_box_y = src_box->top;
        src_box_w = src_box->right - src_box->left + 1;
        src_box_h = src_box->bottom - src_box->top + 1;
    }
    int dst_box_x = 0;
    int dst_box_y = 0;
    int dst_box_w = dst->width;
    int dst_box_h = dst->height;
    if (dst_box != NULL) {
        dst_box_x = dst_box->left;
        dst_box_y = dst_box->top;
        dst_box_w = dst_box->right - dst_box->left + 1;
        dst_box_h = dst_box->bottom - dst_box->top + 1;
    }

    // fill pad color
    if (dst_box_w != dst->width || dst_box_h != dst->height) {
        int dst_size = get_image_size(dst);
        memset(dst->virt_addr, color, dst_size);
    }

    // int need_release_dst_buffer = 0;
    int reti = 0;
    if (src->format == image_format_t::IMAGE_FORMAT_RGB888) {
        reti = crop_and_scale_image_c(3, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == image_format_t::IMAGE_FORMAT_RGBA8888) {
        reti = crop_and_scale_image_c(4, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == image_format_t::IMAGE_FORMAT_GRAY8) {
        reti = crop_and_scale_image_c(1, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == image_format_t::IMAGE_FORMAT_YUV420SP_NV12 || src->format == image_format_t::IMAGE_FORMAT_YUV420SP_NV21) {
        reti = crop_and_scale_image_yuv420sp(src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else {
        printf("no support format %d\n", src->format);
    }
    if (reti != 0) {
        printf("convert_image_cpu fail %d\n", reti);
        return -1;
    }
    // printf("finish\n");
    return 0;
}

int convert_image(image_buffer_t* src_img, image_buffer_t* dst_img, Bbox* src_box, Bbox* dst_box, char color)
{
    int ret;
    ret = convert_image_cpu(src_img, dst_img, src_box, dst_box, color);
    return ret;
}

int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, LetterBoxInfo* letterbox, char color)
{
    int ret = 0;
    int allow_slight_change = 1;
    int src_w = src_image->width;
    int src_h = src_image->height;
    int dst_w = dst_image->width;
    int dst_h = dst_image->height;
    int resize_w = dst_w;
    int resize_h = dst_h;

    int padding_w = 0;
    int padding_h = 0;

    int _left_offset = 0;
    int _top_offset = 0;
    float scale = 1.0;

    Bbox src_box;
    src_box.left = 0;
    src_box.top = 0;
    src_box.right = src_image->width - 1;
    src_box.bottom = src_image->height - 1;

    Bbox dst_box;
    dst_box.left = 0;
    dst_box.top = 0;
    dst_box.right = dst_image->width - 1;
    dst_box.bottom = dst_image->height - 1;

    float _scale_w = (float)dst_w / src_w;
    float _scale_h = (float)dst_h / src_h;
    if(_scale_w < _scale_h) {
        scale = _scale_w;
        resize_h = (int) src_h*scale;
    } else {
        scale = _scale_h;
        resize_w = (int) src_w*scale;
    }
    // slight change image size for align
    if (allow_slight_change == 1 && (resize_w % 4 != 0)) {
        resize_w -= resize_w % 4;
    }
    if (allow_slight_change == 1 && (resize_h % 2 != 0)) {
        resize_h -= resize_h % 2;
    }
    // padding
    padding_h = dst_h - resize_h;
    padding_w = dst_w - resize_w;
    // center
    if (_scale_w < _scale_h) {
        dst_box.top = padding_h / 2;
        if (dst_box.top % 2 != 0) {
            dst_box.top -= dst_box.top % 2;
            if (dst_box.top < 0) {
                dst_box.top = 0;
            }
        }
        dst_box.bottom = dst_box.top + resize_h - 1;
        _top_offset = dst_box.top;
    } else {
        dst_box.left = padding_w / 2;
        if (dst_box.left % 2 != 0) {
            dst_box.left -= dst_box.left % 2;
            if (dst_box.left < 0) {
                dst_box.left = 0;
            }
        }
        dst_box.right = dst_box.left + resize_w - 1;
        _left_offset = dst_box.left;
    }
    // printf("scale=%f dst_box=(%d %d %d %d) allow_slight_change=%d _left_offset=%d _top_offset=%d padding_w=%d padding_h=%d\n",
    //     scale, dst_box.left, dst_box.top, dst_box.right, dst_box.bottom, allow_slight_change,
    //     _left_offset, _top_offset, padding_w, padding_h);

    //set offset and scale
    if(letterbox != NULL){
        letterbox->scale = scale;
        letterbox->x_pad = _left_offset;
        letterbox->y_pad = _top_offset;
    }
    // alloc memory buffer for dst image,
    // remember to free
    if (dst_image->virt_addr == NULL && dst_image->fd <= 0) {
        int dst_size = get_image_size(dst_image);
        dst_image->virt_addr = (uint8_t *)malloc(dst_size);
        if (dst_image->virt_addr == NULL) {
            printf("malloc size %d error\n", dst_size);
            return -1;
        }
    }
    ret = convert_image(src_image, dst_image, &src_box, &dst_box, color);
    return ret;
}

} // namespace robosense
} // namespace perception