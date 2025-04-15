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
#include "perception/yolov8/postprocess.h"

namespace robosense {
namespace perception {

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(std::vector<float>& tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        // float exp_t[dfl_len];
        std::vector<float> exp_t(dfl_len);
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }

        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c = 0; c < OBJ_CLASS_NUM_IN; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }
            // compute box
            if (max_score > score_thres_i8){
                offset = i * grid_w + j;
                float box[4];
                // float before_dfl[dfl_len*4];
                std::vector<float> before_dfl(dfl_len*4);
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM_IN; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                // float before_dfl[dfl_len*4];
                std::vector<float> before_dfl(dfl_len*4);
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);
                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

int post_process(Yolov8DetParams::Ptr params_ptr, std::vector<void*> outputs, LetterBoxInfo *letter_box, ObjectList *od_results)
{
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = params_ptr->config_ptr_->nn_input_width;
    int model_in_h = params_ptr->config_ptr_->nn_input_height;
    float nms_threshold = params_ptr->config_ptr_->nms_threshold;
    float conf_threshold = params_ptr->config_ptr_->box_conf_threshold;
    int dfl_len = params_ptr->model_attr_ptr_->output_attrs[0].dims[1] / 4U;
    int output_per_branch = params_ptr->model_attr_ptr_->n_output / 3;
    for (int i = 0; i < 3; i++)
    {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = outputs[i * output_per_branch + 2];
            score_sum_zp = params_ptr->model_attr_ptr_->output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = params_ptr->model_attr_ptr_->output_attrs[i * output_per_branch + 2].scale;
        }
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;


        grid_h = params_ptr->model_attr_ptr_->output_attrs[box_idx].dims[2];
        grid_w = params_ptr->model_attr_ptr_->output_attrs[box_idx].dims[3];

        stride = model_in_h / grid_h;
        if (params_ptr->model_attr_ptr_->is_quant)
        {
            validCount += process_i8((int8_t *)outputs[box_idx], params_ptr->model_attr_ptr_->output_attrs[box_idx].zp, params_ptr->model_attr_ptr_->output_attrs[box_idx].scale,
                                     (int8_t *)outputs[score_idx], params_ptr->model_attr_ptr_->output_attrs[score_idx].zp, params_ptr->model_attr_ptr_->output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            validCount += process_fp32((float *)outputs[box_idx], (float *)outputs[score_idx], (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len,
                                       filterBoxes, objProbs, classId, conf_threshold);
        }
    }
    // printf("validCount=%d\n", validCount);
    // no object detect
    if (validCount <= 0)
    {
        od_results->count = 0;
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE_IN)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

void post_process_scale_1(Yolov8DetParams::Ptr params_ptr, std::vector<std::vector<cv::Rect2d>>& bboxes, std::vector<std::vector<float>>& scores) {
  float CONF_THRES_RAW_ = -log(1 / params_ptr->config_ptr_->box_conf_threshold - 1);     // 利用反函数作用阈值，利用单调性筛选
  // 7.1.3 将BPU推理完的内存地址转换为对应类型的指针
  auto *s_bbox_raw = reinterpret_cast<int32_t *>(params_ptr->msg_ptr_->nn_outputs[0]);
  auto s_bbox_scale = reinterpret_cast<float*>(params_ptr->model_attr_ptr_->output_attrs[0].scale_pointer);
  auto *s_cls_raw = reinterpret_cast<float *>(params_ptr->msg_ptr_->nn_outputs[3]);
  int H_8 = 640 / 8;
  int W_8 = 640 / 8;
  for (int h = 0; h < H_8; h++) {
    for (int w = 0; w < W_8; w++) {
      // 7.1.4 取对应H和W位置的C通道, 记为数组的形式
      // cls对应OBJ_CLASS_NUM_IN个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
      // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
      float *cur_s_cls_raw = s_cls_raw;
      int32_t *cur_s_bbox_raw = s_bbox_raw;
      s_cls_raw += OBJ_CLASS_NUM_IN;
      s_bbox_raw += REG * 4;

      // 7.1.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
      int cls_id = 0;
      for (int i = 1; i < OBJ_CLASS_NUM_IN; i++) {
        if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id]) {
          cls_id = i;
        }
      }

      // 7.1.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
      if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW_) {
          continue;
      }

      // 7.1.7 计算这个目标的分数
      float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

      // 7.1.8 对bbox_raw信息进行反量化, DFL计算
      float ltrb[4], sum, dfl;
      for (int i = 0; i < 4; i++) {
        ltrb[i] = 0.;
        sum = 0.;
        for (int j = 0; j < REG; j++) {
          dfl = std::exp(float(cur_s_bbox_raw[REG * i + j]) * s_bbox_scale[j]);
          ltrb[i] += dfl * j;
          sum += dfl;
        }
        ltrb[i] /= sum;
      }

      // 7.1.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
      if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
          continue;
      }

      // 7.1.10 dist 2 bbox (ltrb 2 xyxy)
      float x1 = (w + 0.5 - ltrb[0]) * 8.0;
      float y1 = (h + 0.5 - ltrb[1]) * 8.0;
      float x2 = (w + 0.5 + ltrb[2]) * 8.0;
      float y2 = (h + 0.5 + ltrb[3]) * 8.0;
      // 7.1.11 对应类别加入到对应的std::vector中
      bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
      scores[cls_id].push_back(score);
      }
    }
}

void post_process_scale_2(Yolov8DetParams::Ptr params_ptr, std::vector<std::vector<cv::Rect2d>>& bboxes, std::vector<std::vector<float>>& scores) {
  float CONF_THRES_RAW_ = -log(1 / params_ptr->config_ptr_->box_conf_threshold - 1);     // 利用反函数作用阈值，利用单调性筛选
  // 7.2.3 将BPU推理完的内存地址转换为对应类型的指针
  auto *m_bbox_raw = reinterpret_cast<int32_t *>(params_ptr->msg_ptr_->nn_outputs[1]);
  auto m_bbox_scale = reinterpret_cast<float*>(params_ptr->model_attr_ptr_->output_attrs[1].scale_pointer);
  auto *m_cls_raw = reinterpret_cast<float *>(params_ptr->msg_ptr_->nn_outputs[4]);
  int H_16 = 640 / 16;
  int W_16 = 640 / 16;
  for (int h = 0; h < H_16; h++) {
    for (int w = 0; w < W_16; w++) {
      // 7.2.4 取对应H和W位置的C通道, 记为数组的形式
      // cls对应OBJ_CLASS_NUM_IN个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
      // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
      float *cur_m_cls_raw = m_cls_raw;
      int32_t *cur_m_bbox_raw = m_bbox_raw;
      m_cls_raw += OBJ_CLASS_NUM_IN;
      m_bbox_raw += REG * 4;

      // 7.2.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
      int cls_id = 0;
      for (int i = 1; i < OBJ_CLASS_NUM_IN; i++) {
        if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id]) {
          cls_id = i;
        }
      }
      // 7.2.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
      if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW_)
        continue;

      // 7.2.7 计算这个目标的分数
      float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

      // 7.2.8 对bbox_raw信息进行反量化, DFL计算
      float ltrb[4], sum, dfl;
      for (int i = 0; i < 4; i++) {
        ltrb[i] = 0.;
        sum = 0.;
        for (int j = 0; j < REG; j++)
        {
            dfl = std::exp(float(cur_m_bbox_raw[REG * i + j]) * m_bbox_scale[j]);
            ltrb[i] += dfl * j;
            sum += dfl;
        }
        ltrb[i] /= sum;
      }

      // 7.2.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
      if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
          continue;
      }

      // 7.2.10 dist 2 bbox (ltrb 2 xyxy)
      float x1 = (w + 0.5 - ltrb[0]) * 16.0;
      float y1 = (h + 0.5 - ltrb[1]) * 16.0;
      float x2 = (w + 0.5 + ltrb[2]) * 16.0;
      float y2 = (h + 0.5 + ltrb[3]) * 16.0;
      // 7.2.11 对应类别加入到对应的std::vector中
      bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
      scores[cls_id].push_back(score);
    }
  }
}

void post_process_scale_3(Yolov8DetParams::Ptr params_ptr, std::vector<std::vector<cv::Rect2d>>& bboxes, std::vector<std::vector<float>>& scores) {
  // 7.3.3 将BPU推理完的内存地址转换为对应类型的指针
  float CONF_THRES_RAW_ = -log(1 / params_ptr->config_ptr_->box_conf_threshold - 1);     // 利用反函数作用阈值，利用单调性筛选
  auto *l_bbox_raw = reinterpret_cast<int32_t *>(params_ptr->msg_ptr_->nn_outputs[2]);
  auto l_bbox_scale = reinterpret_cast<float*>(params_ptr->model_attr_ptr_->output_attrs[2].scale_pointer);
  auto *l_cls_raw = reinterpret_cast<float *>(params_ptr->msg_ptr_->nn_outputs[5]);
  int H_32 = 640 / 32;
  int W_32 = 640 / 32;
  for (int h = 0; h < H_32; h++) {
    for (int w = 0; w < W_32; w++) {
      // 7.3.4 取对应H和W位置的C通道, 记为数组的形式
      // cls对应OBJ_CLASS_NUM_IN个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
      // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
      float *cur_l_cls_raw = l_cls_raw;
      int32_t *cur_l_bbox_raw = l_bbox_raw;
      l_cls_raw += OBJ_CLASS_NUM_IN;
      l_bbox_raw += REG * 4;

      // 7.3.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
      int cls_id = 0;
      for (int i = 1; i < OBJ_CLASS_NUM_IN; i++) {
        if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id]) {
          cls_id = i;
        }
      }

      // 7.3.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
      if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW_)
          continue;

      // 7.3.7 计算这个目标的分数
      float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

      // 7.3.8 对bbox_raw信息进行反量化, DFL计算
      float ltrb[4], sum, dfl;
      for (int i = 0; i < 4; i++) {
        ltrb[i] = 0.;
        sum = 0.;
        for (int j = 0; j < REG; j++) {
          dfl = std::exp(float(cur_l_bbox_raw[REG * i + j]) * l_bbox_scale[j]);
          ltrb[i] += dfl * j;
          sum += dfl;
        }
        ltrb[i] /= sum;
      }

      // 7.3.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
      if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
          continue;
      }

      // 7.3.10 dist 2 bbox (ltrb 2 xyxy)
      float x1 = (w + 0.5 - ltrb[0]) * 32.0;
      float y1 = (h + 0.5 - ltrb[1]) * 32.0;
      float x2 = (w + 0.5 + ltrb[2]) * 32.0;
      float y2 = (h + 0.5 + ltrb[3]) * 32.0;
      // 7.3.11 对应类别加入到对应的std::vector中
      bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
      scores[cls_id].push_back(score);
    }
  }
}

void nmsAndDecode(Yolov8DetParams::Ptr params_ptr, const std::vector<std::vector<cv::Rect2d>>& bboxes, const std::vector<std::vector<float>>& scores, const LetterBoxInfo& letter_box) {
  // 7.4 对每一个类别进行NMS
  std::vector<std::vector<int>> indices(OBJ_CLASS_NUM_IN);
  for (int i = 0; i < OBJ_CLASS_NUM_IN; i++) {
    cv::dnn::NMSBoxes(bboxes[i], scores[i], params_ptr->config_ptr_->box_conf_threshold,
      params_ptr->config_ptr_->nms_threshold, indices[i], 1.f, 300);
  }
  //decode
  std::vector<DetObject> det_objs;
  for (int i = 0; i < OBJ_CLASS_NUM_IN; i++) {
    // 8.1 每一个类别分别渲染
    for (auto iter = indices[i].begin(); iter != indices[i].end(); ++iter) {
      // 8.2 获取基本的 bbox 信息
      DetObject det_obj;
      det_obj.box.left = int((bboxes[i][*iter].x - letter_box.x_pad) / letter_box.scale);
      det_obj.box.top = int((bboxes[i][*iter].y - letter_box.y_pad) / letter_box.scale);
      auto width = bboxes[i][*iter].width / letter_box.scale;
      det_obj.box.right = int(det_obj.box.left + width);
      auto height = bboxes[i][*iter].height / letter_box.scale;
      det_obj.box.bottom = int(det_obj.box.top + height);
      det_obj.cls_id = i;
      det_obj.prop = scores[i][*iter];
      det_objs.emplace_back(det_obj);
    }
  }
  // 类间nms
  std::vector<cv::Rect2d> temp_bboxes;
  std::vector<float> temp_scores;
  std::vector<int> index;
  for (size_t i=0; i<det_objs.size(); i++) {
    temp_bboxes.emplace_back(cv::Rect2d(det_objs[i].box.left,
                                        det_objs[i].box.top,
                                        det_objs[i].box.right - det_objs[i].box.left,
                                        det_objs[i].box.bottom - det_objs[i].box.top));
    temp_scores.emplace_back(det_objs[i].prop);
  }
  cv::dnn::NMSBoxes(temp_bboxes, temp_scores, params_ptr->config_ptr_->box_conf_threshold,
                0.7, index, 1.f, 300);
  for (auto idx : index) {
    params_ptr->msg_ptr_->od_results.results.emplace_back(det_objs[idx]);
  }
}

void post_process_hbdnn(Yolov8DetParams::Ptr params_ptr) {
  params_ptr->msg_ptr_->od_results.results.clear();

  std::vector<std::vector<cv::Rect2d>> bboxes(OBJ_CLASS_NUM_IN); // 每个id的xyhw 信息使用一个std::vector<cv::Rect2d>存储
  std::vector<std::vector<float>> scores(OBJ_CLASS_NUM_IN);      // 每个id的score信息使用一个std::vector<float>存储

  post_process_scale_1(params_ptr, bboxes, scores);
  post_process_scale_2(params_ptr, bboxes, scores);
  post_process_scale_3(params_ptr, bboxes, scores);
  nmsAndDecode(params_ptr, bboxes, scores, params_ptr->msg_ptr_->lb);
  params_ptr->msg_ptr_->od_results.count = params_ptr->msg_ptr_->od_results.results.size();
}

} // namespace robosense
} // namespace perception