## 转TensorRT模型
### 依赖安装
转TensorRT模型依赖 TensorRT, 依赖的安装请参考

https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/install-guide/index.html#installing-tar

### yolov8n

```shell
cd super_sensor_sdk_ros2/perception
python3 python/convert.py --model_path modules/config/deepmodel/Yolov8/yolov8n.onnx --output_path yolov8n.trt --model_type trt
```
### ppseg

```shell
cd super_sensor_sdk_ros2/perception
python3 python/convert.py --model_path modules/config/deepmodel/PPSeg/pp_liteseg_cityscapes.onnx --output_path ppseg.trt --model_type trt
```
### PromptDA
```shell
cd super_sensor_sdk_ros2/perception
python3 python/convert.py --model_path modules/config/deepmodel/PromptDA/Prompt_vits.onnx --output_path promptda.trt --model_type trt
```

## 转RKNN模型

RKNN的模型转换依赖 RKNN-Toolkit2, 依赖的安装请参考 https://github.com/airockchip/rknn-toolkit2/tree/master/doc 的 Quick Start 文档

### yolov8n

```shell
cd super_sensor_sdk_ros2/perception
python3 python/convert.py --model_path modules/config/deepmodel/Yolov8/yolov8n.onnx --output_path yolov8n.rknn --model_type rknn --platform rk3588 --do_quant --dataset_path python/COCO/coco_subset_20.txt
```

### ppseg

```shell
cd super_sensor_sdk_ros2/perception
python3 python/convert.py --model_path modules/config/deepmodel/PPSeg/pp_liteseg_cityscapes.onnx  --output_path ppseg.rknn --model_type rknn --platform rk3588 --do_quant --dataset_path python/PPSeg/dataset.txt
```

### PromptDA

```shell
cd super_sensor_sdk_ros2/perception
python3 python/convert.py --model_path modules/config/deepmodel/PromptDA/promptda_image.onnx  --output_path promptda_image.rknn --model_type rknn --platform rk3588
python3 python/convert.py --model_path modules/config/deepmodel/PromptDA/promptda_lidar.onnx  --output_path promptda_lidar.rknn --model_type rknn --platform rk3588
```

## 在 nvidia GPU 上离线跑模型
离线跑语义分割或稠密深度估计模型，需要先将ros2文件转为python可以直接读取的pkl文件

```shell
cd super_sensor_sdk_ros2/perception
python3 conver_ros2_to_pkl.py --ros2path ${ros2 path} --pklpath ${pkl path} --ros1path ${ros1 path} --calib ${calib yaml}
```
语义分割 mask2former
```shell
python3 python/mask2former/mask2former_on_pc.py --input_info_path ${pkl path} --videoout_path ${video path}
```
稠密深度估计 PromptDa

下载模型权重 https://huggingface.co/depth-anything/prompt-depth-anything-vitl/resolve/main/model.ckpt
```shell
python3 python/PromptDA/test_on_pc.py --model_res_save_path ${model res path} --input_info_path ${pkl path} --ckpt_path ${model ckpt path}
python3 python/PromptDA/generate_video.py --input_info_path ${pkl path} --videoout_path ${video path} -model_res_save_path ${model res path}
```



