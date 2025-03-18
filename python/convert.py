import sys
import subprocess
import os
import argparse
DATASET_PATH = 'python/COCO/coco_subset_20.txt'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Convert Script")
    parser.add_argument("--model_path", type=str, required=True, help="onnx model path to the input model file")
    parser.add_argument("--output_path", type=str, required=True, help="path to save the output file")
    parser.add_argument("--do_quant", action="store_true", help="Enable model quantization")
    parser.add_argument("--model_type", type=str, required=True, choices=["trt", "rknn"], help="Type of the model")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode")
    args, unknown = parser.parse_known_args()
    if args.model_type == "rknn":
        parser.add_argument("--platform", type=str, required=True, help="Target platform for RKNN model")
    if args.do_quant:
        parser.add_argument("--dataset_path", type=str, required=True, help="dataset for quant", default=DATASET_PATH)
    args = parser.parse_args()
    return args

def convert_rknn(args):
    from rknn.api import RKNN
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=args.platform)
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.platform)
    # rknn.config(mean_values=[[0.485*255, 0.456*255, 0.406*255]], std_values=[[0.229*255, 0.224*255, 0.225*255]], target_platform=args.platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    if args.do_quant:
        print('--> Compiling model with quantization')
        ret = rknn.build(do_quantization=args.do_quant, dataset=args.dataset_path)
    else:
        print('--> Compiling model without quantization')
        ret = rknn.build(do_quantization=args.do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()

def check_trtexec_exists():
    import shutil
    # 检查 trtexec 是否在系统路径中
    trtexec_path = shutil.which("trtexec")
    if trtexec_path:
        print(f"trtexec 命令已找到，路径为: {trtexec_path}")
        return True
    else:
        print("""未找到 trtexec 命令，请确保 TensorRT 已正确安装，并将 trtexec 添加到 PATH 环境变量中
        export PATH=/path/to/TensorRT/bin:$PATH
        export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH""")
    return False

def convert_trt(args):
    if args.fp16:
        cmd = ["trtexec", f"--onnx={args.model_path}", f"--saveEngine={args.output_path}", "--fp16"]
    else:
        cmd = ["trtexec", f"--onnx={args.model_path}", f"--saveEngine={args.output_path}"]
    env = os.environ.copy()
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line.strip())
        process.wait()
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
        else:
            print("Command executed successfully.")

if __name__ == '__main__':
    args = parse_arguments()
    print("Parsed Arguments:")
    print(f"Model Path: {args.model_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Do Quantization: {args.do_quant}")
    print(f"Model Type: {args.model_type}")
    if args.model_type == "rknn":
        print(f"Platform: {args.platform}")
        convert_rknn(args)
    elif args.model_type == "trt":
        if (check_trtexec_exists()):
            convert_trt(args)
    else:
        print("ERROR: Invalid output model type: {}".format(args.model_type))
        exit(1)