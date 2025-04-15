set -e
cd $(dirname $0) || exit

config_file="./ppseg_config_nv12.yaml"
model_type="onnx"
# build model
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}
