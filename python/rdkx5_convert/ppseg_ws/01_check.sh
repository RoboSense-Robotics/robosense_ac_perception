#!/usr/bin/env sh
# set -e -v
cd $(dirname $0)

model_type="onnx"
onnx_model="/open_explorer/ppseg_ws/model/ppseg_1024_1792.onnx"
march="bayes-e"

hb_mapper checker --model-type ${model_type} \
                  --model ${onnx_model} \
                  --input-shape x 1x3x1024x1792 \
                  --march ${march}
