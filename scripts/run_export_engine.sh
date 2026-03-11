CUR_SHELL_DIR=$(cd `dirname $0`; pwd)

export_py_path=$CUR_SHELL_DIR/../export_trt.py 
output_path=$CUR_SHELL_DIR/../result/acc_models 
python $export_py_path --model "/home/zxw/models/lingbot-depth-pretrain-vitl-14/model.pt" \
    --output $output_path \
    --precision fp16 \

build_engine_sh_path=$CUR_SHELL_DIR/build_engine.sh
# bash $build_engine_sh_path \
#     --onnx $output_path/full.onnx \
#     --engine $output_path/full.engine \
#     --precision fp16 \

# Encoder
# bash ./build_engine.sh \
#     --onnx result/acc_models/encoder.onnx \
#     --engine result/acc_models/encoder.engine \
#     --precision fp32 \

# bash ./build_engine.sh \
#     --onnx result/acc_models/preprocess.onnx \
#     --engine result/acc_models/preprocess.engine \
#     --precision fp32 \

# # Decoder
# bash ./build_engine.sh \
#     --onnx result/acc_models/decoder.onnx \
#     --engine result/acc_models/decoder.engine \
