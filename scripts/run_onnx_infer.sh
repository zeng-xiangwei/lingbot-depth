CUR_SHELL_DIR=$(cd `dirname $0`; pwd)
onnx_infer_py_path=$CUR_SHELL_DIR/onnx_infer.py
onnx_model_path=$CUR_SHELL_DIR/../result/acc_models/full.onnx
input_path=$CUR_SHELL_DIR/../examples/my/

python $onnx_infer_py_path \
    --model $onnx_model_path \
    --input $input_path


# python onnx_infer.py \
#     --model result/acc_models/encoder.onnx \
#     --input examples/my/ \
#     --encoder


# python onnx_infer.py \
#     --model result/acc_models/preprocess.onnx \
#     --input examples/my/ \
#     --encoder