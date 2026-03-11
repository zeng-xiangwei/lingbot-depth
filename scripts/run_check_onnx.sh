CUR_SHELL_DIR=$(cd `dirname $0`; pwd)

onnx_path=$CUR_SHELL_DIR/../result/acc_models/full.onnx
python check_onnx.py $onnx_path