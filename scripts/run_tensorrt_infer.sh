CUR_SHELL_DIR=$(cd `dirname $0`; pwd)
tensorrt_infer_py_path=$CUR_SHELL_DIR/tensorrt_infer.py
engine_path=$CUR_SHELL_DIR/../result/acc_models/full.engine
input_path=$CUR_SHELL_DIR/../examples/my/

python $tensorrt_infer_py_path \
    --engine $engine_path \
    --input $input_path


# python tensorrt_infer.py \
#     --engine result/acc_models/encoder.engine \
#     --input examples/my/ \
#     --encoder


# python tensorrt_infer.py \
#     --engine result/acc_models/preprocess.engine \
#     --input examples/my/ \
#     --encoder