CUR_SHELL_DIR=$(cd `dirname $0`; pwd)

example_py_path=$CUR_SHELL_DIR/../example.py
python $example_py_path --model "/home/zxw/models/lingbot-depth-pretrain-vitl-14/model.pt" \
    --example 3 \

# python $example_py_path --model "/home/zxw/models/lingbot-depth-pretrain-vitl-14-v0.5/model.pt" \
#     --example my2 \

# python $example_py_path --model "/home/zxw/models/lingbot-depth-postrain-dc-vitl14/model.pt" \
#     --example my2 \