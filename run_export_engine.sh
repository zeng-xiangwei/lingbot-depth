python export_trt.py --model "/home/zxw/models/lingbot-depth-pretrain-vitl-14/model.pt" \
    --output "result/acc_models" \
    --precision fp16 \

# bash ./build_engine.sh \
#     --onnx result/acc_models/full.onnx \
#     --engine result/acc_models/full.engine \
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
