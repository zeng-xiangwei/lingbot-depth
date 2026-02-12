python export_trt.py --model "/home/zxw/models/lingbot-depth-pretrain-vitl-14/model.pt" \
    --output "result/acc_models" \
    --precision fp16 \

# Encoder
# bash ./build_engine.sh \
#     --onnx result/acc_models/encoder.onnx \
#     --engine result/acc_models/encoder.engine \

# # Decoder
# bash ./build_engine.sh \
#     --onnx result/acc_models/decoder.onnx \
#     --engine result/acc_models/decoder.engine \
