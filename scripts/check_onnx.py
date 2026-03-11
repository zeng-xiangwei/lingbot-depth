#!/usr/bin/env python3
"""Check ONNX model inputs and outputs."""
import sys
import onnx

def check_onnx(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    
    print(f"\nModel: {path}")
    print("="*60)
    
    print("\nInputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"  {inp.name}: shape={shape}, dtype={dtype}")
    
    print("\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f"  {out.name}: shape={shape}, dtype={dtype}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_onnx.py <onnx_file>")
    else:
        check_onnx(sys.argv[1])
