#!/usr/bin/env python3
"""
TensorRT Inference for LingBot-Depth Model
Two-stage inference: Encoder -> Decoder
"""
import os
os.environ['XFORMERS_DISABLED'] = '1'

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict
import time
import argparse

# TensorRT imports
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    print(f"TensorRT version: {trt.__version__}")
except ImportError:
    print("Warning: TensorRT not installed.")

from mdm.model.v2 import MDMModel


def preprocess_image(image_path):
    """Load and preprocess RGB image."""
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np


def load_depth(depth_path, scale=1000.0):
    """Load depth map from PNG."""
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
    return np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)


def depth_to_color(depth_map, vmin=None, vmax=None):
    """Convert depth to color visualization."""
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if vmin is None:
        vmin = depth_map[valid].min() if valid.any() else 0
    if vmax is None:
        vmax = depth_map[valid].max() if valid.any() else 1
    normalized = np.clip((depth_map - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = [0, 0, 0]
    return colored


class SplitTensorRTEngine:
    """Two-stage TensorRT inference with pre-allocated buffers."""
    
    def __init__(self, encoder_path: str, decoder_path: str):
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
    def load(self):
        """Load both engines and pre-allocate buffers."""
        print(f"Loading encoder: {self.encoder_path}")
        with open(self.encoder_path, 'rb') as f:
            self.encoder = self.runtime.deserialize_cuda_engine(f.read())
        
        print(f"Loading decoder: {self.decoder_path}")
        with open(self.decoder_path, 'rb') as f:
            self.decoder = self.runtime.deserialize_cuda_engine(f.read())
        
        # Pre-allocate encoder buffers
        print("Pre-allocating encoder buffers...")
        self.encoder_context = self.encoder.create_execution_context()
        self.encoder_bindings = []
        self.encoder_inputs = {}
        self.encoder_outputs = {}
        
        for i in range(self.encoder.num_io_tensors):
            name = self.encoder.get_tensor_name(i)
            shape = tuple(self.encoder.get_tensor_shape(name))
            mode = self.encoder.get_tensor_mode(name)
            print(f"  {name}: shape={shape}, mode={mode}")
            
            tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
            self.encoder_bindings.append(tensor)
            
            if mode == trt.TensorIOMode.INPUT:
                self.encoder_inputs[name] = tensor
            else:
                self.encoder_outputs[name] = tensor
        
        # Pre-allocate decoder buffers
        print("Pre-allocating decoder buffers...")
        self.decoder_context = self.decoder.create_execution_context()
        self.decoder_bindings = []
        self.decoder_inputs = {}
        self.decoder_outputs = {}
        
        for i in range(self.decoder.num_io_tensors):
            name = self.decoder.get_tensor_name(i)
            shape = tuple(self.decoder.get_tensor_shape(name))
            mode = self.decoder.get_tensor_mode(name)
            print(f"  {name}: shape={shape}, mode={mode}")
            
            tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
            self.decoder_bindings.append(tensor)
            
            if mode == trt.TensorIOMode.INPUT:
                self.decoder_inputs[name] = tensor
            else:
                self.decoder_outputs[name] = tensor
        
        print("Engines loaded and buffers pre-allocated.")
        return self
    
    def infer(self, image_np: np.ndarray, depth_np: np.ndarray) -> Dict:
        """Run two-stage inference using pre-allocated buffers."""
        # Convert HWC to CHW
        image_chw = np.transpose(image_np, (2, 0, 1)).astype(np.float32)
        
        # ===== Stage 1: Encoder =====
        print("Running Encoder...")
        t0 = time.time()
        
        # Copy data to pre-allocated buffers
        for name, tensor in self.encoder_inputs.items():
            if 'image' in name:
                tensor.copy_(torch.from_numpy(image_chw))
            elif 'depth' in name:
                tensor.copy_(torch.from_numpy(depth_np[np.newaxis]))
            elif 'base_h' in name:
                tensor.fill_(52)
            elif 'base_w' in name:
                tensor.fill_(69)
        
        # Get bindings pointer list
        bindings_ptr = [t.data_ptr() for t in self.encoder_bindings]
        t1 = time.time()
        
        # Execute
        self.encoder_context.execute_v2(bindings_ptr)
        t2 = time.time()
        print(f"Encoder: prepare {(t1-t0)*1000:.1f}ms, execute {(t2-t1)*1000:.1f}ms")
        
        # Get outputs - copy to CPU
        encoder_out = {}
        t3 = time.time()
        for name, tensor in self.encoder_outputs.items():
            encoder_out[name] = tensor.cpu().numpy()
            print(f"  Output {name}: shape={encoder_out[name].shape}")
        
        # ===== Stage 2: Decoder =====
        print("Running Decoder...")
        t4 = time.time()
        
        # Copy encoder outputs to decoder inputs
        for name, tensor in self.decoder_inputs.items():
            if 'features' in name and 'features' in encoder_out:
                tensor.copy_(torch.from_numpy(encoder_out['features']))
            elif 'cls_token' in name and 'cls_token' in encoder_out:
                tensor.copy_(torch.from_numpy(encoder_out['cls_token']))
        
        # Get bindings pointer list
        bindings_ptr = [t.data_ptr() for t in self.decoder_bindings]
        t5 = time.time()
        
        # Execute
        self.decoder_context.execute_v2(bindings_ptr)
        t6 = time.time()
        print(f"Decoder: prepare {(t5-t4)*1000:.1f}ms, execute {(t6-t5)*1000:.1f}ms")
        
        # Get outputs
        results = {}
        t7 = time.time()
        for name, tensor in self.decoder_outputs.items():
            arr = tensor.cpu().numpy()
            # 如果是 FP16，转换为 FP32
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            results[name] = arr
            print(f"  Output {name}: shape={arr.shape}, dtype={arr.dtype}")
        
        return results


class PyTorchModel:
    """PyTorch fallback."""
    
    def __init__(self, model_path: str, precision: str = 'fp32'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading PyTorch model: {model_path}")
        self.model = MDMModel.from_pretrained(model_path).to(self.device).eval()
        if precision == 'fp16':
            self.model.half()
        print("Model loaded.")
    
    def infer(self, image: torch.Tensor, depth: torch.Tensor) -> Dict:
        """Run inference."""
        image, depth = image.to(self.device), depth.to(self.device)
        if image.dtype == torch.float16:
            image, depth = image.half(), depth.half()
        with torch.inference_mode():
            out = self.model.infer(image, depth_in=depth)
        return {k: v.cpu().numpy() for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description='TensorRT Inference')
    parser.add_argument('--encoder', type=str, required=True, help='Encoder engine path')
    parser.add_argument('--decoder', type=str, required=True, help='Decoder engine path')
    parser.add_argument('--input', type=str, default='examples/0', help='Input directory')
    parser.add_argument('--output', type=str, default='result_trt', help='Output directory')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    image_np = preprocess_image(str(input_dir / 'rgb.png'))
    depth_np = load_depth(str(input_dir / 'raw_depth.png'))
    print(f"Image: {image_np.shape[:2]}, Depth range: {depth_np[depth_np>0].min():.2f}-{depth_np.max():.2f}")
    
    # Run inference
    if TENSORRT_AVAILABLE:
        engine = SplitTensorRTEngine(args.encoder, args.decoder).load()
        times = []
        for i in range(args.num_runs):
            start = time.time()
            results = engine.infer(image_np, depth_np)
            times.append(time.time() - start)
            depth_pred = results['depth_reg']
        avg_time = sum(times) / len(times)
        print(f"\nTensorRT: {avg_time*1000:.1f}ms avg, FPS: {1/avg_time:.1f}")
    else:
        print("Using PyTorch fallback.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_tensor = torch.tensor(image_np / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        pytorch = PyTorchModel('/home/zxw/models/lingbot-depth-pretrain-vitl-14/model.pt')
        results = pytorch.infer(image_tensor, torch.tensor(depth_np).unsqueeze(0).cuda())
        depth_pred = results['depth']
    
    # Save results
    # Decoder 输出已经是 (B, H, W) 格式
    if depth_pred.ndim == 3 and depth_pred.shape[0] == 1:
        depth_pred = depth_pred.squeeze(0)
    elif depth_pred.ndim == 2:
        pass  # 已经是 (H, W)
    else:
        depth_pred = depth_pred.squeeze()
    np.save(output_dir / 'depth_refined.npy', depth_pred)
    cv2.imwrite(str(output_dir / 'depth_refined.png'), depth_to_color(depth_pred))
    cv2.imwrite(str(output_dir / 'rgb.png'), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
