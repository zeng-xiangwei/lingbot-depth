#!/usr/bin/env python3
"""
TensorRT Inference for LingBot-Depth Model
Single Engine: Encoder + Decoder merged
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
# TENSORRT_AVAILABLE = False
from mdm.model.v2 import MDMModel


def preprocess_image(image_path):
    """Load and preprocess RGB image."""
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    image_np = image_np.astype(np.float32) / 255.0
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


class TensorRTEngine:
    """Single TensorRT engine for full model (Encoder + Decoder merged)."""
    
    def __init__(self, engine_path: str):
        self.engine_path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
    def load(self):
        """Load engine and pre-allocate buffers."""
        print(f"Loading engine: {self.engine_path}")
        with open(self.engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.bindings = []
        self.inputs = {}
        self.outputs = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            mode = self.engine.get_tensor_mode(name)
            print(f"  {name}: shape={shape}, mode={mode}")
            
            tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
            self.bindings.append(tensor)
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = tensor
            else:
                self.outputs[name] = tensor
        
        print("Engine loaded and buffers pre-allocated.")
        return self
    
    def infer(self, image_np: np.ndarray, depth_np: np.ndarray) -> Dict:
        """Run inference using pre-allocated buffers."""
        # Convert HWC to CHW
        image_chw = np.transpose(image_np, (2, 0, 1))[np.newaxis, :, :, :]
        depth_chw = depth_np[np.newaxis, np.newaxis, :, :]
        
        print("Running inference...")
        t0 = time.time()
        
        # Copy data to pre-allocated buffers
        for name, tensor in self.inputs.items():
            if 'image' in name:
                input_tensor = torch.from_numpy(image_chw)
                tensor.copy_(input_tensor)
                print(f"  Input {name}: shape={image_chw.shape}")
                print(f"    Min: {input_tensor.min().item():.6f}, Max: {input_tensor.max().item():.6f}, Mean: {input_tensor.mean().item():.6f}")
            elif 'depth' in name:
                input_tensor = torch.from_numpy(depth_chw)
                tensor.copy_(input_tensor)
                print(f"  Input {name}: shape={depth_chw.shape}")
                print(f"    Min: {input_tensor.min().item():.6f}, Max: {input_tensor.max().item():.6f}, Mean: {input_tensor.mean().item():.6f}")
        
        # Get bindings pointer list
        bindings_ptr = [t.data_ptr() for t in self.bindings]
        t1 = time.time()
        
        # Execute
        self.context.execute_v2(bindings_ptr)
        t2 = time.time()
        
        # Get outputs
        results = {}
        t3 = time.time()
        for name, tensor in self.outputs.items():
            arr = tensor.cpu().numpy()
            # 如果是 FP16，转换为 FP32
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            results[name] = arr
            print(f"  Output {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, dtype={arr.dtype}")
        
        print(f"Inference time: prepare {(t1-t0)*1000:.1f}ms, execute {(t2-t1)*1000:.1f}ms, total {(t3-t0)*1000:.1f}ms")
        
        return results


class TensorRTEncoderEngine:
    """TensorRT engine for encoder-only model."""
    
    def __init__(self, engine_path: str):
        self.engine_path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
    def load(self):
        """Load engine and pre-allocate buffers."""
        print(f"Loading encoder engine: {self.engine_path}")
        with open(self.engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.bindings = []
        self.inputs = {}
        self.outputs = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            mode = self.engine.get_tensor_mode(name)
            print(f"  {name}: shape={shape}, mode={mode}")
            
            tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
            self.bindings.append(tensor)
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = tensor
            else:
                self.outputs[name] = tensor
        
        print("Encoder engine loaded and buffers pre-allocated.")
        return self
    
    def infer(self, image_np: np.ndarray, depth_np: np.ndarray) -> Dict:
        """Run encoder inference using pre-allocated buffers."""
        # Convert HWC to CHW
        image_chw = np.transpose(image_np, (2, 0, 1))[np.newaxis, :, :, :]
        depth_chw = depth_np[np.newaxis, np.newaxis, :, :]
        
        print("Running encoder inference...")
        t0 = time.time()
        
        # Copy data to pre-allocated buffers
        for name, tensor in self.inputs.items():
            if 'image' in name:
                input_tensor = torch.from_numpy(image_chw)
                tensor.copy_(input_tensor)
                print(f"  Input {name}: shape={image_chw.shape}, min={image_chw.min():.4f}, max={image_chw.max():.4f}, mean={image_chw.mean():.4f}, dtype={image_chw.dtype}")
            elif 'depth' in name:
                input_tensor = torch.from_numpy(depth_chw)
                tensor.copy_(input_tensor)
                print(f"  Input {name}: shape={depth_chw.shape}, min={depth_chw.min():.4f}, max={depth_chw.max():.4f}, mean={depth_chw.mean():.4f}, dtype={depth_chw.dtype}")
        
        # Get bindings pointer list
        bindings_ptr = [t.data_ptr() for t in self.bindings]
        t1 = time.time()
        
        # Execute
        self.context.execute_v2(bindings_ptr)
        t2 = time.time()
        
        # Get outputs
        results = {}
        t3 = time.time()
        for name, tensor in self.outputs.items():
            arr = tensor.cpu().numpy()
            # 如果是 FP16，转换为 FP32
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            results[name] = arr
            print(f"  Output {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, dtype={arr.dtype}")
        
        print(f"Inference time: prepare {(t1-t0)*1000:.1f}ms, execute {(t2-t1)*1000:.1f}ms, total {(t3-t0)*1000:.1f}ms")
        
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
        t0 = time.time()
        image, depth = image.to(self.device), depth.to(self.device)
        t1 = time.time()
        print(f"Input data cpu to gpu time: {(t1 - t0)*1000:.4f} ms")
        if image.dtype == torch.float16:
            image, depth = image.half(), depth.half()
        with torch.inference_mode():
            t2 = time.time()
            out = self.model.infer(image, depth_in=depth)
            t3 = time.time()
            print(f"Model inference time: {(t3 - t2)*1000:.4f} ms")
        # Use pinned memory for faster GPU->CPU transfer
        t4 = time.time()
        result = {k: v.cpu().numpy() for k, v in out.items()}
        t5 = time.time()
        print(f"Output data gpu to cpu time: {(t5 - t4)*1000:.4f} ms")
        return result


def main():
    parser = argparse.ArgumentParser(description='TensorRT Inference')
    parser.add_argument('--engine', type=str, help='Engine path (encoder or full model)')
    parser.add_argument('--encoder', action='store_true', help='Use encoder-only model')
    parser.add_argument('--input', type=str, default='examples/0', help='Input directory')
    parser.add_argument('--output', type=str, default='result_trt', help='Output directory')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    image_np = preprocess_image(str(input_dir / 'rgb.png'))
    depth_np = load_depth(str(input_dir / 'raw_depth.png'))
    print(f"Image: {image_np.shape[:2]}, Depth range: {depth_np[depth_np>0].min():.2f}-{depth_np.max():.2f}")
    
    # Run inference based on model type
    if args.encoder:
        # Encoder-only model
        if args.engine and TENSORRT_AVAILABLE:
            engine = TensorRTEncoderEngine(args.engine).load()
            results = engine.infer(image_np, depth_np)
            
            # Save encoder outputs
            x = results.get('x')
            cls_token = results.get('cls_token')
            
            if x is not None:
                np.save(output_dir / 'x_trt.npy', x)
                print(f"\nEncoder x saved: {output_dir / 'x_trt.npy'}")
                print(f"  x shape: {x.shape}")
            
            if cls_token is not None:
                np.save(output_dir / 'cls_token_trt.npy', cls_token)
                print(f"  cls_token saved: {output_dir / 'cls_token_trt.npy'}")
                print(f"  cls_token shape: {cls_token.shape}")
        else:
            print("Please specify --engine for encoder model")
    else:
        # Full model (encoder + decoder)
        if TENSORRT_AVAILABLE:
            engine = TensorRTEngine(args.engine).load()
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
            times = []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = torch.tensor(image_np, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            pytorch = PyTorchModel('/home/zxw/models/lingbot-depth-pretrain-vitl-14/model.pt')
            results = pytorch.infer(image_tensor, torch.tensor(depth_np).unsqueeze(0).cuda())
            for i in range(args.num_runs):
                start = time.time()
                results = pytorch.infer(image_tensor, torch.tensor(depth_np).unsqueeze(0).cuda())
                end = time.time()
                print(f"Run {i+1}/{args.num_runs}: {(end - start)*1000:.1f}ms")
                depth_pred = results['depth']
                times.append(time.time() - start)
            avg_time = sum(times) / len(times)
            print(f"\nPyTorch: {avg_time*1000:.1f}ms avg, FPS: {1/avg_time:.1f}")
        
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
        cv2.imwrite(str(output_dir / 'rgb.png'), cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
