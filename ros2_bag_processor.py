#!/usr/bin/env python3
"""
ROS2 Bag Processor with LingBot-Depth Model (Streaming Version)

Reads a ROS2 bag, processes depth images using LingBot-Depth model,
and writes the refined depth images to a new bag.

This version uses delay buffer for proper depth-RGB synchronization.

Usage:
    python ros2_bag_processor.py
    python ros2_bag_processor.py --input /path/to/input.bag --output /path/to/output.bag
"""

import os
os.environ['XFORMERS_DISABLED'] = '1'

import cv2
import torch
import numpy as np
import time
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import yaml

# ROS2 dependencies
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.time import Time
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    pass

# Try to import rosbag2
BAG_AVAILABLE = False
try:
    import rosbag2_py
    from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
    BAG_AVAILABLE = True
except ImportError:
    pass

from mdm.model.v2 import MDMModel


# ============== Camera Intrinsics (Hard-coded) ==============
@dataclass
class CameraIntrinsics:
    """Camera intrinsics configuration."""
    width: int = 640
    height: int = 480
    fx: float = 609.379
    fy: float = 609.931
    cx: float = 321.428
    cy: float = 245.055
    scale: float = 0.001  # Depth scale (millimeters to meters)
    
    def get_normalized_matrix(self) -> np.ndarray:
        """Return normalized intrinsics matrix."""
        intrinsics = np.array([
            [self.fx / self.width, 0, self.cx / self.width],
            [0, self.fy / self.height, self.cy / self.height],
            [0, 0, 1]
        ], dtype=np.float32)
        return intrinsics


# ============== Default Configuration ==============
DEFAULT_CONFIG = {
    'input_bag': '/home/zxw/zxw_ws/data/test_chair_change/office_chair_change__long_202510221927',
    'output_bag': '/home/zxw/zxw_ws/data/test_chair_change/office_chair_change__long_202510221927_refined_all',
    'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
    'rgb_topic': '/camera/camera/color/image_raw',
    'refined_topic': '/lingbot/depth_refined',
    'model_name': '/home/zxw/models/lingbot-depth-pretrain-vitl-14/model.pt',
    'skip_frames': 1,
    'use_fp16': True,
}


# ============== Message Serializer/Deserializer ==============

def get_message_type(msg_type: str):
    """Get the message class from type string."""
    try:
        import importlib
        # For "sensor_msgs/msg/Image", we want module="sensor_msgs.msg", class="Image"
        parts = msg_type.replace('/', '.').split('.')
        # parts = ['sensor_msgs', 'msg', 'Image']
        if len(parts) >= 3:
            module_name = '.'.join(parts[:-1])  # "sensor_msgs.msg"
        else:
            module_name = parts[0]  # fallback
        class_name = parts[-1]  # "Image"
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as e:
        print(f"   Error getting message type: {e}")
        return None


def deserialize_message(data: bytes, msg_type: str) -> Optional:
    """Deserialize a ROS2 message from bytes using rclpy.serialization."""
    try:
        from rclpy.serialization import deserialize_message
        from rclpy.time import Time
        msg_class = get_message_type(msg_type)
        if msg_class is None:
            return None
        return deserialize_message(data, msg_class)
    except Exception as e:
        print(f"   Deserialization error: {e}")
    return None


def serialize_message(msg) -> bytes:
    """Serialize a ROS2 message to bytes using rclpy.serialization."""
    try:
        from rclpy.serialization import serialize_message as rcl_serialize
        return rcl_serialize(msg)
    except Exception as e:
        print(f"   Serialization error: {e}")
        return b''


# ============== YAML Metadata Helper ==============

def load_qos_from_yaml(bag_path: str) -> Dict[str, str]:
    """
    Load QoS profiles from the bag's metadata.yaml file.
    
    Args:
        bag_path: Path to the bag directory
        
    Returns:
        Dict mapping topic names to their QoS profile strings
    """
    yaml_path = os.path.join(bag_path, 'metadata.yaml')
    qos_profiles = {}
    
    if not os.path.exists(yaml_path):
        return qos_profiles
    
    try:
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        bag_info = metadata.get('rosbag2_bagfile_information', {})
        topics = bag_info.get('topics_with_message_count', [])
        
        for topic_info in topics:
            topic_meta = topic_info.get('topic_metadata', {})
            topic_name = topic_meta.get('name', '')
            qos = topic_meta.get('offered_qos_profiles', '')
            if topic_name and qos:
                qos_profiles[topic_name] = qos
        
        print(f"   Loaded QoS profiles for {len(qos_profiles)} topics from metadata.yaml")
        
    except Exception as e:
        print(f"   Warning: Failed to load QoS from metadata.yaml: {e}")
    
    return qos_profiles


class LingBotDepthProcessor:
    """Processes RGB-D images using LingBot-Depth model."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_CONFIG['model_name'],
        intrinsics: CameraIntrinsics = None,
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.use_fp16 = use_fp16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.cv_bridge = None
        if ROS2_AVAILABLE:
            from cv_bridge import CvBridge
            self.cv_bridge = CvBridge()
        
    def load_model(self):
        """Load the LingBot-Depth model."""
        print(f"Loading model: {self.model_name}")
        start_time = time.time()
        self.model = MDMModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        load_time = time.time() - start_time
        print(f"   Model loaded in {load_time:.2f}s")
        return self.model
    
    def process(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> np.ndarray:
        """Process RGB-D image pair and return refined depth."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        rgb_tensor = torch.tensor(
            rgb_image / 255.0,
            dtype=torch.float32,
            device=self.device
        ).permute(2, 0, 1).unsqueeze(0)
        
        depth_tensor = torch.tensor(
            depth_image * self.intrinsics.scale,
            dtype=torch.float32,
            device=self.device
        )
        
        # Get intrinsics
        intrinsics = self.intrinsics.get_normalized_matrix()
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                output = self.model.infer(
                    rgb_tensor,
                    depth_in=depth_tensor,
                    apply_mask=True,
                    use_fp16=self.use_fp16,
                    intrinsics=intrinsics_tensor,
                )
        
        refined_depth = output['depth'].squeeze().cpu().numpy()
        # meters to millimeters
        refined_depth = refined_depth / self.intrinsics.scale  
        return refined_depth


# ============== Streaming Bag Processor with Delay Buffer ==============

class StreamingRosBagProcessor:
    """Processes ROS2 bag files with delay buffer for proper depth-RGB synchronization."""
    
    def __init__(
        self,
        config: Dict,
        intrinsics: CameraIntrinsics = None,
    ):
        self.config = config
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.processor = LingBotDepthProcessor(
            model_name=config['model_name'],
            intrinsics=self.intrinsics,
            use_fp16=config['use_fp16'],
        )
        
        # Message buffers - using header.stamp as key
        self.depth_buffer: Dict[int, tuple] = {}  # header_stamp_nanosec -> (rosbag_data, rosbag_timestamp, msg_type)
        self.rgb_buffer: Dict[int, tuple] = {}    # header_stamp_nanosec -> (rosbag_data, rosbag_timestamp, msg_type)
        
        self.msg_type_cache: Dict[str, str] = {}  # topic -> type
        self.qos_profiles: Dict[str, str] = {}   # topic -> QoS profile string
        self.frame_counter = 0  # Global frame counter for skip_frames
        
    def process_streaming(self, input_bag: str, output_bag: str, skip_frames: int = 1, test_seconds: float = 0) -> Dict:
        """
        Process bag with delay buffer - collect messages, then process in order.
        """
        if not BAG_AVAILABLE:
            print("Error: rosbag2_py not available")
            return {'processed': 0}
        
        # Load model first
        self.processor.load_model()
        
        # Load QoS profiles from original bag's metadata.yaml
        self.qos_profiles = load_qos_from_yaml(input_bag)
        
        # Open input bag
        reader = SequentialReader()
        storage_options = StorageOptions(uri=input_bag, storage_id='sqlite3')
        reader.open(storage_options, ConverterOptions())
        
        topic_types = reader.get_all_topics_and_types()
        topic_type_map = {t.name: t.type for t in topic_types}
        
        # Cache message types
        for topic, msg_type in topic_type_map.items():
            self.msg_type_cache[topic] = msg_type
        
        # Open output bag
        writer = SequentialWriter()
        writer.open(
            StorageOptions(uri=output_bag, storage_id='sqlite3'),
            ConverterOptions(),
        )
        
        # Register all topics from input with QoS profiles
        for topic, msg_type in topic_type_map.items():
            qos = self.qos_profiles.get(topic, '')
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=topic,
                    type=msg_type,
                    serialization_format='cdr',
                    offered_qos_profiles=qos,
                )
            )
        
        # Register refined topic with QoS from depth topic
        depth_qos = self.qos_profiles.get(self.config['depth_topic'], '')
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=self.config['refined_topic'],
                type='sensor_msgs/msg/Image',
                serialization_format='cdr',
                offered_qos_profiles=depth_qos,
            )
        )
        
        # Statistics
        stats = {
            'total_messages': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'inference_times': [],
            'depth_buffer_sizes': [],
            'rgb_buffer_sizes': [],
        }
        
        frame_count = 0
        start_time = time.time()
        last_print_time = start_time
        test_end_time = start_time + test_seconds if test_seconds > 0 else None
        
        self.rosbag_start_time: Optional[int] = None
        
        print("\nProcessing in delay-buffer mode...")
        if test_seconds > 0:
            print(f"   [TEST MODE] Will process only {test_seconds} seconds of data")
        
        while reader.has_next():
            # Check test mode time limit
            if test_end_time is not None and time.time() >= test_end_time:
                print(f"\n   [TEST MODE] Time limit reached ({test_seconds}s). Stopping...")
                break
            
            topic, data, timestamp = reader.read_next()
            stats['total_messages'] += 1
            writer.write(topic, data, timestamp)
            
            # Track start time from rosbag
            if self.rosbag_start_time is None:
                self.rosbag_start_time = timestamp
            
            # Process based on topic
            if topic == self.config['depth_topic']:
                frame_count += 1
                stats['depth_buffer_sizes'].append(len(self.depth_buffer))
                
                # Deserialize to get header.stamp
                depth_msg = deserialize_message(data, topic_type_map.get(topic, ''))
                if depth_msg is None:
                    continue
                
                # Get header.stamp
                header_stamp = depth_msg.header.stamp.sec * 1_000_000_000 + depth_msg.header.stamp.nanosec
                
                # Store in depth buffer
                self.depth_buffer[header_stamp] = (data, timestamp, topic_type_map.get(topic, ''))
                
                # Check if we should process from the buffer
                self._process_buffers(writer, skip_frames, stats)
                
            elif topic == self.config['rgb_topic']:
                stats['rgb_buffer_sizes'].append(len(self.rgb_buffer))
                
                # Deserialize to get header.stamp
                rgb_msg = deserialize_message(data, topic_type_map.get(topic, ''))
                if rgb_msg is None:
                    continue
                
                # Get header.stamp
                header_stamp = rgb_msg.header.stamp.sec * 1_000_000_000 + rgb_msg.header.stamp.nanosec
                
                # Store in RGB buffer
                self.rgb_buffer[header_stamp] = (data, timestamp, topic_type_map.get(topic, ''))
                
                # Check if we should process from the buffer
                self._process_buffers(writer, skip_frames, stats)
            
            # Print progress
            current_time = time.time()
            if current_time - last_print_time >= 5.0:
                elapsed = current_time - start_time
                fps = stats['processed'] / elapsed if elapsed > 0 else 0
                print(f"   Processed {stats['processed']} frames, Depth buffer: {len(self.depth_buffer)}, RGB buffer: {len(self.rgb_buffer)} ({fps:.1f} fps)")
                last_print_time = current_time
        
        # Process remaining messages in buffer
        print("\nFlushing remaining buffer...")
        self._flush_buffers(writer, skip_frames, stats)
        
        # Cleanup
        del writer
        del reader
        
        elapsed = time.time() - start_time
        stats['elapsed'] = elapsed
        stats['fps'] = stats['processed'] / elapsed if elapsed > 0 else 0
        
        return stats
    
    def _process_buffers(self, writer, skip_frames: int, stats: Dict):
        """Process messages from buffers in order."""
        if not self.depth_buffer or not self.rgb_buffer:
            return
        
        # Find intersection of timestamps
        depth_stamps = set(self.depth_buffer.keys())
        rgb_stamps = set(self.rgb_buffer.keys())
        common_stamps = depth_stamps & rgb_stamps
        
        if not common_stamps:
            return
        
        # Sort common timestamps
        sorted_stamps = sorted(common_stamps)
        
        # Process oldest timestamps if they're old enough
        for stamp in sorted_stamps:
            # Process this frame
            self._process_frame(stamp, writer, skip_frames, stats)
    
    def _flush_buffers(self, writer, skip_frames: int, stats: Dict):
        """Process all remaining messages in buffers."""
        frame_count = 0
        
        while True:
            if not self.depth_buffer or not self.rgb_buffer:
                break
            
            # Find intersection
            depth_stamps = set(self.depth_buffer.keys())
            rgb_stamps = set(self.rgb_buffer.keys())
            common_stamps = depth_stamps & rgb_stamps
            
            if not common_stamps:
                break
            
            # Process all remaining timestamps in order
            for stamp in sorted(common_stamps):
                self._process_frame(stamp, writer, skip_frames, stats)
                frame_count += 1
            
            break
        
        print(f"   Flushed {frame_count} frames from buffers")
    
    def _process_frame(self, stamp: int, writer, skip_frames: int, stats: Dict):
        """Process a single frame with matching depth and RGB timestamps."""
        try:
            # Get depth and RGB data
            depth_data, depth_rosbag_ts, depth_type = self.depth_buffer[stamp]
            rgb_data, rgb_rosbag_ts, rgb_type = self.rgb_buffer[stamp]
            
            # Deserialize messages
            depth_msg = deserialize_message(depth_data, depth_type)
            rgb_msg = deserialize_message(rgb_data, rgb_type)
            
            if depth_msg is None or rgb_msg is None:
                stats['failed'] += 1
                self.depth_buffer.pop(stamp, None)
                self.rgb_buffer.pop(stamp, None)
                return
            
            frame_id = depth_msg.header.frame_id
            
            # Skip frames based on counter
            self.frame_counter += 1
            if self.frame_counter % skip_frames != 0:
                self.depth_buffer.pop(stamp, None)
                self.rgb_buffer.pop(stamp, None)
                return
            
            # Convert to numpy arrays
            if self.processor.cv_bridge is not None:
                rgb_image = self.processor.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
                depth_image = self.processor.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            else:
                rgb_image = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
                depth_image = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
            
            # Process with timing
            infer_start = time.perf_counter()
            refined_depth = self.processor.process(rgb_image, depth_image)
            infer_time = time.perf_counter() - infer_start
            stats['inference_times'].append(infer_time)
            
            # Create output message using header.stamp
            output_msg = self._create_depth_message(refined_depth, stamp, frame_id)
            
            # Serialize and write refined depth
            output_data = serialize_message(output_msg)
            rosbag_ts = max(depth_rosbag_ts, rgb_rosbag_ts)
            writer.write(self.config['refined_topic'], output_data, rosbag_ts)
            
            # Remove from buffers
            self.depth_buffer.pop(stamp, None)
            self.rgb_buffer.pop(stamp, None)
            
            stats['processed'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            print(f"   Failed to process frame at timestamp {stamp}: {e}")
            self.depth_buffer.pop(stamp, None)
            self.rgb_buffer.pop(stamp, None)
    
    def _create_depth_message(self, depth_array: np.ndarray, timestamp: int, frame_id: str):
        """Create a sensor_msgs/Image message from depth array."""
        msg = Image()
        try:
            rclpy.logging.initialize()
            from rclpy.time import Time
            time_obj = Time()
            time_obj.nanoseconds = timestamp
            msg.header.stamp = time_obj.to_msg()
        except Exception:
            msg.header.stamp.sec = int(timestamp // 1_000_000_000)
            msg.header.stamp.nanosec = int(timestamp % 1_000_000_000)
        
        msg.header.frame_id = frame_id
        msg.height = depth_array.shape[0]
        msg.width = depth_array.shape[1]
        msg.encoding = '16UC1'
        msg.is_bigendian = False
        msg.step = int(depth_array.shape[1] * 2)  # 2 bytes per uint16
        msg.data = depth_array.astype(np.uint16).tobytes()
        
        return msg


# ============== Main Function ==============

def main():
    parser = argparse.ArgumentParser(
        description='Process ROS2 bag with LingBot-Depth model (streaming)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ros2_bag_processor.py
  python ros2_bag_processor.py --input /path/to/input.db3 --output /path/to/output.db3
  python ros2_bag_processor.py --test-seconds 3
        """
    )

    parser.add_argument('--input', type=str, default=DEFAULT_CONFIG['input_bag'], help='Input bag path')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_bag'], help='Output bag path')
    parser.add_argument('--depth-topic', type=str, default=DEFAULT_CONFIG['depth_topic'], help='Depth topic')
    parser.add_argument('--rgb-topic', type=str, default=DEFAULT_CONFIG['rgb_topic'], help='RGB topic')
    parser.add_argument('--refined-topic', type=str, default=DEFAULT_CONFIG['refined_topic'], help='Output topic')
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model_name'], help='Model name')
    parser.add_argument('--skip-frames', type=int, default=DEFAULT_CONFIG['skip_frames'], help='Process every Nth frame')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16')
    parser.add_argument('--test-seconds', type=float, default=0, help='Test mode: process only N seconds of data')

    args = parser.parse_args()

    print("=" * 70)
    print("ROS2 Bag Processor with LingBot-Depth (Streaming)".center(70))
    print("=" * 70)

    config = DEFAULT_CONFIG.copy()
    config.update({
        'input_bag': args.input,
        'output_bag': args.output,
        'depth_topic': args.depth_topic,
        'rgb_topic': args.rgb_topic,
        'refined_topic': args.refined_topic,
        'model_name': args.model,
        'skip_frames': args.skip_frames,
        'use_fp16': not args.no_fp16,
        'test_seconds': args.test_seconds,
    })

    print(f"\nInput bag:  {config['input_bag']}")
    print(f"Output bag: {config['output_bag']}")
    print(f"Model:      {config['model_name']}")
    print(f"Skip frames: {config['skip_frames']}")
    if config['test_seconds'] > 0:
        print(f"TEST MODE:  Processing only {config['test_seconds']} seconds of data")
    print(f"FP16:       {config['use_fp16']}")

    intrinsics = CameraIntrinsics()
    processor = StreamingRosBagProcessor(config, intrinsics)

    # Process in streaming mode
    stats = processor.process_streaming(
        config['input_bag'],
        config['output_bag'],
        skip_frames=config['skip_frames'],
        test_seconds=config['test_seconds'],
    )

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Total messages: {stats['total_messages']}")
    print(f"  Processed:     {stats['processed']} frames")
    print(f"  Skipped:       {stats['skipped']} frames")
    print(f"  Failed:        {stats['failed']} frames")
    print(f"  Total time:    {stats.get('elapsed', 0):.1f}s")
    print(f"  FPS:           {stats.get('fps', 0):.2f}")
    
    # Inference time statistics
    if stats['inference_times']:
        infer_times = stats['inference_times']
        avg_infer = sum(infer_times) / len(infer_times)
        min_infer = min(infer_times)
        max_infer = max(infer_times)
        
        print(f"\n  Inference Time Statistics:")
        print(f"    Min:    {min_infer*1000:.1f} ms")
        print(f"    Max:    {max_infer*1000:.1f} ms")
        print(f"    Avg:    {avg_infer*1000:.1f} ms")
        print(f"    FPS:    {1.0/avg_infer:.2f}")
    
    print(f"{'='*70}")
    
    print("Done!")

    return 0


if __name__ == '__main__':
    exit(main())
