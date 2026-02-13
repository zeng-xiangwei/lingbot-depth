#!/bin/bash
# =============================================================================
# TensorRT Engine Build Script for LingBot-Depth Model
# 
# This script converts ONNX model to TensorRT engine on Jetson AGX Orin.
# Designed for JetPack 6.2
# =============================================================================

export PATH=$PATH:/usr/src/tensorrt/bin

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TensorRT Engine Builder${NC}"
echo -e "${GREEN}========================================${NC}"

# Default values
ONNX_FILE="model.onnx"
ENGINE_FILE="model.engine"
PRECISION="fp16"
OPT_SHAPES="image:1x3x480x640,depth:1x1x480x640"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --onnx)
            ONNX_FILE="$2"
            shift 2
            ;;
        --engine)
            ENGINE_FILE="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --opt-shapes)
            OPT_SHAPES="$2"
            shift 2
            ;;
        --max-shapes)
            MAX_SHAPES="$2"
            shift 2
            ;;
        --min-shapes)
            MIN_SHAPES="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE_SIZE="$2"
            shift 2
            ;;
        --streams)
            streams="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --onnx FILE       ONNX model file (default: model.onnx)"
            echo "  --engine FILE     Output TensorRT engine file (default: model.engine)"
            echo "  --precision PREC  Precision: fp16, fp32, or int8 (default: fp16)"
            echo "  --opt-shapes SHAPES  Optimization shapes (default: image:1x3x480x640,depth:1x1x480x640)"
            echo "  --max-shapes SHAPES  Maximum shapes for workspace (default: image:4x3x480x640,depth:4x1x480x640)"
            echo "  --min-shapes SHAPES  Minimum shapes for workspace (default: image:1x3x480x640,depth:1x1x480x640)"
            echo "  --workspace SIZE   Workspace size in MB (default: 4096)"
            echo "  --streams NUM     Number of streams (default: 1)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --onnx model.onnx --engine model.engine --precision fp16"
            echo "  $0 --onnx model.onnx --precision fp32"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if ONNX file exists
if [[ ! -f "$ONNX_FILE" ]]; then
    echo -e "${RED}Error: ONNX file not found: $ONNX_FILE${NC}"
    echo "Please export the model first using: python export_onnx.py --model /path/to/model --output model.onnx"
    exit 1
fi

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo -e "${RED}Error: trtexec not found.${NC}"
    echo "Please install TensorRT SDK on your Jetson AGX Orin."
    echo "For JetPack 6.2: sudo apt install tensorrt"
    exit 1
fi

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  ONNX file:      $ONNX_FILE"
echo "  Engine file:   $ENGINE_FILE"
echo "  Precision:     $PRECISION"
echo "  Opt shapes:    $OPT_SHAPES"
echo "  Max shapes:   $MAX_SHAPES"
echo "  Min shapes:   $MIN_SHAPES"
echo "  Workspace:     ${WORKSPACE_SIZE}MB"
echo "  Streams:       $streams"
echo ""

# Build TensorRT engine
echo -e "${GREEN}Building TensorRT engine...${NC}"

# Build command based on precision
if [[ "$PRECISION" == "fp16" ]]; then
    PRECISION_FLAG="--fp16"
elif [[ "$PRECISION" == "fp32" ]]; then
    PRECISION_FLAG=""
elif [[ "$PRECISION" == "int8" ]]; then
    PRECISION_FLAG="--int8"
else
    echo -e "${RED}Error: Unknown precision: $PRECISION${NC}"
    exit 1
fi

# Build the command
# Note: ONNX is already exported with static shapes, no --optShapes needed
CMD="trtexec \
    --onnx=$ONNX_FILE \
    --saveEngine=$ENGINE_FILE \
    $PRECISION_FLAG \
    "

echo "Running: $CMD"
echo ""

# Run the command
START_TIME=$(date +%s)
$CMD
END_TIME=$(date +%s)

# Check if engine was created
if [[ -f "$ENGINE_FILE" ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Engine built successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Engine file: $ENGINE_FILE"
    
    # Get file size
    FILE_SIZE=$(du -h "$ENGINE_FILE" | cut -f1)
    echo "File size:   $FILE_SIZE"
    
    # Get build time
    BUILD_TIME=$((END_TIME - START_TIME))
    echo "Build time:  ${BUILD_TIME}s"
    
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Run inference: python tensorrt_infer.py --engine $ENGINE_FILE --input examples/0/"
    echo "  2. Benchmark:     python tensorrt_infer.py --engine $ENGINE_FILE --input examples/0/ --benchmark"
else
    echo -e "${RED}Error: Engine file not created${NC}"
    exit 1
fi
