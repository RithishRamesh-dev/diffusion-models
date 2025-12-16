#!/bin/bash

set -e

# Setup
export HF_HOME="${HF_HOME:-$HOME/hf_models}"
mkdir -p "$HF_HOME"
mkdir -p ./outputs

CONTAINER_NAME="wan22-container"
IMAGE_NAME="amdsiloai/pytorch-xdit:v25.11.2"

# Default prompt
DEFAULT_PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

PROMPT="${CUSTOM_PROMPT:-$DEFAULT_PROMPT}"

echo "Starting Wan2.2 I2V inference..."
echo "Using prompt: ${PROMPT:0:100}..."

# Remove existing container if present
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Download model
echo "Downloading model..."
docker run --rm \
    --mount type=bind,source="$HF_HOME",target=/hf_home \
    -e HF_HOME=/hf_home \
    $IMAGE_NAME \
    bash -c "pip install -q huggingface-hub && huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Run inference
echo "Running inference..."
docker run --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --user root \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --network host \
    --privileged \
    --name $CONTAINER_NAME \
    --mount type=bind,source="$(pwd)/outputs",target=/outputs \
    --mount type=bind,source="$HF_HOME",target=/hf_home \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e OMP_NUM_THREADS=16 \
    -e MIOPEN_FIND_MODE=3 \
    -e MIOPEN_FIND_ENFORCE=3 \
    -e MIOPEN_DEBUG_CONV_DIRECT=0 \
    -e HF_HOME=/hf_home \
    $IMAGE_NAME \
    bash -c "
        torchrun --nproc_per_node=8 /app/external/xDiT/examples/wan_i2v_example.py \
            --height 720 --width 1280 --num_frames 81 \
            --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
            --ulysses_degree 8 \
            --prompt \"${PROMPT}\" \
            --use_torch_compile \
            --num_inference_steps 40 \
            --img_file_path https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG && \
        cp ./i2v_output.mp4 /outputs/
    "

echo "Done! Output saved to ./outputs/i2v_output.mp4"