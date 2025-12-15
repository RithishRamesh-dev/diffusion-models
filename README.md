# diffusion-models

Step 1: Create the Container Interactively
bash# Set up directories first
export HF_HOME="$HOME/hf_models"
mkdir -p "$HF_HOME"
mkdir -p ./wan_22_i2v_outputs

# Launch interactive container
docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --user root \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --network host \
    --privileged \
    --name wan22-benchmark \
    --mount type=bind,source="$(pwd)/wan_22_i2v_outputs",target=/outputs \
    --mount type=bind,source="$HF_HOME",target=/hf_home \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e OMP_NUM_THREADS=16 \
    -e MIOPEN_FIND_MODE=3 \
    -e MIOPEN_FIND_ENFORCE=3 \
    -e MIOPEN_DEBUG_CONV_DIRECT=0 \
    -e HF_HOME=/hf_home \
    amdsiloai/pytorch-xdit:v25.11.2 \
    bash
Step 2: Inside the Container, Download Model
bash# Install huggingface-hub if not present
pip install huggingface-hub

# Download the model
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers

# Verify download
ls -lh /hf_home/
Step 3: Run Inference Inside Container
bash# Set your prompt
export PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

# Run inference
torchrun --nproc_per_node=8 /app/external/xDiT/examples/wan_i2v_example.py \
    --height 720 --width 1280 --num_frames 81 \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --ulysses_degree 8 \
    --prompt "${PROMPT}" \
    --use_torch_compile \
    --num_inference_steps 40 \
    --img_file_path https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG

# Copy output
cp ./i2v_output.mp4 /outputs/

# Exit container
exit