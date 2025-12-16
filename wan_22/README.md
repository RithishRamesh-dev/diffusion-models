# Wan2.2 I2V - Image to Video Generation

High-quality image-to-video generation using the Wan2.2-I2V-A14B model with multi-GPU support.

## 📋 Overview

Wan2.2 is a state-of-the-art image-to-video generation model that can create smooth, high-resolution video sequences from a single input image and text prompt.

### Key Features
- **Resolution**: Up to 1280x720 pixels
- **Frame Generation**: 81 frames per video
- **Model Size**: 14 billion parameters
- **Parallelization**: Ulysses degree 8 for multi-GPU inference
- **Optimization**: PyTorch compilation support

## 🚀 Quick Start

### Automated Setup (Recommended)

```bash
# Navigate to wan_22 directory
cd wan_22

# Make scripts executable
chmod +x run_wan22.sh setup_environment.sh

# Set your Hugging Face token (optional, for private models)
export HF_TOKEN="your_token_here"

# Run the complete pipeline
./run_wan22.sh
```

The script will:
1. Set up the environment
2. Pull the Docker image
3. Download the model weights
4. Run inference with example prompt
5. Save output to `./outputs/`

### Custom Prompt

To use your own prompt:

```bash
export CUSTOM_PROMPT="Your detailed description here..."
./run_wan22.sh
```

## 📁 Directory Structure

```
wan_22/
├── README.md                # This file
├── run_wan22.sh            # Main automation script
├── setup_environment.sh    # Environment setup script
└── outputs/                # Generated videos (auto-created)
    └── i2v_output_*.mp4   # Timestamped outputs
```

## 🔧 Manual Setup

If you prefer to run steps manually:

### Step 1: Environment Setup

```bash
# Create necessary directories
export HF_HOME="$HOME/hf_models"
mkdir -p "$HF_HOME"
mkdir -p ./outputs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16
```

### Step 2: Launch Container

```bash
docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --user root \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --network host \
    --privileged \
    --name wan22-inference \
    --mount type=bind,source="$(pwd)/outputs",target=/outputs \
    --mount type=bind,source="$HF_HOME",target=/hf_home \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e OMP_NUM_THREADS=16 \
    -e MIOPEN_FIND_MODE=3 \
    -e MIOPEN_FIND_ENFORCE=3 \
    -e MIOPEN_DEBUG_CONV_DIRECT=0 \
    -e HF_HOME=/hf_home \
    amdsiloai/pytorch-xdit:v25.11.2 \
    bash
```

### Step 3: Inside Container - Download Model

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download model weights
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers

# Verify download
ls -lh /hf_home/hub/
```

### Step 4: Run Inference

```bash
# Set your prompt
export PROMPT="Your detailed video description..."

# Run inference
torchrun --nproc_per_node=8 /app/external/xDiT/examples/wan_i2v_example.py \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --ulysses_degree 8 \
    --prompt "${PROMPT}" \
    --use_torch_compile \
    --num_inference_steps 40 \
    --img_file_path https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG

# Copy output
cp ./i2v_output.mp4 /outputs/

# Exit
exit
```

## ⚙️ Configuration Options

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--height` | 720 | Output video height |
| `--width` | 1280 | Output video width |
| `--num_frames` | 81 | Number of frames to generate |
| `--num_inference_steps` | 40 | Denoising steps (higher = better quality) |
| `--ulysses_degree` | 8 | GPU parallelization degree |
| `--use_torch_compile` | flag | Enable PyTorch compilation |

### Example Custom Configuration

```bash
torchrun --nproc_per_node=8 /app/external/xDiT/examples/wan_i2v_example.py \
    --height 480 \
    --width 854 \
    --num_frames 61 \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --ulysses_degree 8 \
    --prompt "Your prompt" \
    --num_inference_steps 50 \
    --img_file_path /path/to/your/image.jpg
```

## 🖼️ Input Image Guidelines

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

### Recommendations
- **Resolution**: 512x512 to 1920x1080
- **Aspect Ratio**: 16:9 recommended for best results
- **Content**: Clear, well-lit subjects work best
- **File Size**: Under 10MB

### Using Custom Images

```bash
# Local file
--img_file_path /path/to/local/image.jpg

# Remote URL
--img_file_path https://example.com/image.jpg
```

**Next Steps**: Try different prompts and images to explore the model's capabilities!