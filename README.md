# diffusion-models

A containerized implementation of state-of-the-art diffusion models for image-to-video generation, supporting multiple model architectures with AMD GPU acceleration.

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)
- [Hardware Requirements](#hardware-requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository provides automated setup and execution scripts for running diffusion models in Docker containers with AMD ROCm support. Each model has its own directory with dedicated documentation and automation scripts.

## 🤖 Supported Models

### ✅ Currently Available
- **[Wan2.2 I2V](./wan_22/)** - Image-to-Video generation with 14B parameters
  - Resolution: Up to 1280x720
  - Frame generation: 81 frames
  - Multi-GPU support via Ulysses parallelization

### 🚧 Coming Soon
- **Flux 2** - Advanced diffusion model for image generation

## 🔧 Prerequisites

### System Requirements
- Linux-based operating system (Ubuntu 20.04+ recommended)
- Docker installed and configured
- AMD GPU with ROCm support
- Minimum 8 AMD GPUs (or adjust configurations accordingly)

### Software Dependencies
```bash
# Docker (version 20.10+)
docker --version

# Sufficient disk space
# - Model weights: ~50GB per model
# - Container images: ~20GB
# - Output storage: Variable based on usage
```

### Account Setup
- Hugging Face account with authentication token
- Access permissions to model repositories

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/diffusion-models.git
cd diffusion-models
```

### 2. Set Up Environment
```bash
# Export your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Choose a model directory
cd wan_22
```

### 3. Run Automated Setup
```bash
# Make scripts executable
chmod +x *.sh

# Run the automated pipeline
./run_wan22.sh
```

## 📁 Repository Structure

```
diffusion-models/
├── README.md                 # This file
├── wan_22/                   # Wan2.2 I2V model
│   ├── README.md            # Wan2.2 specific documentation
│   ├── run_wan22.sh         # Main automation script
│   ├── setup_environment.sh # Environment setup
│   └── outputs/             # Generated videos (auto-created)
├── flux_2/                   # Flux 2 model (coming soon)
    └── README.md            # Flux 2 documentation
```

## 📚 Documentation

Each model has its own comprehensive README with:
- Detailed setup instructions
- Configuration options
- Usage examples
- Troubleshooting guides

Navigate to the specific model directory for detailed documentation:
- [Wan2.2 Documentation](./wan_22/README.md)
- [Flux 2 Documentation](./flux_2/README.md) *(coming soon)*

## 💻 Hardware Requirements

### Minimum Configuration
- **GPU**: 8x AMD MI250/MI300 series
- **RAM**: 256GB system memory
- **Storage**: 100GB free space (SSD recommended)
- **Network**: High-speed internet for model downloads

### Recommended Configuration
- **GPU**: 8x AMD MI300X
- **RAM**: 512GB system memory
- **Storage**: 500GB NVMe SSD
- **Network**: 1Gbps+ connection

## 🛠️ Advanced Configuration

### Environment Variables

Global settings applicable to all models:

```bash
# Hugging Face configuration
export HF_HOME="$HOME/hf_models"          # Model cache directory
export HF_TOKEN="your_token"               # Authentication token

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # GPU indices
export OMP_NUM_THREADS=16                     # Thread count

# ROCm/MIOpen settings
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_DEBUG_CONV_DIRECT=0
```

See individual model READMEs for detailed instructions.