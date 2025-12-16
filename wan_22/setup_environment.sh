#!/bin/bash

################################################################################
# Environment Setup Script for Wan2.2 I2V
#
# This script configures the environment variables and directories needed
# for running the Wan2.2 image-to-video model with AMD GPUs.
#
# Usage:
#   source ./setup_environment.sh
#   OR
#   ./setup_environment.sh (will export variables to calling shell)
################################################################################

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Setting up Wan2.2 I2V environment...${NC}"

# ============================================================================
# Hugging Face Configuration
# ============================================================================

# Set Hugging Face cache directory
export HF_HOME="${HF_HOME:-$HOME/hf_models}"

# Set Hugging Face token if available (for private/gated models)
# You can set this in your shell profile or pass it as an environment variable
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo -e "${GREEN}✓${NC} HF_TOKEN is set"
else
    echo -e "${YELLOW}⚠${NC} HF_TOKEN not set (may be required for gated models)"
fi

# ============================================================================
# GPU Configuration
# ============================================================================

# Set visible GPU devices (default: all 8 GPUs)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Number of OpenMP threads
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

echo -e "${GREEN}✓${NC} GPU devices: $CUDA_VISIBLE_DEVICES"
echo -e "${GREEN}✓${NC} OMP threads: $OMP_NUM_THREADS"

# ============================================================================
# ROCm / MIOpen Configuration
# ============================================================================

# MIOpen find mode (3 = fast mode)
export MIOPEN_FIND_MODE=3

# MIOpen enforce find mode
export MIOPEN_FIND_ENFORCE=3

# Disable direct convolution debugging
export MIOPEN_DEBUG_CONV_DIRECT=0

# Optional: Set MIOpen cache directory
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.config/miopen}"

echo -e "${GREEN}✓${NC} MIOpen configuration set"

# ============================================================================
# PyTorch Configuration
# ============================================================================

# Enable TF32 for better performance on supported hardware
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# ============================================================================
# Docker Configuration
# ============================================================================

# Docker image to use
export DOCKER_IMAGE="${DOCKER_IMAGE:-amdsiloai/pytorch-xdit:v25.11.2}"

echo -e "${GREEN}✓${NC} Docker image: $DOCKER_IMAGE"

# ============================================================================
# Directory Setup
# ============================================================================

# Create required directories
mkdir -p "$HF_HOME"
mkdir -p "$(pwd)/outputs"

echo -e "${GREEN}✓${NC} Created directory: $HF_HOME"
echo -e "${GREEN}✓${NC} Created directory: $(pwd)/outputs"

# ============================================================================
# Validation
# ============================================================================

echo ""
echo -e "${BLUE}Environment Summary:${NC}"
echo "===================="
echo "HF_HOME:              $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS:      $OMP_NUM_THREADS"
echo "DOCKER_IMAGE:         $DOCKER_IMAGE"
echo "MIOPEN_FIND_MODE:     $MIOPEN_FIND_MODE"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed"
else
    echo -e "${YELLOW}⚠${NC} Docker is not found in PATH"
fi

# Check if GPU devices are accessible
if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
    echo -e "${GREEN}✓${NC} AMD GPU devices found"
else
    echo -e "${YELLOW}⚠${NC} AMD GPU devices not found (ensure ROCm is installed)"
fi

# Check disk space in HF_HOME
AVAILABLE_SPACE=$(df -h "$HF_HOME" | awk 'NR==2 {print $4}')
echo -e "${BLUE}ℹ${NC} Available space in HF_HOME: $AVAILABLE_SPACE"

echo ""
echo -e "${GREEN}Environment setup complete!${NC}"
echo ""

# Export all variables so they're available to calling scripts
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # Script is being sourced
    echo "Variables exported to current shell"
else
    # Script is being executed
    echo "Note: Run with 'source ./setup_environment.sh' to export to current shell"
fi