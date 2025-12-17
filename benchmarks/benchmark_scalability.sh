#!/bin/bash
# Test how performance scales with number of GPUs

OUTPUT_CSV="/benchmarks/scalability_results.csv"
echo "gpus,time,fps,frames" > "$OUTPUT_CSV"

PROMPT="A cat sitting on a surfboard at the beach"
IMG_PATH="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"

for GPUS in 1 2 4 8; do
    echo "Testing with $GPUS GPUs..."
    
    CUDA_DEVICES=$(seq -s, 0 $((GPUS-1)))
    
    START=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun \
        --nproc_per_node=$GPUS \
        /app/external/xDiT/examples/wan_i2v_example.py \
        --height 720 --width 1280 --num_frames 81 \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --ulysses_degree $GPUS \
        --prompt "$PROMPT" \
        --use_torch_compile \
        --num_inference_steps 40 \
        --img_file_path "$IMG_PATH"
    
    END=$(date +%s)
    ELAPSED=$((END - START))
    FPS=$(echo "scale=2; 81 / $ELAPSED" | bc)
    
    echo "$GPUS,$ELAPSED,$FPS,81" >> "$OUTPUT_CSV"
    
    echo "Completed: $GPUS GPUs in ${ELAPSED}s (${FPS} fps)"
done

cp "$OUTPUT_CSV" /outputs/
