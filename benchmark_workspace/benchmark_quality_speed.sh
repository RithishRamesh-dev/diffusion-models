#!/bin/bash
# Test different inference steps to find quality/speed sweet spot

OUTPUT_CSV="/benchmarks/quality_speed_tradeoff.csv"
echo "steps,time,fps" > "$OUTPUT_CSV"

PROMPT="A cat sitting on a surfboard at the beach"
IMG_PATH="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"

for STEPS in 20 25 30 35 40 45 50; do
    echo "Testing with $STEPS inference steps..."
    
    START=$(date +%s.%N)
    
    torchrun --nproc_per_node=8 \
        /app/external/xDiT/examples/wan_i2v_example.py \
        --height 720 --width 1280 --num_frames 81 \
        --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
        --ulysses_degree 8 \
        --prompt "$PROMPT" \
        --use_torch_compile \
        --num_inference_steps $STEPS \
        --img_file_path "$IMG_PATH"
    
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    FPS=$(echo "scale=2; 81 / $ELAPSED" | bc)
    
    echo "$STEPS,$ELAPSED,$FPS" >> "$OUTPUT_CSV"
    
    # Rename output to preserve different quality levels
    mv i2v_output.mp4 "/outputs/quality_test_steps${STEPS}.mp4"
    
    echo "Completed: $STEPS steps in ${ELAPSED}s"
done

cp "$OUTPUT_CSV" /outputs/
