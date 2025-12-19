#!/bin/bash
# Monitor GPU usage during benchmarks

OUTPUT_FILE="/benchmarks/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "Starting GPU monitoring... (Output: $OUTPUT_FILE)"
echo "Press Ctrl+C to stop"

while true; do
    echo "=== $(date) ===" >> "$OUTPUT_FILE"
    rocm-smi >> "$OUTPUT_FILE" 2>&1
    echo "" >> "$OUTPUT_FILE"
    sleep 5
done
