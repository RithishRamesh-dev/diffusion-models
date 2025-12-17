#!/bin/bash
set -e

BENCHMARK_DIR="/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUITE_DIR="${BENCHMARK_DIR}/suite_${TIMESTAMP}"

mkdir -p "$SUITE_DIR"
cd "$SUITE_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     WAN 2.2 I2V COMPREHENSIVE BENCHMARK SUITE                 ║"
echo "║     Started: $(date)                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Copy config files
cp /benchmarks/benchmark_configs.json .
cp /benchmarks/run_benchmarks.py .
cp /benchmarks/benchmark_scalability.sh .
cp /benchmarks/benchmark_quality_speed.sh .

# Start GPU monitoring
echo "Starting GPU monitoring..."
/benchmarks/monitor_gpus.sh > gpu_monitor.log 2>&1 &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"

# Run main benchmark suite
echo ""
echo "=== Phase 1: Main Benchmark Suite ==="
python3 run_benchmarks.py 2>&1 | tee main_benchmark.log

# Run scalability tests
echo ""
echo "=== Phase 2: Scalability Testing ==="
./benchmark_scalability.sh 2>&1 | tee scalability.log

# Run quality/speed trade-off tests
echo ""
echo "=== Phase 3: Quality vs Speed Trade-off ==="
./benchmark_quality_speed.sh 2>&1 | tee quality_speed.log

# Stop monitoring
echo ""
echo "Stopping GPU monitoring..."
kill $MONITOR_PID

# Generate summary report
echo ""
echo "=== Generating Summary Report ==="

cat > summary_report.txt << REPORT
╔════════════════════════════════════════════════════════════════╗
║     WAN 2.2 I2V BENCHMARK SUMMARY REPORT                       ║
║     Completed: $(date)                          ║
╚════════════════════════════════════════════════════════════════╝

SUITE DIRECTORY: ${SUITE_DIR}

FILES GENERATED:
$(ls -lh *.csv *.log *.txt 2>/dev/null)

SYSTEM INFORMATION:
$(rocm-smi --showproductname 2>/dev/null || echo "ROCm info unavailable")

GPU COUNT: $(rocm-smi --showid | grep -c GPU || echo "Unknown")

DISK USAGE:
$(df -h /hf_home /outputs 2>/dev/null || echo "Unavailable")

MAIN BENCHMARK RESULTS:
$(tail -20 main_benchmark.log)

SCALABILITY RESULTS:
$(cat scalability_results.csv 2>/dev/null || echo "Not available")

QUALITY/SPEED TRADE-OFF:
$(cat quality_speed_tradeoff.csv 2>/dev/null || echo "Not available")

REPORT

cat summary_report.txt

# Copy everything to outputs
echo ""
echo "=== Copying results to /outputs ==="
cp -r "$SUITE_DIR" /outputs/

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     BENCHMARK SUITE COMPLETE                                   ║"
echo "║     Results saved to: /outputs/suite_${TIMESTAMP}             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
