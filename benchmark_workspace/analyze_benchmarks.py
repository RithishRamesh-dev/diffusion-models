#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob
import os

def analyze_main_results(csv_file):
    """Analyze main benchmark results"""
    df = pd.read_csv(csv_file)
    
    print("\n=== MAIN BENCHMARK ANALYSIS ===\n")
    
    # Success rate
    success_rate = (df['success'].sum() / len(df)) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Filter successful runs
    df_success = df[df['success'] == True]
    
    if len(df_success) == 0:
        print("No successful runs to analyze")
        return
    
    # Stats by test configuration
    print("\n--- Performance by Configuration ---")
    stats = df_success.groupby('test_name').agg({
        'elapsed_time': ['mean', 'std', 'min', 'max'],
        'fps': ['mean', 'std']
    }).round(2)
    print(stats)
    
    # Resolution impact
    print("\n--- Performance by Resolution ---")
    df_success['resolution'] = df_success['width'].astype(str) + 'x' + df_success['height'].astype(str)
    res_stats = df_success.groupby('resolution').agg({
        'elapsed_time': 'mean',
        'fps': 'mean'
    }).round(2)
    print(res_stats)
    
    # GPU scaling
    print("\n--- Performance by GPU Count ---")
    gpu_stats = df_success.groupby('ulysses_degree').agg({
        'elapsed_time': 'mean',
        'fps': 'mean'
    }).round(2)
    print(gpu_stats)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time by config
    df_success.groupby('test_name')['elapsed_time'].mean().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Average Time by Configuration')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: FPS by config
    df_success.groupby('test_name')['fps'].mean().plot(kind='bar', ax=axes[0,1], color='green')
    axes[0,1].set_title('Average FPS by Configuration')
    axes[0,1].set_ylabel('FPS')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: GPU scaling
    gpu_stats['elapsed_time'].plot(kind='line', marker='o', ax=axes[1,0])
    axes[1,0].set_title('Time vs GPU Count')
    axes[1,0].set_xlabel('Number of GPUs')
    axes[1,0].set_ylabel('Time (seconds)')
    axes[1,0].grid(True)
    
    # Plot 4: Inference steps impact
    steps_stats = df_success.groupby('num_inference_steps')['elapsed_time'].mean()
    steps_stats.plot(kind='line', marker='o', ax=axes[1,1], color='red')
    axes[1,1].set_title('Time vs Inference Steps')
    axes[1,1].set_xlabel('Number of Inference Steps')
    axes[1,1].set_ylabel('Time (seconds)')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_analysis.png', dpi=300)
    print(f"\nPlot saved to: benchmark_analysis.png")

def analyze_scalability(csv_file):
    """Analyze scalability results"""
    print("\n=== SCALABILITY ANALYSIS ===\n")
    
    df = pd.read_csv(csv_file)
    print(df)
    
    # Calculate speedup
    baseline = df[df['gpus'] == 1]['time'].values[0]
    df['speedup'] = baseline / df['time']
    df['efficiency'] = (df['speedup'] / df['gpus']) * 100
    
    print("\n--- Speedup Analysis ---")
    print(df[['gpus', 'time', 'fps', 'speedup', 'efficiency']])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(df['gpus'], df['speedup'], marker='o', linewidth=2, markersize=8)
    axes[0].plot(df['gpus'], df['gpus'], '--', label='Linear scaling', alpha=0.5)
    axes[0].set_title('Speedup vs GPU Count')
    axes[0].set_xlabel('Number of GPUs')
    axes[0].set_ylabel('Speedup')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(df['gpus'], df['efficiency'], marker='o', linewidth=2, markersize=8, color='green')
    axes[1].set_title('Parallel Efficiency')
    axes[1].set_xlabel('Number of GPUs')
    axes[1].set_ylabel('Efficiency (%)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300)
    print(f"\nPlot saved to: scalability_analysis.png")

def analyze_quality_speed(csv_file):
    """Analyze quality vs speed trade-off"""
    print("\n=== QUALITY vs SPEED ANALYSIS ===\n")
    
    df = pd.read_csv(csv_file)
    print(df)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(df['steps'], df['time'], marker='o', linewidth=2, markersize=8)
    axes[0].set_title('Generation Time vs Inference Steps')
    axes[0].set_xlabel('Inference Steps')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].grid(True)
    
    axes[1].plot(df['steps'], df['fps'], marker='o', linewidth=2, markersize=8, color='green')
    axes[1].set_title('FPS vs Inference Steps')
    axes[1].set_xlabel('Inference Steps')
    axes[1].set_ylabel('FPS')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('quality_speed_analysis.png', dpi=300)
    print(f"\nPlot saved to: quality_speed_analysis.png")

if __name__ == '__main__':
    # Find benchmark results
    result_dirs = glob.glob('./wan_22_i2v_outputs/suite_*')
    
    if not result_dirs:
        print("No benchmark results found in ./wan_22_i2v_outputs/")
        sys.exit(1)
    
    # Use most recent
    latest_dir = max(result_dirs, key=os.path.getctime)
    print(f"Analyzing results from: {latest_dir}")
    
    # Find CSV files
    main_csv = glob.glob(f"{latest_dir}/benchmark_results_*.csv")
    scalability_csv = glob.glob(f"{latest_dir}/scalability_results.csv")
    quality_csv = glob.glob(f"{latest_dir}/quality_speed_tradeoff.csv")
    
    if main_csv:
        analyze_main_results(main_csv[0])
    
    if scalability_csv:
        analyze_scalability(scalability_csv[0])
    
    if quality_csv:
        analyze_quality_speed(quality_csv[0])
    
    print("\n=== ANALYSIS COMPLETE ===")
