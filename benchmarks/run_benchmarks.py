#!/usr/bin/env python3
import json
import subprocess
import time
import csv
import os
from datetime import datetime
import sys

def run_inference(config, prompt, output_dir, img_path):
    """Run a single inference and return timing metrics"""
    
    # Prepare CUDA devices based on ulysses_degree
    num_gpus = config['ulysses_degree']
    cuda_devices = ','.join(map(str, range(num_gpus)))
    
    # Build command
    cmd = [
        'torchrun',
        f'--nproc_per_node={num_gpus}',
        '/app/external/xDiT/examples/wan_i2v_example.py',
        '--height', str(config['height']),
        '--width', str(config['width']),
        '--num_frames', str(config['num_frames']),
        '--model', 'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
        '--ulysses_degree', str(config['ulysses_degree']),
        '--prompt', prompt,
        '--use_torch_compile',
        '--num_inference_steps', str(config['num_inference_steps']),
        '--img_file_path', img_path
    ]
    
    # Set environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_devices
    
    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"GPUs: {num_gpus} | Resolution: {config['width']}x{config['height']}")
    print(f"Frames: {config['num_frames']} | Steps: {config['num_inference_steps']}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"{'='*80}\n")
    
    # Run and time
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Check for success
        success = result.returncode == 0
        
        if success:
            print(f"✓ Success in {elapsed:.2f}s")
        else:
            print(f"✗ Failed with return code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")
        
        return {
            'success': success,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("✗ Timeout exceeded (30 minutes)")
        return {
            'success': False,
            'elapsed_time': 1800,
            'stdout': '',
            'stderr': 'Timeout exceeded'
        }
    except Exception as e:
        print(f"✗ Exception: {str(e)}")
        return {
            'success': False,
            'elapsed_time': 0,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    # Load configurations
    with open('benchmark_configs.json', 'r') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    prompts = data['prompts']
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'benchmark_results_{timestamp}.csv'
    
    img_path = 'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG'
    
    # CSV header
    fieldnames = [
        'timestamp', 'test_name', 'prompt_id', 'height', 'width', 
        'num_frames', 'num_inference_steps', 'ulysses_degree',
        'elapsed_time', 'fps', 'success', 'error_message'
    ]
    
    results = []
    total_tests = len(test_cases) * len(prompts)
    current_test = 0
    
    print(f"\n{'#'*80}")
    print(f"# BENCHMARK SUITE - {timestamp}")
    print(f"# Total tests: {total_tests}")
    print(f"# Test cases: {len(test_cases)}")
    print(f"# Prompts: {len(prompts)}")
    print(f"{'#'*80}\n")
    
    # Run all combinations
    for test_case in test_cases:
        for i, prompt in enumerate(prompts):
            current_test += 1
            
            print(f"\n[Test {current_test}/{total_tests}]")
            
            result = run_inference(test_case, prompt, '/outputs', img_path)
            
            # Calculate FPS
            if result['success']:
                fps = test_case['num_frames'] / result['elapsed_time']
            else:
                fps = 0
            
            # Record results
            row = {
                'timestamp': datetime.now().isoformat(),
                'test_name': test_case['name'],
                'prompt_id': i,
                'height': test_case['height'],
                'width': test_case['width'],
                'num_frames': test_case['num_frames'],
                'num_inference_steps': test_case['num_inference_steps'],
                'ulysses_degree': test_case['ulysses_degree'],
                'elapsed_time': result['elapsed_time'],
                'fps': fps,
                'success': result['success'],
                'error_message': result['stderr'][:200] if not result['success'] else ''
            }
            
            results.append(row)
            
            # Save incrementally
            with open(results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\nProgress: {current_test}/{total_tests} tests completed")
            print(f"Results saved to: {results_file}")
    
    # Generate summary
    print(f"\n{'#'*80}")
    print(f"# BENCHMARK COMPLETE")
    print(f"{'#'*80}\n")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_file}")
    
    if successful > 0:
        avg_time = sum(r['elapsed_time'] for r in results if r['success']) / successful
        avg_fps = sum(r['fps'] for r in results if r['success']) / successful
        print(f"\nAverage time (successful): {avg_time:.2f}s")
        print(f"Average FPS (successful): {avg_fps:.2f}")
    
    # Copy results to output directory
    subprocess.run(['cp', results_file, '/outputs/'])
    print(f"\nResults also copied to: /outputs/{results_file}")

if __name__ == '__main__':
    main()
