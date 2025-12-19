#!/usr/bin/env python3
"""
Wan2.2 I2V Benchmarking Script - Fixed Version
Uses correct precision flags: --use_bf16_te_gemms and --use_fp8_gemms
"""

import json
import subprocess
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path
import argparse

class Wan22Benchmarker:
    def __init__(self, config_file, output_dir="benchmark_results"):
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def parse_log_output(self, output):
        """Extract timing information from command output"""
        timings = {
            'pipe_time': None,
            'vae_decode_time': None,
            'total_time': None
        }
        
        # Parse pipe epoch time
        pipe_match = re.search(r'Pipe epoch time:\s*([\d.]+)\s*sec', output)
        if pipe_match:
            timings['pipe_time'] = float(pipe_match.group(1))
        
        # Parse VAE decode time
        vae_match = re.search(r'VAE decode epoch time:\s*([\d.]+)\s*sec', output)
        if vae_match:
            timings['vae_decode_time'] = float(vae_match.group(1))
        
        # Parse JSON timing output if available
        json_match = re.search(r'\[{.*?}\]', output, re.DOTALL)
        if json_match:
            try:
                json_timings = json.loads(json_match.group(0))
                if json_timings:
                    timings.update(json_timings[0])
            except json.JSONDecodeError:
                pass
        
        return timings
    
    def build_command(self, test_case, prompt, precision="default"):
        """Build the torchrun command for a test case
        
        Precision modes:
        - default: BF16 (no special flags)
        - bf16_te: BF16 with BF16 timestep embeddings (--use_bf16_te_gemms)
        - fp8: FP8 quantized linear layers (--use_fp8_gemms)
        - fp8_bf16_te: FP8 + BF16 timestep embeddings (both flags)
        """
        nproc = test_case.get('ulysses_degree', 8)
        
        cmd = [
            'torchrun',
            f'--nproc_per_node={nproc}',
            '/app/Wan/run.py',
            '--task', 'i2v',
            '--height', str(test_case['height']),
            '--width', str(test_case['width']),
            '--model', 'Wan-AI/Wan2.2-I2V-A14B-Diffusers',
            '--img_file_path', '/app/Wan/i2v_input.JPG',
            '--ulysses_degree', str(test_case['ulysses_degree']),
            '--seed', '42',
            '--num_frames', str(test_case['num_frames']),
            '--prompt', prompt,
            '--num_repetitions', '1',
            '--num_inference_steps', str(test_case['num_inference_steps']),
        ]
        
        # Add torch compile flag if specified
        if test_case.get('use_torch_compile', True):
            cmd.append('--use_torch_compile')
        
        # Add precision-specific flags based on mode
        if precision == "bf16_te":
            cmd.append('--use_bf16_te_gemms')
        elif precision == "fp8":
            cmd.append('--use_fp8_gemms')
        elif precision == "fp8_bf16_te":
            cmd.append('--use_fp8_gemms')
            cmd.append('--use_bf16_te_gemms')
        # "default" uses no extra flags (standard BF16)
        
        return cmd
    
    def run_benchmark(self, test_case, prompt, precision="default"):
        """Run a single benchmark configuration"""
        test_name = f"{test_case['name']}_{precision}"
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"{'='*80}")
        
        cmd = self.build_command(test_case, prompt, precision)
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run the command and capture output
            result = subprocess.run(
                cmd,
                cwd='/app/Wan',
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Parse timings from output
            output = result.stdout + result.stderr
            timings = self.parse_log_output(output)
            timings['total_time'] = total_time
            
            # Store results
            benchmark_result = {
                'test_name': test_name,
                'timestamp': datetime.now().isoformat(),
                'config': test_case,
                'precision': precision,
                'prompt': prompt[:100] + "...",
                'timings': timings,
                'return_code': result.returncode,
                'success': result.returncode == 0
            }
            
            # Save output logs
            log_file = self.output_dir / f"{test_name}_{self.timestamp}.log"
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*80}\n")
                f.write(output)
            
            self.results.append(benchmark_result)
            
            print(f"\n{'='*80}")
            print(f"Results for {test_name}:")
            print(f"  Pipe Time: {timings.get('pipe_time', 'N/A')} sec")
            print(f"  VAE Decode Time: {timings.get('vae_decode_time', 'N/A')} sec")
            print(f"  Total Time: {total_time:.2f} sec")
            print(f"  Status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}")
            
            if not result.returncode == 0:
                print(f"\n  Error output (last 1000 chars):")
                print(f"  {output[-1000:]}")
            
            print(f"{'='*80}\n")
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: Benchmark timed out for {test_name}")
            benchmark_result = {
                'test_name': test_name,
                'timestamp': datetime.now().isoformat(),
                'config': test_case,
                'precision': precision,
                'error': 'Timeout',
                'success': False
            }
            self.results.append(benchmark_result)
            return benchmark_result
        
        except Exception as e:
            print(f"ERROR: {str(e)}")
            benchmark_result = {
                'test_name': test_name,
                'timestamp': datetime.now().isoformat(),
                'config': test_case,
                'precision': precision,
                'error': str(e),
                'success': False
            }
            self.results.append(benchmark_result)
            return benchmark_result
    
    def run_all_benchmarks(self, precision_list=None):
        """Run all benchmark configurations
        
        Precision modes:
        - default: Standard BF16 (no flags)
        - bf16_te: BF16 with BF16 timestep embeddings
        - fp8: FP8 quantized linear layers
        - fp8_bf16_te: FP8 + BF16 timestep embeddings
        """
        if precision_list is None:
            precision_list = ['default', 'fp8']
        
        test_cases = self.config.get('test_cases', [])
        prompts = self.config.get('prompts', ['Default prompt'])
        
        total_tests = len(test_cases) * len(prompts) * len(precision_list)
        current_test = 0
        
        print(f"\n{'='*80}")
        print(f"Starting Benchmark Suite: {total_tests} total tests")
        print(f"Precision modes: {', '.join(precision_list)}")
        print(f"{'='*80}\n")
        
        for test_case in test_cases:
            for precision in precision_list:
                for prompt_idx, prompt in enumerate(prompts):
                    current_test += 1
                    print(f"\nProgress: {current_test}/{total_tests}")
                    
                    self.run_benchmark(test_case, prompt, precision)
                    
                    # Small delay between tests
                    time.sleep(2)
        
        # Save final results
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """Save benchmark results to JSON file"""
        results_file = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    def generate_report(self):
        """Generate a human-readable benchmark report"""
        report_file = self.output_dir / f"benchmark_report_{self.timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WAN 2.2 I2V BENCHMARK REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(self.results)}\n")
            
            successful = sum(1 for r in self.results if r.get('success', False))
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {len(self.results) - successful}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PRECISION MODES EXPLANATION\n")
            f.write("="*80 + "\n")
            f.write("default:      Standard BF16 (no special flags)\n")
            f.write("bf16_te:      BF16 + BF16 timestep embeddings (--use_bf16_te_gemms)\n")
            f.write("fp8:          FP8 quantized linear layers (--use_fp8_gemms)\n")
            f.write("fp8_bf16_te:  FP8 + BF16 timestep embeddings (both flags)\n\n")
            
            f.write("="*80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for result in self.results:
                f.write(f"Test: {result['test_name']}\n")
                f.write(f"Status: {'SUCCESS' if result.get('success') else 'FAILED'}\n")
                f.write(f"Precision Mode: {result.get('precision', 'N/A')}\n")
                
                if 'timings' in result:
                    timings = result['timings']
                    f.write(f"  Pipe Time: {timings.get('pipe_time', 'N/A')} sec\n")
                    f.write(f"  VAE Decode Time: {timings.get('vae_decode_time', 'N/A')} sec\n")
                    f.write(f"  Total Time: {timings.get('total_time', 'N/A')} sec\n")
                
                if 'config' in result:
                    config = result['config']
                    f.write(f"  Resolution: {config['width']}x{config['height']}\n")
                    f.write(f"  Frames: {config['num_frames']}\n")
                    f.write(f"  Steps: {config['num_inference_steps']}\n")
                    f.write(f"  Ulysses Degree: {config['ulysses_degree']}\n")
                
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                
                f.write("\n" + "-"*80 + "\n\n")
            
            # Performance comparison table
            f.write("="*80 + "\n")
            f.write("PERFORMANCE COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Test Name':<40} {'Precision':<15} {'Pipe Time':<12} {'Total Time':<12}\n")
            f.write("-"*80 + "\n")
            
            for result in self.results:
                if result.get('success') and 'timings' in result:
                    timings = result['timings']
                    pipe_time = timings.get('pipe_time', 'N/A')
                    total_time = timings.get('total_time', 'N/A')
                    
                    pipe_str = f"{pipe_time:.2f}" if isinstance(pipe_time, (int, float)) else str(pipe_time)
                    total_str = f"{total_time:.2f}" if isinstance(total_time, (int, float)) else str(total_time)
                    
                    f.write(f"{result['test_name']:<40} {result.get('precision', 'N/A'):<15} {pipe_str:<12} {total_str:<12}\n")
            
            # Speedup analysis by precision
            f.write("\n" + "="*80 + "\n")
            f.write("PRECISION SPEEDUP ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            # Group by base test name
            test_groups = {}
            for result in self.results:
                if result.get('success') and 'timings' in result:
                    base_name = result['config']['name']
                    if base_name not in test_groups:
                        test_groups[base_name] = []
                    test_groups[base_name].append(result)
            
            for base_name, results in test_groups.items():
                f.write(f"\n{base_name}:\n")
                default_time = None
                
                # Find default precision time
                for result in results:
                    if result.get('precision') == 'default':
                        default_time = result['timings'].get('pipe_time')
                        break
                
                if default_time:
                    f.write(f"  Baseline (default): {default_time:.2f}s\n")
                    for result in results:
                        precision = result.get('precision', 'N/A')
                        pipe_time = result['timings'].get('pipe_time')
                        if pipe_time and precision != 'default':
                            speedup = default_time / pipe_time
                            f.write(f"  {precision:<15} {pipe_time:.2f}s ({speedup:.2f}x)\n")
                else:
                    f.write("  No baseline found for comparison\n")
        
        print(f"Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Wan2.2 I2V Benchmarking Tool')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to benchmark configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to store benchmark results')
    parser.add_argument('--precision', type=str, nargs='+', 
                        default=['default', 'fp8'],
                        choices=['default', 'bf16_te', 'fp8', 'fp8_bf16_te'],
                        help='Precision modes to test')
    
    args = parser.parse_args()
    
    # Verify we're in the right environment
    if not os.path.exists('/app/Wan'):
        print("ERROR: This script must be run inside the ROCm PyTorch xDiT container")
        print("Expected /app/Wan directory not found")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("WAN 2.2 I2V BENCHMARKING TOOL")
    print("="*80)
    print("\nPrecision Modes Available:")
    print("  default:      Standard BF16 (no special flags)")
    print("  bf16_te:      BF16 + BF16 timestep embeddings")
    print("  fp8:          FP8 quantized linear layers")
    print("  fp8_bf16_te:  FP8 + BF16 timestep embeddings")
    print(f"\nSelected modes: {', '.join(args.precision)}\n")
    
    benchmarker = Wan22Benchmarker(args.config, args.output_dir)
    benchmarker.run_all_benchmarks(precision_list=args.precision)
    
    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()