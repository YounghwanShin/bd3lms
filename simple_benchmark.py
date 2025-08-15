#!/usr/bin/env python3
"""
Simplified benchmark script using direct model calls
For BD3LM block sizes: 4, 8, 16 and AR model
"""

import os
import time
import torch
import json
import numpy as np
from tqdm import tqdm
import csv
import subprocess
import tempfile
from pathlib import Path

class SimpleBenchmark:
    def __init__(self):
        self.results = {}
        
    def run_generation_benchmark(self, model_type, block_size=None, num_samples=10):
        """Run generation benchmark using main.py"""
        results = {}
        
        # Create temporary log directory
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "samples"
            
            # Prepare command based on model type
            if model_type == 'bd3lm':
                cmd = [
                    'python', '-u', 'main.py',
                    f'loader.eval_batch_size=1',
                    'model=small',
                    'algo=bd3lm',
                    'algo.T=5000',
                    'algo.backbone=hf_dit',
                    'data=openwebtext-split',
                    'model.length=1024',
                    f'block_size={block_size}',
                    'wandb=null',
                    'mode=sample_eval',
                    f'eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size{block_size}',
                    'model.attn_backend=sdpa',
                    f'sampling.num_sample_batches={num_samples}',
                    'sampling.nucleus_p=0.9',
                    'sampling.kv_cache=true',
                    f'sampling.logdir={log_dir}'
                ]
            elif model_type == 'ar':
                cmd = [
                    'python', '-u', 'main.py',
                    'mode=sample_eval',
                    'loader.eval_batch_size=1',
                    'data=openwebtext-split',
                    'algo=ar',
                    'model.length=1024',
                    'eval.checkpoint_path=kuleshov-group/ar-noeos-owt',
                    'wandb=null',
                    f'sampling.num_sample_batches={num_samples}',
                    'sampling.nucleus_p=0.9',
                    f'sampling.logdir={log_dir}',
                    'sampling.kv_cache=true'
                ]
            
            print(f"Running generation benchmark for {model_type} (block_size={block_size})...")
            print("Command:", ' '.join(cmd))
            
            # Measure execution time
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=3600  # 1 hour timeout
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
                if result.stderr:
                    print("STDERR:", result.stderr[-1000:])
                
                if result.returncode != 0:
                    print(f"Error: Process returned {result.returncode}")
                    return None
                
                # Parse output for metrics
                output = result.stdout
                gen_ppl = None
                entropy = None
                nfes = None
                
                # Extract metrics from output
                for line in output.split('\n'):
                    if 'Generative perplexity:' in line:
                        try:
                            gen_ppl = float(line.split(':')[-1].strip())
                        except:
                            pass
                    if 'Entropy:' in line:
                        try:
                            entropy = float(line.split(':')[-1].strip())
                        except:
                            pass
                
                # Calculate generation speed
                tokens_generated = 1024 * num_samples  # sequence length * num samples
                generation_speed = tokens_generated / total_time
                
                results = {
                    'model_type': model_type,
                    'block_size': block_size,
                    'generation_time': total_time,
                    'generation_speed': generation_speed,
                    'generative_ppl': gen_ppl,
                    'entropy': entropy,
                    'num_samples': num_samples,
                    'tokens_per_sample': 1024
                }
                
                return results
                
            except subprocess.TimeoutExpired:
                print("Generation benchmark timed out!")
                return None
            except Exception as e:
                print(f"Error running generation benchmark: {e}")
                return None
    
    def run_ppl_benchmark(self, model_type, block_size=None):
        """Run perplexity evaluation using main.py"""
        
        if model_type == 'bd3lm':
            cmd = [
                'python', '-u', 'main.py',
                'loader.eval_batch_size=4',
                'model=small',
                'algo=bd3lm',
                'algo.backbone=hf_dit',
                'data=openwebtext-split',
                'data.insert_valid_special=False',
                'model.length=1024',
                'model.attn_backend=flex',
                f'block_size={block_size}',
                f'eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size{block_size}',
                'wandb=null',
                'mode=ppl_eval'
            ]
        elif model_type == 'ar':
            cmd = [
                'python', '-u', 'main.py',
                'loader.eval_batch_size=4',
                'model=small',
                'algo=ar',
                'data=openwebtext-split',
                'data.insert_valid_special=False',
                'model.length=1024',
                'eval.checkpoint_path=kuleshov-group/ar-noeos-owt',
                'wandb=null',
                'mode=ppl_eval'
            ]
        
        print(f"Running PPL benchmark for {model_type} (block_size={block_size})...")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            print("STDOUT:", result.stdout[-1000:])
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])
            
            if result.returncode != 0:
                print(f"Error: Process returned {result.returncode}")
                return None
            
            # Parse validation perplexity from output
            output = result.stdout
            val_ppl = None
            val_loss = None
            
            for line in output.split('\n'):
                if 'valid/loss' in line or 'val_loss' in line:
                    try:
                        # Look for loss value in the line
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'loss' in part.lower() and i + 1 < len(parts):
                                val_loss = float(parts[i + 1])
                                break
                    except:
                        pass
                if 'valid/ppl' in line or 'val_ppl' in line or 'perplexity' in line.lower():
                    try:
                        parts = line.split()
                        for part in parts:
                            if part.replace('.', '').replace('-', '').isdigit():
                                val_ppl = float(part)
                                break
                    except:
                        pass
            
            # If we got loss but not PPL, calculate it
            if val_loss is not None and val_ppl is None:
                val_ppl = np.exp(val_loss)
            
            return {
                'model_type': model_type,
                'block_size': block_size,
                'validation_ppl': val_ppl,
                'validation_loss': val_loss
            }
            
        except subprocess.TimeoutExpired:
            print("PPL benchmark timed out!")
            return None
        except Exception as e:
            print(f"Error running PPL benchmark: {e}")
            return None
    
    def benchmark_all_models(self):
        """Benchmark all models"""
        models_to_test = [
            ('bd3lm', 4),
            ('bd3lm', 8),
            ('bd3lm', 16),
            ('ar', None)
        ]
        
        all_results = []
        
        for model_type, block_size in models_to_test:
            print(f"\n{'='*60}")
            print(f"Benchmarking {model_type}" + (f" (block_size={block_size})" if block_size else ""))
            print(f"{'='*60}")
            
            result = {}
            result['model_type'] = model_type
            result['block_size'] = block_size
            
            # Generation benchmark
            gen_results = self.run_generation_benchmark(model_type, block_size, num_samples=5)
            if gen_results:
                result.update(gen_results)
            
            # PPL benchmark  
            ppl_results = self.run_ppl_benchmark(model_type, block_size)
            if ppl_results:
                result.update(ppl_results)
            
            # Add average diffusion steps info
            if model_type == 'bd3lm':
                # For BD3LM, diffusion steps depend on T and block size
                result['avg_diffusion_steps'] = 5000 / block_size  # Approximation
            else:
                result['avg_diffusion_steps'] = 1  # AR is one step per token
            
            all_results.append(result)
            
            print(f"Results for {model_type}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        # Save results
        self.save_results(all_results)
        self.print_summary_table(all_results)
        
        return all_results
    
    def save_results(self, results):
        """Save results to files"""
        # Save as JSON
        with open('simple_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        if results:
            with open('simple_benchmark_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        print("\nResults saved to simple_benchmark_results.json and simple_benchmark_results.csv")
    
    def print_summary_table(self, results):
        """Print summary table"""
        print(f"\n{'='*100}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*100}")
        
        print(f"{'Model':<10} {'Block':<6} {'Speed(tok/s)':<12} {'Gen PPL':<10} {'Val PPL':<10} {'Avg Steps':<10}")
        print(f"{'-'*100}")
        
        for result in results:
            model = result.get('model_type', 'N/A')
            block = str(result.get('block_size', 'N/A'))
            speed = f"{result.get('generation_speed', 0):.1f}" if result.get('generation_speed') else 'N/A'
            gen_ppl = f"{result.get('generative_ppl', 0):.2f}" if result.get('generative_ppl') else 'N/A'
            val_ppl = f"{result.get('validation_ppl', 0):.2f}" if result.get('validation_ppl') else 'N/A'
            steps = f"{result.get('avg_diffusion_steps', 0):.1f}" if result.get('avg_diffusion_steps') else 'N/A'
            
            print(f"{model:<10} {block:<6} {speed:<12} {gen_ppl:<10} {val_ppl:<10} {steps:<10}")


def main():
    """Main function"""
    benchmark = SimpleBenchmark()
    benchmark.benchmark_all_models()


if __name__ == '__main__':
    main()
