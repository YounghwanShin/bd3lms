#!/usr/bin/env python3
"""
Accurate inference benchmark for BD3LM models
Measures generation speed, diffusion steps, and generative perplexity
"""

import time
import torch
import json
import numpy as np
import subprocess
import tempfile
import csv
from pathlib import Path
import os
import re

class BD3LMBenchmark:
    def __init__(self):
        self.results = []
        
    def run_single_benchmark(self, block_size, num_samples=5, T_steps=1000):
        """Run benchmark for a single BD3LM model"""
        print(f"\n{'='*60}")
        print(f"Benchmarking BD3LM Block Size {block_size}")
        print(f"{'='*60}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "samples"
            
            cmd = [
                'python', '-u', 'main.py',
                'loader.eval_batch_size=1',
                'model=small',
                'algo=bd3lm',
                f'algo.T={T_steps}',
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
                f'sampling.logdir={log_dir}',
                'seed=42'  # Fixed seed for consistency
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Measure wall-clock time
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=1800  # 30 minutes timeout
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                
                if result.returncode != 0:
                    print(f"Error: Process returned {result.returncode}")
                    print("STDERR:", result.stderr[-2000:])
                    return None
                
                # Parse output for metrics
                output = result.stdout
                gen_ppl = None
                entropy = None
                
                # Extract metrics using regex
                ppl_match = re.search(r'Generative perplexity:\s*tensor\(([0-9\.]+)', output)
                if ppl_match:
                    gen_ppl = float(ppl_match.group(1))
                
                entropy_match = re.search(r'Entropy:\s*tensor\(([0-9\.]+)', output)
                if entropy_match:
                    entropy = float(entropy_match.group(1))
                
                # Calculate metrics
                total_tokens = 1024 * num_samples
                generation_speed = total_tokens / total_time
                
                # Calculate diffusion steps
                # For BD3LM: num_blocks = seq_len / block_size, steps per block = T_steps
                num_blocks = 1024 // block_size
                avg_diffusion_steps = T_steps  # Each block uses T_steps
                
                result_dict = {
                    'model_type': 'bd3lm',
                    'block_size': block_size,
                    'T_steps': T_steps,
                    'num_samples': num_samples,
                    'total_time_seconds': total_time,
                    'generation_speed_tokens_per_sec': generation_speed,
                    'generative_ppl': gen_ppl,
                    'entropy': entropy,
                    'avg_diffusion_steps_per_block': T_steps,
                    'num_blocks': num_blocks,
                    'total_diffusion_steps': T_steps * num_blocks,
                    'effective_steps_per_token': T_steps / block_size
                }
                
                print(f"✓ BD3LM-{block_size} Results:")
                print(f"  Generation Speed: {generation_speed:.2f} tokens/sec")
                print(f"  Total Time: {total_time:.2f} seconds")
                print(f"  Generative PPL: {gen_ppl:.2f}" if gen_ppl else "  Generative PPL: N/A")
                print(f"  Entropy: {entropy:.2f}" if entropy else "  Entropy: N/A")
                print(f"  Avg Diffusion Steps: {T_steps} per block")
                print(f"  Effective Steps per Token: {T_steps/block_size:.2f}")
                
                return result_dict
                
            except subprocess.TimeoutExpired:
                print(f"Benchmark timed out for block_size {block_size}")
                return None
            except Exception as e:
                print(f"Error running benchmark: {e}")
                return None
    
    def run_ar_benchmark(self, num_samples=5):
        """Run benchmark for AR model (GPT-2 style)"""
        print(f"\n{'='*60}")
        print(f"Benchmarking Autoregressive (GPT-2) Model")
        print(f"{'='*60}")
        
        # Create a simple AR generation script since the HF model might not be available
        try:
            import transformers
            
            # Use GPT-2 as AR baseline
            model_name = "gpt2"
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
            model = transformers.GPT2LMHeadModel.from_pretrained(model_name).cuda()
            
            # Set pad token
            tokenizer.pad_token = tokenizer.eos_token
            
            print(f"Using {model_name} as AR baseline")
            
            # Warmup
            prompt = "<|endoftext|>"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                _ = model.generate(
                    inputs.input_ids,
                    max_length=1024,
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Actual benchmark
            start_time = time.time()
            generated_texts = []
            
            for i in range(num_samples):
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=1024,
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                generated_texts.append(text)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            total_tokens = 1024 * num_samples
            generation_speed = total_tokens / total_time
            
            # Calculate generative perplexity using GPT-2 itself
            total_loss = 0
            total_token_count = 0
            
            model.eval()
            with torch.no_grad():
                for text in generated_texts:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
                    outputs = model(**inputs, labels=inputs.input_ids)
                    total_loss += outputs.loss.item() * inputs.input_ids.shape[1]
                    total_token_count += inputs.input_ids.shape[1]
            
            avg_loss = total_loss / total_token_count
            gen_ppl = np.exp(avg_loss)
            
            result_dict = {
                'model_type': 'ar_gpt2',
                'block_size': 1,  # AR generates one token at a time
                'T_steps': 1,
                'num_samples': num_samples,
                'total_time_seconds': total_time,
                'generation_speed_tokens_per_sec': generation_speed,
                'generative_ppl': gen_ppl,
                'entropy': None,
                'avg_diffusion_steps_per_block': 1,
                'num_blocks': 1024,
                'total_diffusion_steps': 1024,  # One step per token
                'effective_steps_per_token': 1.0
            }
            
            print(f"✓ AR (GPT-2) Results:")
            print(f"  Generation Speed: {generation_speed:.2f} tokens/sec")
            print(f"  Total Time: {total_time:.2f} seconds")
            print(f"  Generative PPL: {gen_ppl:.2f}")
            print(f"  Steps per Token: 1.0 (autoregressive)")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
            return result_dict
            
        except Exception as e:
            print(f"Error running AR benchmark: {e}")
            return None
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("Starting BD3LM Inference Benchmark")
        print(f"{'='*80}")
        
        # Test different configurations
        block_sizes = [4, 8, 16]
        T_values = [1000]  # Use smaller T for faster benchmarking
        num_samples = 3  # Reduced for faster testing
        
        # BD3LM benchmarks
        for block_size in block_sizes:
            for T in T_values:
                result = self.run_single_benchmark(block_size, num_samples, T)
                if result:
                    self.results.append(result)
        
        # AR benchmark
        ar_result = self.run_ar_benchmark(num_samples)
        if ar_result:
            self.results.append(ar_result)
        
        # Save and display results
        self.save_results()
        self.print_summary()
        
        return self.results
    
    def save_results(self):
        """Save results to files"""
        # Save as JSON
        with open('bd3lm_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        if self.results:
            with open('bd3lm_benchmark_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        print(f"\n📊 Results saved to bd3lm_benchmark_results.json and bd3lm_benchmark_results.csv")
    
    def print_summary(self):
        """Print comprehensive summary table"""
        if not self.results:
            print("No results to display")
            return
            
        print(f"\n{'='*120}")
        print("BD3LM INFERENCE BENCHMARK SUMMARY")
        print(f"{'='*120}")
        
        print(f"{'Model':<12} {'Block':<6} {'Speed':<12} {'Gen PPL':<10} {'Entropy':<10} {'Steps/Token':<12} {'Total Time':<12}")
        print(f"{'-'*120}")
        
        for result in self.results:
            model = result.get('model_type', 'N/A')
            if model == 'bd3lm':
                model_name = f"BD3LM-{result.get('block_size', 'N/A')}"
            else:
                model_name = "GPT-2"
            
            block = str(result.get('block_size', 'N/A'))
            speed = f"{result.get('generation_speed_tokens_per_sec', 0):.1f}" if result.get('generation_speed_tokens_per_sec') else 'N/A'
            gen_ppl = f"{result.get('generative_ppl', 0):.2f}" if result.get('generative_ppl') else 'N/A'
            entropy = f"{result.get('entropy', 0):.2f}" if result.get('entropy') else 'N/A'
            steps_per_token = f"{result.get('effective_steps_per_token', 0):.2f}" if result.get('effective_steps_per_token') else 'N/A'
            total_time = f"{result.get('total_time_seconds', 0):.1f}s" if result.get('total_time_seconds') else 'N/A'
            
            print(f"{model_name:<12} {block:<6} {speed:<12} {gen_ppl:<10} {entropy:<10} {steps_per_token:<12} {total_time:<12}")
        
        print(f"\nKey Insights:")
        print(f"• Generation Speed: Higher is better (tokens/second)")
        print(f"• Generative PPL: Lower is better (quality of generated text)")
        print(f"• Steps/Token: BD3LM trade-off between quality and efficiency")
        print(f"• BD3LM with larger blocks = fewer steps per token but potentially lower quality")


def main():
    """Main function"""
    benchmark = BD3LMBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == '__main__':
    main()
