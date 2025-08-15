#!/usr/bin/env python3
"""
Benchmark script for BD3LM inference evaluation
Measures:
1. Generation speed (tokens/second)
2. Average diffusion steps
3. Perplexity (PPL)

For BD3LM block sizes: 4, 8, 16
And baseline AR (GPT2) model
"""

import os
import time
import torch
import hydra
import omegaconf
import transformers
import json
import numpy as np
from tqdm import tqdm
import dataloader
import diffusion
import utils
import csv

# Config for different models
MODEL_CONFIGS = {
    'bd3lm_4': {
        'block_size': 4,
        'checkpoint_path': 'kuleshov-group/bd3lm-owt-block_size4',
        'algo': 'bd3lm',
        'backbone': 'hf_dit'
    },
    'bd3lm_8': {
        'block_size': 8,
        'checkpoint_path': 'kuleshov-group/bd3lm-owt-block_size8',
        'algo': 'bd3lm',
        'backbone': 'hf_dit'
    },
    'bd3lm_16': {
        'block_size': 16,
        'checkpoint_path': 'kuleshov-group/bd3lm-owt-block_size16',
        'algo': 'bd3lm',
        'backbone': 'hf_dit'
    },
    'ar_gpt2': {
        'block_size': 1024,  # Full sequence for AR
        'checkpoint_path': 'kuleshov-group/ar-noeos-owt',
        'algo': 'ar',
        'backbone': 'dit'
    }
}

class InferenceBenchmark:
    def __init__(self, base_config_path='configs/config.yaml'):
        self.base_config_path = base_config_path
        self.results = {}
        
    def load_model_config(self, model_name):
        """Load hydra config for a specific model"""
        with hydra.initialize(version_base=None, config_path="configs"):
            cfg = hydra.compose(config_name="config")
            
        model_cfg = MODEL_CONFIGS[model_name]
        
        # Override config values
        cfg.algo.name = model_cfg['algo']
        if 'backbone' in model_cfg:
            cfg.algo.backbone = model_cfg['backbone']
        cfg.block_size = model_cfg['block_size']
        cfg.eval.checkpoint_path = model_cfg['checkpoint_path']
        cfg.mode = 'sample_eval'
        cfg.wandb = None
        
        # Set sampling parameters
        cfg.sampling.num_sample_batches = 5  # Reduced for faster benchmarking
        cfg.sampling.nucleus_p = 0.9
        cfg.sampling.kv_cache = True
        cfg.model.length = 1024
        cfg.loader.eval_batch_size = 1
        
        # Set diffusion steps
        if model_cfg['algo'] == 'bd3lm':
            cfg.algo.T = 5000
        elif model_cfg['algo'] == 'ar':
            cfg.algo.T = 0
            
        return cfg
        
    def load_model(self, config, tokenizer):
        """Load the model from checkpoint"""
        if 'hf' in config.algo.backbone:
            return diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
        
        return diffusion.Diffusion.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer,
            config=config,
            strict=False,
            weights_only=False).to('cuda')
    
    def measure_generation_speed(self, model, config, num_samples=10):
        """Measure generation speed in tokens per second"""
        print(f"Measuring generation speed for {config.algo.name}...")
        
        # Warm up
        _ = model._sample(
            seqlen=config.model.length,
            batch_size_per_gpu=1,
            num_steps=config.algo.T if config.algo.T > 0 else None)
        
        torch.cuda.synchronize()
        
        speeds = []
        for i in tqdm(range(num_samples), desc="Speed test"):
            start_time = time.time()
            
            samples = model._sample(
                seqlen=config.model.length,
                batch_size_per_gpu=1,
                num_steps=config.algo.T if config.algo.T > 0 else None)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            generation_time = end_time - start_time
            tokens_generated = config.model.length
            speed = tokens_generated / generation_time
            speeds.append(speed)
            
        return np.mean(speeds), np.std(speeds)
    
    def measure_diffusion_steps(self, model, config, num_samples=10):
        """Measure average number of diffusion steps"""
        if config.algo.name == 'ar':
            return 1.0, 0.0  # AR models use 1 step per token
            
        print(f"Measuring diffusion steps for {config.algo.name}...")
        
        steps_list = []
        for i in tqdm(range(num_samples), desc="Steps measurement"):
            # Reset NFE counter
            model.metrics.gen_nfes = []
            
            _ = model._sample(
                seqlen=config.model.length,
                batch_size_per_gpu=1,
                num_steps=config.algo.T if config.algo.T > 0 else None)
            
            # Get average steps
            if len(model.metrics.gen_nfes) > 0:
                avg_steps = np.mean(model.metrics.gen_nfes)
                steps_list.append(avg_steps)
        
        return np.mean(steps_list), np.std(steps_list)
    
    def measure_perplexity(self, model, config, tokenizer):
        """Measure perplexity on validation set"""
        print(f"Measuring perplexity for {config.algo.name}...")
        
        # Create data loader for validation
        _, valid_ds = dataloader.get_dataloaders(
            config, tokenizer, skip_train=True, valid_seed=42)
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_ds, desc="PPL measurement")):
                if i >= 50:  # Limit to 50 batches for faster evaluation
                    break
                    
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to('cuda')
                
                # Compute loss
                loss = model.training_step(batch, 0)['loss']
                
                # Accumulate
                batch_size = batch['input_ids'].shape[0]
                seq_len = batch['input_ids'].shape[1]
                total_loss += loss.item() * batch_size * seq_len
                total_tokens += batch_size * seq_len
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def benchmark_model(self, model_name):
        """Benchmark a single model"""
        print(f"\n{'='*50}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*50}")
        
        # Load config and model
        config = self.load_model_config(model_name)
        tokenizer = dataloader.get_tokenizer(config)
        model = self.load_model(config, tokenizer)
        
        if hasattr(model, 'ema') and model.ema:
            model.ema.store(model._get_parameters())
            model.ema.copy_to(model._get_parameters())
        
        model.eval()
        
        results = {}
        
        # 1. Generation Speed
        try:
            speed_mean, speed_std = self.measure_generation_speed(model, config)
            results['generation_speed_mean'] = speed_mean
            results['generation_speed_std'] = speed_std
            print(f"Generation Speed: {speed_mean:.2f} ± {speed_std:.2f} tokens/sec")
        except Exception as e:
            print(f"Error measuring generation speed: {e}")
            results['generation_speed_mean'] = None
            results['generation_speed_std'] = None
        
        # 2. Diffusion Steps
        try:
            steps_mean, steps_std = self.measure_diffusion_steps(model, config)
            results['diffusion_steps_mean'] = steps_mean
            results['diffusion_steps_std'] = steps_std
            print(f"Diffusion Steps: {steps_mean:.2f} ± {steps_std:.2f}")
        except Exception as e:
            print(f"Error measuring diffusion steps: {e}")
            results['diffusion_steps_mean'] = None
            results['diffusion_steps_std'] = None
        
        # 3. Perplexity
        try:
            ppl = self.measure_perplexity(model, config, tokenizer)
            results['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")
        except Exception as e:
            print(f"Error measuring perplexity: {e}")
            results['perplexity'] = None
        
        # Add model info
        results['model_name'] = model_name
        results['block_size'] = MODEL_CONFIGS[model_name]['block_size']
        results['algo'] = MODEL_CONFIGS[model_name]['algo']
        
        self.results[model_name] = results
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def run_all_benchmarks(self):
        """Run benchmarks for all models"""
        print("Starting comprehensive benchmark...")
        
        for model_name in MODEL_CONFIGS.keys():
            try:
                self.benchmark_model(model_name)
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                continue
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to JSON and CSV files"""
        # Save as JSON
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        csv_data = []
        for model_name, results in self.results.items():
            csv_data.append(results)
        
        if csv_data:
            with open('benchmark_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        print("\nResults saved to benchmark_results.json and benchmark_results.csv")
    
    def print_summary(self):
        """Print a summary table of results"""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Model':<12} {'Block Size':<10} {'Speed (tok/s)':<15} {'Diff Steps':<12} {'PPL':<10}")
        print(f"{'-'*80}")
        
        for model_name, results in self.results.items():
            speed_str = f"{results.get('generation_speed_mean', 0):.2f}" if results.get('generation_speed_mean') else "N/A"
            steps_str = f"{results.get('diffusion_steps_mean', 0):.2f}" if results.get('diffusion_steps_mean') else "N/A"
            ppl_str = f"{results.get('perplexity', 0):.2f}" if results.get('perplexity') else "N/A"
            
            print(f"{model_name:<12} {results.get('block_size', 'N/A'):<10} {speed_str:<15} {steps_str:<12} {ppl_str:<10}")


def main():
    """Main function"""
    benchmark = InferenceBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == '__main__':
    main()
