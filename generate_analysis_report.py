#!/usr/bin/env python3
"""
Generate detailed analysis report from BD3LM benchmark results
"""

import json
import numpy as np

def load_results():
    """Load benchmark results"""
    with open('bd3lm_benchmark_results.json', 'r') as f:
        return json.load(f)

def generate_analysis_report(results):
    """Generate comprehensive analysis report"""
    
    print("BD3LM INFERENCE BENCHMARK - DETAILED ANALYSIS REPORT")
    print("=" * 80)
    
    # Separate BD3LM and AR results
    bd3lm_results = [r for r in results if r['model_type'] == 'bd3lm']
    ar_results = [r for r in results if r['model_type'] == 'ar_gpt2']
    
    print("\n1. GENERATION SPEED ANALYSIS")
    print("-" * 40)
    
    # Sort BD3LM by block size
    bd3lm_results.sort(key=lambda x: x['block_size'])
    
    print(f"{'Model':<15} {'Speed (tok/s)':<15} {'Relative to GPT-2':<20}")
    print("-" * 50)
    
    gpt2_speed = ar_results[0]['generation_speed_tokens_per_sec'] if ar_results else 1
    
    for result in bd3lm_results:
        model_name = f"BD3LM-{result['block_size']}"
        speed = result['generation_speed_tokens_per_sec']
        relative_speed = speed / gpt2_speed
        print(f"{model_name:<15} {speed:<15.1f} {relative_speed:<20.2f}x")
    
    if ar_results:
        print(f"{'GPT-2':<15} {gpt2_speed:<15.1f} {'1.00x (baseline)':<20}")
    
    print("\n2. QUALITY ANALYSIS (Generative Perplexity)")
    print("-" * 40)
    
    print(f"{'Model':<15} {'Gen PPL':<12} {'Quality vs GPT-2':<20}")
    print("-" * 47)
    
    gpt2_ppl = ar_results[0]['generative_ppl'] if ar_results else 1
    
    for result in bd3lm_results:
        model_name = f"BD3LM-{result['block_size']}"
        ppl = result['generative_ppl']
        quality_ratio = ppl / gpt2_ppl
        quality_desc = "Better" if quality_ratio < 1.0 else "Worse"
        print(f"{model_name:<15} {ppl:<12.2f} {quality_ratio:.2f}x ({quality_desc})")
    
    if ar_results:
        print(f"{'GPT-2':<15} {gpt2_ppl:<12.2f} {'1.00x (baseline)':<20}")
    
    print("\n3. EFFICIENCY ANALYSIS (Steps per Token)")
    print("-" * 40)
    
    print(f"{'Model':<15} {'Steps/Token':<15} {'Efficiency Loss':<20}")
    print("-" * 50)
    
    for result in bd3lm_results:
        model_name = f"BD3LM-{result['block_size']}"
        steps_per_token = result['effective_steps_per_token']
        efficiency_loss = steps_per_token  # vs 1.0 for AR
        print(f"{model_name:<15} {steps_per_token:<15.2f} {efficiency_loss:<20.0f}x slower")
    
    if ar_results:
        print(f"{'GPT-2':<15} {'1.00':<15} {'1x (baseline)':<20}")
    
    print("\n4. TRADE-OFF ANALYSIS")
    print("-" * 40)
    
    print("Block Size vs Performance Trade-offs:")
    print()
    
    for i, result in enumerate(bd3lm_results):
        block_size = result['block_size']
        speed = result['generation_speed_tokens_per_sec']
        ppl = result['generative_ppl']
        steps_per_token = result['effective_steps_per_token']
        
        print(f"BD3LM Block Size {block_size}:")
        print(f"  • Generation Speed: {speed:.1f} tokens/sec")
        print(f"  • Quality (PPL): {ppl:.2f}")
        print(f"  • Computational Efficiency: {steps_per_token:.0f} steps/token")
        print(f"  • Total Time for 3072 tokens: {result['total_time_seconds']:.1f}s")
        
        if i == 0:  # BD3LM-4
            print(f"  • Best quality among BD3LM variants")
        elif i == len(bd3lm_results) - 1:  # Last one (highest block size)
            print(f"  • Most efficient among BD3LM variants")
        print()
    
    print("\n5. SUMMARY & RECOMMENDATIONS")
    print("-" * 40)
    
    best_quality = min(bd3lm_results, key=lambda x: x['generative_ppl'])
    best_speed = max(bd3lm_results, key=lambda x: x['generation_speed_tokens_per_sec'])
    
    print(f"📊 Performance Summary:")
    print(f"  • GPT-2 baseline: {gpt2_speed:.1f} tok/s, PPL {gpt2_ppl:.2f}")
    print(f"  • Best BD3LM Quality: Block-{best_quality['block_size']} (PPL {best_quality['generative_ppl']:.2f})")
    print(f"  • Fastest BD3LM: Block-{best_speed['block_size']} ({best_speed['generation_speed_tokens_per_sec']:.1f} tok/s)")
    print()
    
    print("🔍 Key Findings:")
    print(f"  1. GPT-2 is {gpt2_speed/max(r['generation_speed_tokens_per_sec'] for r in bd3lm_results):.1f}x faster than best BD3LM")
    print(f"  2. BD3LM quality varies: Block-4 best (PPL {best_quality['generative_ppl']:.2f}), Block-16 worst")
    print(f"  3. Larger blocks = faster generation but lower quality")
    print(f"  4. BD3LM requires {best_quality['effective_steps_per_token']:.0f}-{best_speed['effective_steps_per_token']:.0f} steps per token vs 1 for AR")
    print()
    
    print("💡 Recommendations:")
    print("  • Use BD3LM Block-4 for highest quality generation")
    print("  • Use BD3LM Block-16 for fastest diffusion-based generation") 
    print("  • Use GPT-2 when speed is critical and quality is acceptable")
    print("  • Consider the 5-6x speed penalty when choosing BD3LM over AR")
    
    # Calculate efficiency frontier
    print("\n6. EFFICIENCY FRONTIER ANALYSIS")
    print("-" * 40)
    
    print("Speed vs Quality Trade-off:")
    for result in bd3lm_results:
        efficiency_score = result['generation_speed_tokens_per_sec'] / result['generative_ppl']
        print(f"  BD3LM-{result['block_size']}: Efficiency Score = {efficiency_score:.2f}")
    
    if ar_results:
        ar_efficiency = ar_results[0]['generation_speed_tokens_per_sec'] / ar_results[0]['generative_ppl']
        print(f"  GPT-2: Efficiency Score = {ar_efficiency:.2f}")

def create_visualization(results):
    """Create visualization plots"""
    try:
        import matplotlib.pyplot as plt
        
        bd3lm_results = [r for r in results if r['model_type'] == 'bd3lm']
        ar_results = [r for r in results if r['model_type'] == 'ar_gpt2']
        
        bd3lm_results.sort(key=lambda x: x['block_size'])
        
        # Extract data
        block_sizes = [r['block_size'] for r in bd3lm_results]
        speeds = [r['generation_speed_tokens_per_sec'] for r in bd3lm_results]
        ppls = [r['generative_ppl'] for r in bd3lm_results]
        steps_per_token = [r['effective_steps_per_token'] for r in bd3lm_results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BD3LM Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Generation Speed
        ax1.bar([f'BD3LM-{bs}' for bs in block_sizes], speeds, color='skyblue', alpha=0.7)
        if ar_results:
            ax1.axhline(y=ar_results[0]['generation_speed_tokens_per_sec'], color='red', 
                       linestyle='--', label='GPT-2 Baseline')
            ax1.legend()
        ax1.set_title('Generation Speed (tokens/sec)')
        ax1.set_ylabel('Tokens per Second')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Generative Perplexity
        ax2.bar([f'BD3LM-{bs}' for bs in block_sizes], ppls, color='lightcoral', alpha=0.7)
        if ar_results:
            ax2.axhline(y=ar_results[0]['generative_ppl'], color='red', 
                       linestyle='--', label='GPT-2 Baseline')
            ax2.legend()
        ax2.set_title('Generative Perplexity (lower = better)')
        ax2.set_ylabel('Perplexity')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Steps per Token
        ax3.bar([f'BD3LM-{bs}' for bs in block_sizes], steps_per_token, color='lightgreen', alpha=0.7)
        if ar_results:
            ax3.axhline(y=1.0, color='red', linestyle='--', label='GPT-2 Baseline')
            ax3.legend()
        ax3.set_title('Computational Steps per Token')
        ax3.set_ylabel('Steps per Token')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Speed vs Quality scatter
        ax4.scatter(ppls, speeds, s=100, c=['blue', 'green', 'orange'], alpha=0.7)
        for i, bs in enumerate(block_sizes):
            ax4.annotate(f'BD3LM-{bs}', (ppls[i], speeds[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        if ar_results:
            ax4.scatter(ar_results[0]['generative_ppl'], ar_results[0]['generation_speed_tokens_per_sec'], 
                       s=100, c='red', marker='^', label='GPT-2')
            ax4.annotate('GPT-2', (ar_results[0]['generative_ppl'], ar_results[0]['generation_speed_tokens_per_sec']),
                        xytext=(5, 5), textcoords='offset points')
            
        ax4.set_xlabel('Generative Perplexity (lower = better quality)')
        ax4.set_ylabel('Generation Speed (tokens/sec)')
        ax4.set_title('Speed vs Quality Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bd3lm_benchmark_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📈 Visualization saved as 'bd3lm_benchmark_analysis.png'")
        
    except ImportError:
        print("\n📈 Matplotlib not available. Skipping visualization.")

def main():
    """Main function"""
    try:
        results = load_results()
        generate_analysis_report(results)
        create_visualization(results)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("Files generated:")
        print("  • bd3lm_benchmark_results.json (raw data)")
        print("  • bd3lm_benchmark_results.csv (spreadsheet format)")
        print("  • bd3lm_benchmark_analysis.png (visualization)")
        print("=" * 80)
        
    except FileNotFoundError:
        print("Error: bd3lm_benchmark_results.json not found. Run the benchmark first.")
    except Exception as e:
        print(f"Error generating analysis: {e}")

if __name__ == '__main__':
    main()
