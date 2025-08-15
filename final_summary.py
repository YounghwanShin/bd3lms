#!/usr/bin/env python3
"""
BD3LM Inference Benchmark - Final Summary Table
"""

import json

def create_summary_table():
    """Create a clean summary table"""
    
    # Load results
    with open('bd3lm_benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    print("BD3LM vs GPT-2 INFERENCE BENCHMARK - FINAL RESULTS")
    print("=" * 75)
    print()
    
    # Header
    print(f"{'Model':<12} {'Block':<6} {'Speed':<12} {'PPL':<8} {'Steps':<8} {'Time':<8}")
    print(f"{'Type':<12} {'Size':<6} {'(tok/s)':<12} {'(↓)':<8} {'/tok':<8} {'(s)':<8}")
    print("-" * 75)
    
    # Sort results
    bd3lm_results = [r for r in results if r['model_type'] == 'bd3lm']
    ar_results = [r for r in results if r['model_type'] == 'ar_gpt2']
    
    bd3lm_results.sort(key=lambda x: x['block_size'])
    
    # Print BD3LM results
    for result in bd3lm_results:
        model = "BD3LM"
        block = str(result['block_size'])
        speed = f"{result['generation_speed_tokens_per_sec']:.1f}"
        ppl = f"{result['generative_ppl']:.1f}"
        steps = f"{result['effective_steps_per_token']:.0f}"
        time = f"{result['total_time_seconds']:.1f}"
        
        print(f"{model:<12} {block:<6} {speed:<12} {ppl:<8} {steps:<8} {time:<8}")
    
    # Print GPT-2 result
    if ar_results:
        ar = ar_results[0]
        model = "GPT-2"
        block = "1"
        speed = f"{ar['generation_speed_tokens_per_sec']:.1f}"
        ppl = f"{ar['generative_ppl']:.1f}"
        steps = "1"
        time = f"{ar['total_time_seconds']:.1f}"
        
        print(f"{model:<12} {block:<6} {speed:<12} {ppl:<8} {steps:<8} {time:<8}")
    
    print()
    print("LEGEND:")
    print("• Speed: Generation speed in tokens per second (higher = better)")
    print("• PPL: Generative perplexity (lower = better quality)")  
    print("• Steps/tok: Computational steps per token (lower = more efficient)")
    print("• Time: Total time for generating 3,072 tokens (3 samples × 1,024 tokens)")
    
    print()
    print("KEY FINDINGS:")
    print("🚀 Speed: GPT-2 is ~5x faster than BD3LM models")
    print("📝 Quality: GPT-2 has the best text quality (lowest PPL)")
    print("⚡ Efficiency: BD3LM-16 is most efficient diffusion model (62 steps/token)")
    print("🎯 Trade-off: BD3LM-4 offers best quality among diffusion models")
    
    print()
    print("RECOMMENDATIONS:")
    print("✅ Use GPT-2 for production applications requiring speed")
    print("✅ Use BD3LM-4 for research requiring diffusion-based generation")
    print("✅ Use BD3LM-16 for fastest diffusion generation with acceptable quality loss")

if __name__ == '__main__':
    create_summary_table()
