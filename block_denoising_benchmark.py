import os
import time
import json
import hydra
import torch
import transformers
import lightning as L
import omegaconf
from tqdm import tqdm
import numpy as np

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

class StepCountingDiffusion(diffusion.Diffusion):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.block_steps = []
        self.total_blocks = 0
    
    def _semi_ar_sampler(self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
        if seqlen is None:
            seqlen = self.config.model.length
        sampling_steps = 0
        self.block_steps = []
        self.total_blocks = 0
        
        mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
        if mdlm_semi_ar:
            num_strides = self.config.model.length // 512
            num_strides -= 1

        ones = torch.ones((n_samples, 1), dtype=self.dtype, device=self.device)
        
        if self.config.sampling.kv_cache:
            self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

        for stride_num in tqdm(range(num_strides)):
            block_start_steps = sampling_steps
            
            if stride_num == 0:
                x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
                x_accum[:, 0] = self.tokenizer.bos_token_id
            else:
                if mdlm_semi_ar:
                    x = self._sample_prior(n_samples, 512).to(self.device)
                else:
                    x = self._sample_prior(n_samples, self.block_size).to(self.device)
                x_accum = torch.cat((x_accum, x), dim=1)

            end_idx = (stride_num + 1) * self.block_size
            start_idx = max(end_idx - context_size, 0)
            fwd_idx = torch.arange(start_idx, end_idx)
            if mdlm_semi_ar and stride_num > 0:
                fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

            dt = 1 / num_steps
            p_x0_cache = None
            timesteps = torch.linspace(1, 0, num_steps, device=self.device)
            t = 1
            
            for i in range(num_steps):
                if self.mask_index not in x_accum:
                    break

                if self.config.sampling.first_hitting:
                    u = np.random.rand()
                    num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
                    if num_masked > 0:
                        t *= u**(1 / num_masked)
                elif not self.config.sampling.first_hitting:
                    t = timesteps[i]

                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x_accum[:, fwd_idx],
                    t=t * ones,
                    dt=dt,
                    p_x0=p_x0_cache,)
                
                if p_x0_cache is None:
                    sampling_steps += 1
               
                x_accum[:, fwd_idx] = x_next

            block_end_steps = sampling_steps
            block_steps_count = block_end_steps - block_start_steps
            self.block_steps.append(block_steps_count)
            self.total_blocks += 1

            if x_accum.shape[1] > 256:
                stop, x_accum = self._check_stop_conds(x_accum)
                if (stop and not self.config.sampling.var_length) or (stop and x.shape[-1] == 1):
                    return None, None
                elif stop:
                    break
                    
        return x_accum, sampling_steps

def load_bd3lm_model(block_size):
    with hydra.initialize(version_base=None, config_path="configs"):
        config = hydra.compose(config_name="config", overrides=[
            f"model=small",
            f"algo=bd3lm", 
            f"data=openwebtext-split",
            f"model.length=1024",
            f"block_size={block_size}",
            f"algo.T=50",
            f"algo.backbone=hf_dit",
            f"model.attn_backend=sdpa",
            f"loader.eval_batch_size=1",
            f"sampling.num_sample_batches=1",
            f"sampling.nucleus_p=0.9",
            f"sampling.kv_cache=true",
            f"sampling.first_hitting=false",
            f"mode=sample_eval"
        ])
        
        config.eval = omegaconf.OmegaConf.create({
            'checkpoint_path': f'kuleshov-group/bd3lm-owt-block_size{block_size}',
            'perplexity_batch_size': 1,
            'disable_ema': False,
            'gen_ppl_eval_model_name_or_path': 'gpt2'
        })
        
        tokenizer = dataloader.get_tokenizer(config)
        
        model = StepCountingDiffusion(config, tokenizer=tokenizer).to('cuda')
        
        hf_model = transformers.AutoModelForMaskedLM.from_pretrained(
            config.eval.checkpoint_path,
            trust_remote_code=True
        )
        model.load_state_dict(hf_model.state_dict(), strict=False)
        del hf_model
        torch.cuda.empty_cache()
        
        model.eval()
        return model, config

def load_gpt2_model():
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model = model.to('cuda')
    return model, tokenizer

def compute_validation_ppl(model, tokenizer):
    model.eval()
    
    import datasets
    import itertools
    dataset = datasets.load_dataset(
        'openwebtext', 
        split='train',
        streaming=False,
        trust_remote_code=True
    )
    
    validation_data = dataset.select(range(len(dataset) - 100000, len(dataset)))
    
    total_nll = 0
    total_tokens = 0
    model_length = model.config.model.length
    
    with torch.no_grad():
        for i, sample in enumerate(validation_data.select(range(20))):
            text = sample['text']
            inputs = tokenizer(text, return_tensors='pt', max_length=model_length, truncation=True, padding='max_length')
            input_ids = inputs['input_ids'].to(model.device)
            
            if input_ids.shape[1] != model_length:
                continue
                
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(model.device)
            losses = model._loss(input_ids, attention_mask, sampling_eps_min=1e-3, sampling_eps_max=1)
            
            valid_mask = losses.token_mask
            valid_tokens = valid_mask.sum().item()
            
            if valid_tokens > 0:
                total_nll += (losses.nlls * valid_mask).sum().item()
                total_tokens += valid_tokens
    
    if total_tokens > 0:
        avg_nll = total_nll / total_tokens
        return torch.exp(torch.tensor(avg_nll)).item()
    return None

def measure_bd3lm_block_steps(block_size):
    model, config = load_bd3lm_model(block_size)
    tokenizer = dataloader.get_tokenizer(config)
    
    val_ppl = compute_validation_ppl(model, tokenizer)
    
    start_time = time.time()
    samples = model.restore_model_and_sample(num_steps=config.algo.T)
    end_time = time.time()
    
    generation_time = end_time - start_time
    tokens_per_sec = 1024 / generation_time
    
    avg_steps_per_block = sum(model.block_steps) / len(model.block_steps) if model.block_steps else 0
    total_steps = sum(model.block_steps)
    
    gen_ppl = model.metrics.gen_ppl.compute().item() if hasattr(model.metrics, 'gen_ppl') else None
    
    return {
        'model': f'BD3LM-{block_size}',
        'block_size': block_size,
        'total_blocks': model.total_blocks,
        'avg_denoising_steps_per_block': round(avg_steps_per_block, 2),
        'total_denoising_steps': total_steps,
        'generation_speed_tokens_per_sec': round(tokens_per_sec, 1),
        'generation_time_sec': round(generation_time, 2),
        'validation_perplexity': round(val_ppl, 2) if val_ppl else None,
        'generative_perplexity': round(gen_ppl, 2) if gen_ppl else None,
        'block_steps_detail': model.block_steps
    }

def measure_gpt2_performance():
    model, tokenizer = load_gpt2_model()
    
    start_time = time.time()
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=1024,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    generation_time = end_time - start_time
    actual_tokens = len(outputs[0])
    tokens_per_sec = actual_tokens / generation_time
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        gen_loss = outputs.loss
    
    gen_ppl = torch.exp(gen_loss).item()
    
    import datasets
    import itertools
    dataset = datasets.load_dataset(
        'openwebtext', 
        split='train',
        streaming=True,
        trust_remote_code=True
    )
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(itertools.islice(dataset, 0, 2000)):
            text = sample['text']
            inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(model.device)
            
            if input_ids.shape[1] != 1024:
                continue
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                total_tokens += attention_mask.sum().item()
    
    val_ppl = torch.exp(torch.tensor(total_loss / 20)).item() if total_tokens > 0 else None
    
    return {
        'model': 'GPT-2',
        'generation_speed_tokens_per_sec': round(tokens_per_sec, 1),
        'generation_time_sec': round(generation_time, 2),
        'actual_tokens_generated': actual_tokens,
        'validation_perplexity': round(val_ppl, 2) if val_ppl else None,
        'generative_perplexity': round(gen_ppl, 2)
    }

def run_block_denoising_benchmark():
    results = {}
    block_sizes = [4, 8, 16]
    
    for block_size in block_sizes:
        result = measure_bd3lm_block_steps(block_size)
        results[f'bd3lm_block_{block_size}'] = result
    
    gpt2_result = measure_gpt2_performance()
    results['gpt2'] = gpt2_result
    
    with open('block_denoising_steps_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    for model_key, result in results.items():
        print(f"\n{result['model']}:")
        print(f"  생성 속도: {result['generation_speed_tokens_per_sec']} tokens/sec")
        if 'avg_denoising_steps_per_block' in result:
            print(f"  평균 블록당 디노이징 스텝: {result['avg_denoising_steps_per_block']}")
            print(f"  총 디노이징 스텝: {result['total_denoising_steps']}")
            print(f"  총 블록 수: {result['total_blocks']}")
        if result.get('validation_perplexity'):
            print(f"  Validation PPL: {result['validation_perplexity']}")
        if result.get('generative_perplexity'):
            print(f"  Generative PPL: {result['generative_perplexity']}")

if __name__ == "__main__":
    L.seed_everything(42)
    run_block_denoising_benchmark()
