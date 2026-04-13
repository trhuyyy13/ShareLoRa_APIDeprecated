#!/usr/bin/env python3
"""
SharedLoRA: Unified single-model LoRA fine-tuning for API migration.

Pipeline:
  1. Read selected_layer_groups.json → pick top-N editable layers by frequency
  2. Load base model, freeze everything
  3. Attach ONE shared LoRA adapter only on editable layers
  4. Train on full deprecated→updated dataset (loss on target tokens only)
  5. Evaluate all cases with standard EDAPI metrics:
     efficacy, generalization, portability, specificity
  6. Save adapter + results

Usage:
  python run.py --config config/deepseek-1.3b.yaml \
                --data_path data/deepseek-1.3b/all.json \
                --layer_config layer_config/deepseek-1.3b/selected_layer_groups.json

  python run.py --config config/deepseek-1.3b.yaml --dry_run
"""

import os
import sys
import json
import yaml
import argparse
import logging
import random
from copy import deepcopy
from pathlib import Path
from time import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# ── Add project root to path for evaluate imports ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from evaluate.edapi_evaluate import compute_edit_quality
from evaluate.evaluate_utils import MATCH_METRICS

LOG = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


# ═══════════════════════════════════════════════════════════════
# 1. Layer Selection
# ═══════════════════════════════════════════════════════════════

def select_editable_layers(layer_config_path: str, num_layers: int = 8):
    """
    Read selected_layer_groups.json and pick top-N editable layers.
    
    Returns:
        editable_layers: list[int] - layers to attach LoRA
        common_layers: list[int] - layers that are frozen (general knowledge)
    """
    with open(layer_config_path, 'r') as f:
        config = json.load(f)
    
    common_layers = config['common_layers']
    layer_frequency = config['layer_frequency']
    
    # Sort by count descending, take top-N
    sorted_layers = sorted(
        layer_frequency.items(),
        key=lambda x: -x[1]['count']
    )
    editable_layers = [int(layer_id) for layer_id, _ in sorted_layers[:num_layers]]
    
    return editable_layers, common_layers, config


# ═══════════════════════════════════════════════════════════════
# 2. Data Loading & Preparation
# ═══════════════════════════════════════════════════════════════

def load_data(data_path: str):
    """Load all.json and return raw data list."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    LOG.info(f"Loaded {len(data)} samples from {data_path}")
    return data


def prepare_requests(data, model_name):
    """
    Convert raw data to request format for training and evaluation.
    Adapted from EDAPI's prepare_requests function.
    """
    requests = []
    for d in data:
        req = {
            'case_id': d['case-id'],
            'prompt': d['probing input'],
            'target_new': d['reference'],
            'rephrase_prompt': d['rephrase'],
            'rephrase_target_new': d['rephrase_reference'],
            'reference_dict': d['reference dict'],
            'alias_dict': d['alias dict'],
            'rephrase_reference_dict': d['reference dict'] | d['rephrase_reference_dict'],
            'new_api': [[d['replacement api']]],
            'specificity': {
                'prompts': [item['probing input'] for item in d['Specificity-SimilarContext']],
                'ground_truth': [item['prediction'] for item in d['Specificity-SimilarContext']],
                'pred-api': [item['pred-api'] for item in d['Specificity-SimilarContext']],
            },
            'portability': d['portability'],
            'target_api': d['replacement api'],
            'probing_predictions': d['probing predictions'][0][0],
            'api_predicted': d['probing predictions'][0][1],
            'deprecated_api': d['deprecated api'],
            'expected_call': d['expected call'],
        }
        requests.append(req)
    return requests


# ═══════════════════════════════════════════════════════════════
# 3. Model Setup
# ═══════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str, device: int = 0):
    """Load model and tokenizer, matching EDAPI's init_deepseek1b pattern."""
    LOG.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
        local_files_only=True,
    )
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.to(f'cuda:{device}')
    LOG.info(f"Model loaded on cuda:{device}")
    return model, tokenizer


def setup_shared_lora(model, editable_layers, config):
    """
    Freeze entire base model, then attach a SINGLE shared LoRA 
    adapter only on the editable layers.
    """
    # Freeze everything
    model.requires_grad_(False)
    
    # Prepare model for LoRA
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Create single shared LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        layers_to_transform=editable_layers,
        target_modules=config['target_modules'],
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()
    
    return peft_model


# ═══════════════════════════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════════════════════════

def train(model, tokenizer, requests, config):
    """
    Train the shared LoRA on full dataset.
    Loss is computed only on target tokens (prompt tokens masked with -100).
    Includes early stopping based on training loss plateau.
    """
    device = torch.device(f'cuda:{config["device"]}')
    
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    patience = config['patience']
    
    total_steps_per_epoch = (len(texts) + batch_size - 1) // batch_size
    
    LOG.info("=" * 60)
    LOG.info("TRAINING CONFIGURATION")
    LOG.info(f"  Total samples:     {len(texts)}")
    LOG.info(f"  Epochs:            {num_epochs}")
    LOG.info(f"  Batch size:        {batch_size}")
    LOG.info(f"  Steps per epoch:   {total_steps_per_epoch}")
    LOG.info(f"  Learning rate:     {config['lr']}")
    LOG.info(f"  Weight decay:      {config['weight_decay']}")
    LOG.info(f"  Patience:          {patience}")
    LOG.info(f"  LoRA rank:         {config['rank']}")
    LOG.info(f"  LoRA alpha:        {config['lora_alpha']}")
    LOG.info("=" * 60)
    
    best_loss = float('inf')
    patience_counter = 0
    training_log = []
    
    for epoch in range(num_epochs):
        epoch_loss = AverageMeter()
        model.train()
        
        # Shuffle data each epoch
        indices = list(range(len(texts)))
        random.shuffle(indices)
        shuffled_texts = [texts[i] for i in indices]
        shuffled_targets = [targets[i] for i in indices]
        
        pbar = tqdm(
            total=total_steps_per_epoch,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            unit="step",
            bar_format="{l_bar}{bar:30}{r_bar}"
        )
        
        for txt_batch, tgt_batch in zip(
            chunks(shuffled_texts, batch_size),
            chunks(shuffled_targets, batch_size)
        ):
            mask_token = -100
            opt.zero_grad()
            
            # Build input: prompt + target
            full_prompt = [f"{p} {t}" for p, t in zip(txt_batch, tgt_batch)]
            prompt_ids = tokenizer(
                list(txt_batch), return_tensors="pt",
                padding=True, truncation=True
            )["input_ids"]
            num_prompt_toks = [
                int((ids != tokenizer.pad_token_id).sum()) for ids in prompt_ids
            ]
            
            tokens = tokenizer(
                full_prompt, return_tensors="pt",
                padding=True, truncation=True
            )
            bs = tokens["input_ids"].shape[0]
            tokens["labels"] = tokens["input_ids"].clone()
            
            # Mask prompt tokens and pad tokens → only compute loss on target
            num_pad_toks = [
                int((ids == tokenizer.pad_token_id).sum()) for ids in tokens["labels"]
            ]
            for j in range(len(txt_batch)):
                tokens["labels"][j][
                    num_pad_toks[j]:num_pad_toks[j]+num_prompt_toks[j]
                ] = mask_token
            tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
            tokens = tokens.to(device)
            
            pred = model(**tokens)
            loss = pred.loss
            
            if loss is not None:
                loss.backward()
                opt.step()
                epoch_loss.update(loss.item(), n=bs)
            
            pbar.set_postfix({'loss': f'{epoch_loss.avg:.4f}'})
            pbar.update(1)
        
        pbar.close()
        
        epoch_info = {
            'epoch': epoch + 1,
            'avg_loss': round(epoch_loss.avg, 6),
        }
        training_log.append(epoch_info)
        
        print(f"  ✓ Epoch {epoch+1}/{num_epochs} — avg_loss: {epoch_loss.avg:.4f}")
        LOG.info(f"Epoch {epoch+1}/{num_epochs} — avg_loss: {epoch_loss.avg:.4f}")
        
        # Early stopping check
        if epoch_loss.avg < best_loss:
            best_loss = epoch_loss.avg
            patience_counter = 0
        else:
            patience_counter += 1
            LOG.info(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                LOG.info(f"  ⏹ Early stopping at epoch {epoch+1}")
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break
    
    return model, training_log


# ═══════════════════════════════════════════════════════════════
# 5. Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_model(model, tokenizer, requests, config, output_dir):
    """
    Evaluate the edited model on all test cases using EDAPI's 4 metrics:
    efficacy, generalization, portability, specificity.
    """
    model.eval()
    
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve portability links
    case_lookup = {r['case_id']: r for r in requests}
    
    all_metrics = []
    
    for index, request in tqdm(enumerate(requests), total=len(requests), desc="Evaluating"):
        if request["case_id"] in ['']:
            continue
        request = request.copy()
        
        # Resolve portability
        if request["portability"] != "":
            port_id = request["portability"]
            if port_id in case_lookup:
                request["portability"] = case_lookup[port_id]
            else:
                LOG.warning(f"Case {request['case_id']}: portability target '{port_id}' not found")
                request["portability"] = ""
        
        start = time()
        torch.cuda.reset_peak_memory_stats(f"cuda:{config['device']}")
        
        try:
            metric_result = compute_edit_quality(
                model, tokenizer, request,
                test_generation=False,
            )
            all_metrics.append({
                'case_id': request['case_id'],
                'time': round(time() - start, 3),
                'max_memory': round(torch.cuda.max_memory_allocated(f"cuda:{config['device']}") / 1024**2, 2),
                'post': metric_result,
            })
        except Exception as e:
            LOG.error(f"Case {request['case_id']} error: {e}")
            continue
        
        # Periodic save
        if (index + 1) % 20 == 0:
            with open(results_dir / "run_000.json", 'w') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    
    # Final save
    with open(results_dir / "run_000.json", 'w') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    
    # Compute mean metrics
    mean_metrics = {}
    for metric_name in ['efficacy', 'generalization', 'portability', 'specificity']:
        mean_metrics[metric_name] = {}
        for match_metric in MATCH_METRICS:
            vals = [item['post'][metric_name][match_metric] for item in all_metrics
                    if metric_name in item['post']]
            if vals:
                mean_metrics[metric_name][match_metric] = (
                    round(np.mean(vals) * 100, 2),
                    round(np.std(vals) * 100, 2),
                )
            else:
                mean_metrics[metric_name][match_metric] = (0, 0)
    
    mean_metrics["time"] = (
        round(np.mean([m["time"] for m in all_metrics]), 3),
        round(np.std([m["time"] for m in all_metrics]), 3),
    )
    mean_metrics["max_memory"] = (
        round(np.mean([m["max_memory"] for m in all_metrics]), 3),
        round(np.std([m["max_memory"] for m in all_metrics]), 3),
    )
    
    with open(results_dir / "mean_run_000.json", 'w') as f:
        json.dump(mean_metrics, f, ensure_ascii=False, indent=2)
    
    return all_metrics, mean_metrics


# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="SharedLoRA: Unified LoRA for API Migration")
    parser.add_argument('--config', type=str, default='config/deepseek-1.3b.yaml',
                        help='Path to config YAML')
    parser.add_argument('--data_path', type=str, default='data/deepseek-1.3b/all.json',
                        help='Path to all.json dataset')
    parser.add_argument('--layer_config', type=str,
                        default='layer_config/deepseek-1.3b/selected_layer_groups.json',
                        help='Path to selected_layer_groups.json')
    parser.add_argument('--output_dir', type=str, default='results/deepseek-1.3b',
                        help='Output directory for results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only show config and layer selection, skip training')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    time_now = datetime.now().strftime("%m-%d-%H-%M")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    
    seed_everything(42)
    
    # ── Load config ──
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Ensure numeric fields are proper types (YAML may parse scientific notation as string)
    for key in ['lr', 'weight_decay', 'lora_alpha', 'lora_dropout']:
        if key in config and isinstance(config[key], str):
            config[key] = float(config[key])
    LOG.info(f"Config: {config}")
    
    # ── Step 1: Select editable layers ──
    editable_layers, common_layers, layer_info = select_editable_layers(
        args.layer_config,
        num_layers=config.get('num_editable_layers', 8)
    )
    
    print("\n" + "=" * 60)
    print("LAYER SELECTION")
    print("=" * 60)
    print(f"  Common layers (FROZEN):   {sorted(common_layers)}")
    print(f"  Editable layers (LoRA):   {editable_layers}")
    print(f"  Total model layers:       24 (deepseek-1.3b)")
    print(f"  Frozen: {24 - len(editable_layers)} layers | Editable: {len(editable_layers)} layers")
    
    # Show frequency info
    print(f"\n  Layer frequency (in api_specific_layers):")
    for layer_id in editable_layers:
        freq = layer_info['layer_frequency'].get(str(layer_id), {})
        print(f"    Layer {layer_id:>2}: count={freq.get('count', 'N/A')}, "
              f"ratio={freq.get('ratio', 'N/A')}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n✅ Dry run complete. Config and layers verified.")
        return
    
    # ── Step 2: Load model ──
    model, tokenizer = load_model_and_tokenizer(
        config['model_name'],
        device=config.get('device', 0)
    )
    
    # ── Step 3: Setup shared LoRA ──
    print("\n" + "=" * 60)
    print("SETTING UP SHARED LoRA")
    print("=" * 60)
    model = setup_shared_lora(model, editable_layers, config)
    
    # ── Step 4: Load data ──
    raw_data = load_data(args.data_path)
    requests = prepare_requests(raw_data, config['model_name'])
    LOG.info(f"Prepared {len(requests)} requests for training")
    
    # Count APIs
    apis = set(r['target_api'] for r in requests)
    print(f"\n  Dataset: {len(requests)} samples across {len(apis)} APIs")
    
    # ── Step 5: Train ──
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    start_train = time()
    model, training_log = train(model, tokenizer, requests, config)
    train_time = time() - start_train
    print(f"\n  Training completed in {train_time:.1f}s")
    
    # Save training log
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_log.json", 'w') as f:
        json.dump({
            'config': config,
            'editable_layers': editable_layers,
            'common_layers': common_layers,
            'train_time': round(train_time, 2),
            'epochs': training_log,
        }, f, indent=2)
    
    # Save LoRA adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  ✓ Adapter saved to {adapter_dir}")
    
    # ── Step 6: Evaluate ──
    print("\n" + "=" * 60)
    print("EVALUATION (4 EDAPI Metrics)")
    print("=" * 60)
    all_metrics, mean_metrics = evaluate_model(
        model, tokenizer, requests, config, args.output_dir
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for metric_name in ['efficacy', 'generalization', 'portability', 'specificity']:
        print(f"\n  {metric_name.upper()}:")
        for match_metric in MATCH_METRICS:
            mean_val, std_val = mean_metrics[metric_name][match_metric]
            print(f"    {match_metric:<20}: {mean_val:.2f}% ± {std_val:.2f}%")
    
    print(f"\n  TIME:   {mean_metrics['time'][0]:.3f}s ± {mean_metrics['time'][1]:.3f}s")
    print(f"  MEMORY: {mean_metrics['max_memory'][0]:.1f}MB ± {mean_metrics['max_memory'][1]:.1f}MB")
    print("=" * 60)
    print(f"\n✅ All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
