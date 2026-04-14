#!/usr/bin/env python3
"""
Module-level sensitivity profiling for API-specific routing.

For each API, computes gradient-based importance scores at the MODULE level
(q_proj, v_proj, o_proj, down_proj) within the candidate editable layers.

Output: route_maps.json — per-API top-k module route maps.

Usage:
    python profile_module_sensitivity.py --config config/deepseek-1.3b.yaml
    python profile_module_sensitivity.py --config config/deepseek-1.3b.yaml --top_k 4
"""

import os
import sys
import json
import yaml
import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG = logging.getLogger(__name__)


# ==============================================================
# Utilities
# ==============================================================

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_editable_layers(layer_config_path, num_layers=8):
    """Read selected_layer_groups.json and pick top-N editable layers."""
    with open(layer_config_path, 'r') as f:
        config = json.load(f)

    layer_frequency = config['layer_frequency']
    sorted_layers = sorted(
        layer_frequency.items(),
        key=lambda x: -x[1]['count']
    )
    editable_layers = [int(layer_id) for layer_id, _ in sorted_layers[:num_layers]]
    return editable_layers, config


def load_data(data_path):
    """Load all.json and return raw data list."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def group_by_api(data):
    """Group samples by replacement API."""
    groups = defaultdict(list)
    for d in data:
        groups[d['replacement api']].append(d)
    return dict(groups)


# ==============================================================
# Candidate Module Discovery
# ==============================================================

def build_candidate_module_names(editable_layers, module_types):
    """
    Build list of candidate module keys as 'layer_idx.module_type'.

    E.g., ['12.q_proj', '12.v_proj', '12.o_proj', '12.down_proj', '13.q_proj', ...]
    """
    candidates = []
    for layer_idx in editable_layers:
        for mod_type in module_types:
            candidates.append(f"{layer_idx}.{mod_type}")
    return candidates


def resolve_param_name(model, layer_idx, module_type):
    """
    Resolve the full parameter name for a given layer and module type.

    For deepseek-coder / Llama-like architectures:
      - Attention: model.layers.{i}.self_attn.{q_proj|k_proj|v_proj|o_proj}.weight
      - MLP:      model.layers.{i}.mlp.{gate_proj|up_proj|down_proj}.weight
    """
    attn_modules = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
    mlp_modules = {'gate_proj', 'up_proj', 'down_proj'}

    if module_type in attn_modules:
        return f"model.layers.{layer_idx}.self_attn.{module_type}.weight"
    elif module_type in mlp_modules:
        return f"model.layers.{layer_idx}.mlp.{module_type}.weight"
    else:
        raise ValueError(f"Unknown module type: {module_type}")


# ==============================================================
# Sensitivity Computation
# ==============================================================

def compute_target_only_loss(model, tokenizer, prompt_text, target_text, device):
    """
    Compute cross-entropy loss ONLY on target tokens.
    Prompt tokens are masked with -100.
    """
    mask_token = -100
    full_text = prompt_text + " " + target_text

    # Tokenize prompt to know its length
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True
    )["input_ids"]
    num_prompt_toks = int((prompt_ids != tokenizer.pad_token_id).sum())

    # Tokenize full text
    tokens = tokenizer(
        full_text, return_tensors="pt", truncation=True
    )
    tokens["labels"] = tokens["input_ids"].clone()

    # Mask prompt tokens
    num_pad_toks = int((tokens["labels"] == tokenizer.pad_token_id).sum())
    tokens["labels"][0][num_pad_toks:num_pad_toks + num_prompt_toks] = mask_token
    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    tokens = tokens.to(device)
    output = model(**tokens)
    return output.loss


def profile_single_api(model, tokenizer, api_samples, candidate_keys,
                       editable_layers, device, max_samples=None):
    """
    Compute module-level importance scores for a single API.

    For each sample:
      1. Enable gradients ONLY on candidate module weights
      2. Forward + target-only loss + backward
      3. Collect grad² for each candidate module
      4. Zero grad

    Returns:
        importance: dict[str, float] — mean-squared-gradient per module key
    """
    # Parse candidate_keys into (layer_idx, module_type) pairs
    candidate_params = {}
    for key in candidate_keys:
        layer_idx, module_type = key.split('.')
        param_name = resolve_param_name(model, int(layer_idx), module_type)
        # Find the actual parameter
        for name, param in model.named_parameters():
            if name == param_name:
                candidate_params[key] = param
                break

    # Ensure all base params have requires_grad=False
    for param in model.parameters():
        param.requires_grad = False

    # Enable grad only for candidate modules
    for key, param in candidate_params.items():
        param.requires_grad = True

    # Collect importance scores
    grad_squared_accum = defaultdict(list)
    samples = api_samples
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)

    for sample in samples:
        prompt = sample['probing input']
        target = sample['reference']

        try:
            loss = compute_target_only_loss(model, tokenizer, prompt, target, device)
            if loss is None or loss.item() == 0:
                continue
            loss.backward()

            for key, param in candidate_params.items():
                if param.grad is not None:
                    grad_sq = (param.grad ** 2).mean().item()
                    grad_squared_accum[key].append(grad_sq)

            model.zero_grad()
        except Exception as e:
            LOG.warning("Error processing sample: %s", str(e))
            model.zero_grad()
            continue

    # Disable grad for all candidate modules
    for key, param in candidate_params.items():
        param.requires_grad = False

    # Compute mean importance
    importance = {}
    for key in candidate_keys:
        scores = grad_squared_accum.get(key, [])
        importance[key] = float(np.mean(scores)) if scores else 0.0

    return importance


# ==============================================================
# Main
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile module-level sensitivity for API routing"
    )
    parser.add_argument('--config', type=str, default='config/deepseek-1.3b.yaml',
                        help='Path to config YAML')
    parser.add_argument('--data_path', type=str, default='data/deepseek-1.3b/all.json',
                        help='Path to all.json dataset')
    parser.add_argument('--layer_config', type=str,
                        default='layer_config/deepseek-1.3b/selected_layer_groups.json',
                        help='Path to selected_layer_groups.json')
    parser.add_argument('--output', type=str, default='route_maps.json',
                        help='Output path for route maps')
    parser.add_argument('--top_k', type=int, default=4,
                        help='Number of top modules to select per API')
    parser.add_argument('--max_samples_per_api', type=int, default=None,
                        help='Max samples per API for profiling (None = use all)')
    parser.add_argument('--candidate_modules', type=str, nargs='+',
                        default=['q_proj', 'v_proj', 'o_proj', 'down_proj'],
                        help='Module types to profile')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    seed_everything(42)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Select editable layers (candidate zone)
    editable_layers, layer_info = select_editable_layers(
        args.layer_config,
        num_layers=config.get('num_editable_layers', 8)
    )

    print("")
    print("=" * 60)
    print("MODULE SENSITIVITY PROFILING")
    print("=" * 60)
    print(f"  Model:             {config['model_name']}")
    print(f"  Candidate layers:  {editable_layers}")
    print(f"  Module types:      {args.candidate_modules}")
    print(f"  Top-k:             {args.top_k}")

    # Build candidate module keys
    candidate_keys = build_candidate_module_names(
        editable_layers, args.candidate_modules
    )
    print(f"  Total candidates:  {len(candidate_keys)} modules")

    # Load data and group by API
    raw_data = load_data(args.data_path)
    api_groups = group_by_api(raw_data)
    print(f"  Total samples:     {len(raw_data)}")
    print(f"  APIs:              {len(api_groups)}")
    print("=" * 60)

    # Load model
    print("")
    print("Loading model...")
    device = torch.device(f"cuda:{config.get('device', 0)}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    # Profile each API
    print("")
    print("Profiling APIs...")
    all_importance = {}
    route_maps = {}

    for api_name in tqdm(sorted(api_groups.keys()), desc="APIs"):
        samples = api_groups[api_name]
        LOG.info("Profiling %s (%d samples)", api_name, len(samples))

        importance = profile_single_api(
            model, tokenizer, samples, candidate_keys,
            editable_layers, device,
            max_samples=args.max_samples_per_api,
        )
        all_importance[api_name] = importance

        # Select top-k modules
        sorted_modules = sorted(
            importance.items(),
            key=lambda x: -x[1]
        )
        top_k_modules = [m for m, _ in sorted_modules[:args.top_k]]
        route_maps[api_name] = top_k_modules

        # Print summary
        print(f"\n  {api_name} ({len(samples)} samples):")
        for rank, (mod, score) in enumerate(sorted_modules[:args.top_k]):
            print(f"    [{rank+1}] {mod:20s} = {score:.6e}")

    # Save results
    output = {
        "config": {
            "model_name": config['model_name'],
            "editable_layers": editable_layers,
            "candidate_modules": args.candidate_modules,
            "top_k": args.top_k,
            "total_candidate_modules": len(candidate_keys),
            "num_apis": len(api_groups),
        },
        "route_maps": route_maps,
        "importance_scores": all_importance,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("")
    print("=" * 60)
    print("ROUTING SUMMARY")
    print("=" * 60)

    # Analyze module usage frequency across APIs
    module_usage = defaultdict(int)
    for api_name, modules in route_maps.items():
        for mod in modules:
            module_usage[mod] += 1

    print("\n  Module usage across APIs (how often each module is selected):")
    for mod, count in sorted(module_usage.items(), key=lambda x: -x[1]):
        print(f"    {mod:20s}: {count:3d}/{len(api_groups)} APIs ({100*count/len(api_groups):.1f}%)")

    print(f"\n  Route maps saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
