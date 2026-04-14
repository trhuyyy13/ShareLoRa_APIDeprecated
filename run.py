#!/usr/bin/env python3
"""
Module-Routed Shared LoRA: Unified LoRA with API-specific module routing.

Pipeline:
  1. Load route_maps.json (from profile_module_sensitivity.py)
  2. Attach shared LoRA to ALL candidate modules in editable layers
  3. Train with API-grouped batching: each API only activates its routed modules
  4. Evaluate with route-aware inference

Usage:
    python run.py --config config/deepseek-1.3b.yaml
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
from collections import defaultdict
from pathlib import Path
from time import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# -- Add project root to path for evaluate imports --
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_metrics.edapi_evaluate import compute_edit_quality
from eval_metrics.evaluate_utils import MATCH_METRICS

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


# ==============================================================
# 1. Layer + Route Map Loading
# ==============================================================

def select_editable_layers(layer_config_path, num_layers=8):
    """
    Read selected_layer_groups.json and pick top-N editable layers.

    Returns:
        editable_layers: list of int - layers to attach LoRA
        common_layers: list of int - layers that are frozen (general knowledge)
        config: dict - full layer config
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


def load_route_maps(route_maps_path):
    """
    Load route_maps.json produced by profile_module_sensitivity.py.

    Returns:
        route_maps: dict[str, list[str]] — api_name -> list of module keys
        route_config: dict — metadata about the routing config
    """
    with open(route_maps_path, 'r') as f:
        data = json.load(f)

    route_maps = data['route_maps']
    route_config = data['config']

    LOG.info("Loaded route maps for %d APIs (top_k=%d)",
             len(route_maps), route_config['top_k'])

    return route_maps, route_config


# ==============================================================
# 2. Data Loading & Preparation
# ==============================================================

def load_data(data_path):
    """Load all.json and return raw data list."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    LOG.info("Loaded %d samples from %s", len(data), data_path)
    return data


def prepare_requests(data, model_name):
    """
    Convert raw data to request format for training and evaluation.
    Adapted from EDAPI prepare_requests function.
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
            'rephrase_reference_dict': {**d['reference dict'], **d['rephrase_reference_dict']},
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


def group_requests_by_api(requests):
    """Group prepared requests by target_api for route-aware training."""
    groups = defaultdict(list)
    for r in requests:
        groups[r['target_api']].append(r)
    return dict(groups)


# ==============================================================
# 3. Model Setup
# ==============================================================

def load_model_and_tokenizer(model_name, device=0):
    """Load model and tokenizer (same as EDAPI init_deepseek1b)."""
    LOG.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.to("cuda:%d" % device)
    LOG.info("Model loaded on cuda:%d", device)
    return model, tokenizer


def setup_shared_lora(model, editable_layers, config):
    """
    Freeze entire base model, then attach a SINGLE shared LoRA
    adapter on ALL candidate modules within the editable layers.

    Unlike the old approach (only q_proj, v_proj), we now attach LoRA
    to all routing candidate modules: q_proj, v_proj, o_proj, down_proj
    (and optionally gate_proj, up_proj).
    """
    # Freeze everything
    model.requires_grad_(False)

    # Prepare model for LoRA
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Create single shared LoRA config targeting ALL candidate modules
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


# ==============================================================
# 4. Module Routing (Mask/Gate)
# ==============================================================

def build_lora_module_map(peft_model):
    """
    Build a map from 'layer_idx.module_type' -> LoRA module reference.

    This allows us to selectively enable/disable specific LoRA modules
    during training and inference.

    Returns:
        module_map: dict[str, Module] — key -> LoRA module
        original_scaling: dict[str, float] — key -> original scaling value
    """
    module_map = {}
    original_scaling = {}

    for name, module in peft_model.named_modules():
        # Check if this is a LoRA-wrapped module (has lora_A attribute)
        if hasattr(module, 'lora_A') and hasattr(module, 'scaling'):
            # Parse the module name to extract layer index and module type
            # Name format: base_model.model.model.layers.{i}.self_attn.{type}
            #           or: base_model.model.model.layers.{i}.mlp.{type}
            parts = name.split('.')
            if 'layers' in parts:
                layer_idx_pos = parts.index('layers') + 1
                if layer_idx_pos < len(parts):
                    layer_idx = int(parts[layer_idx_pos])
                    module_type = parts[-1]  # q_proj, v_proj, o_proj, down_proj, etc.
                    key = f"{layer_idx}.{module_type}"
                    module_map[key] = module

                    # Store original scaling for restoration
                    adapter_name = list(module.scaling.keys())[0]  # Usually 'default'
                    original_scaling[key] = module.scaling[adapter_name]

    LOG.info("Built LoRA module map with %d modules", len(module_map))
    return module_map, original_scaling


def activate_route(module_map, original_scaling, route_modules, training=True):
    """
    Activate ONLY the modules in route_modules; mask all others.

    For active modules: restore original scaling, enable grad
    For inactive modules: set scaling=0, disable grad

    Args:
        module_map: dict from build_lora_module_map
        original_scaling: dict of original scaling values
        route_modules: list[str] — module keys to activate (e.g. ['12.q_proj', '13.v_proj'])
        training: bool — if True, also controls requires_grad
    """
    route_set = set(route_modules)
    adapter_name = None

    for key, module in module_map.items():
        if adapter_name is None:
            adapter_name = list(module.scaling.keys())[0]

        if key in route_set:
            # ACTIVATE: restore original scaling
            module.scaling[adapter_name] = original_scaling[key]
            if training:
                for param in module.lora_A[adapter_name].parameters():
                    param.requires_grad = True
                for param in module.lora_B[adapter_name].parameters():
                    param.requires_grad = True
        else:
            # DEACTIVATE: zero scaling, disable grad
            module.scaling[adapter_name] = 0.0
            if training:
                for param in module.lora_A[adapter_name].parameters():
                    param.requires_grad = False
                for param in module.lora_B[adapter_name].parameters():
                    param.requires_grad = False


def activate_all_modules(module_map, original_scaling):
    """Re-enable all LoRA modules (for full evaluation or reset)."""
    adapter_name = None
    for key, module in module_map.items():
        if adapter_name is None:
            adapter_name = list(module.scaling.keys())[0]
        module.scaling[adapter_name] = original_scaling[key]
        for param in module.lora_A[adapter_name].parameters():
            param.requires_grad = True
        for param in module.lora_B[adapter_name].parameters():
            param.requires_grad = True


# ==============================================================
# 5. Training
# ==============================================================

def compute_l2_penalty(module_map, route_modules):
    """
    Compute L2 regularization on active LoRA adapter weights.
    Only penalizes the routed (active) modules.
    """
    route_set = set(route_modules)
    penalty = 0.0
    count = 0
    adapter_name = None

    for key, module in module_map.items():
        if key not in route_set:
            continue
        if adapter_name is None:
            adapter_name = list(module.lora_A.keys())[0]

        for param in module.lora_A[adapter_name].parameters():
            penalty += (param ** 2).sum()
            count += param.numel()
        for param in module.lora_B[adapter_name].parameters():
            penalty += (param ** 2).sum()
            count += param.numel()

    return penalty / max(count, 1)


def train(model, tokenizer, requests, route_maps, module_map,
          original_scaling, config):
    """
    Train the shared LoRA with API-grouped batching and module routing.

    For each epoch:
      - Shuffle API order
      - For each API:
          1. Activate only routed modules (mask others)
          2. Train on API's samples (mini-batches)
          3. Deactivate
    """
    device = torch.device("cuda:%d" % config["device"])

    # Group requests by API
    api_groups = group_requests_by_api(requests)
    api_names = list(api_groups.keys())

    # Optimizer over ALL LoRA parameters (active or not)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    patience = config['patience']
    lambda_reg = config.get('lambda_reg', 1e-4)

    total_samples = sum(len(v) for v in api_groups.values())
    total_steps_per_epoch = sum(
        (len(v) + batch_size - 1) // batch_size for v in api_groups.values()
    )

    LOG.info("=" * 60)
    LOG.info("TRAINING CONFIGURATION (Module-Routed)")
    LOG.info("  Total samples:     %d", total_samples)
    LOG.info("  APIs:              %d", len(api_names))
    LOG.info("  Epochs:            %d", num_epochs)
    LOG.info("  Batch size:        %d", batch_size)
    LOG.info("  Steps per epoch:   ~%d", total_steps_per_epoch)
    LOG.info("  Learning rate:     %s", config['lr'])
    LOG.info("  Weight decay:      %s", config['weight_decay'])
    LOG.info("  Lambda reg:        %s", lambda_reg)
    LOG.info("  LoRA rank:         %d", config['rank'])
    LOG.info("=" * 60)

    best_loss = float('inf')
    patience_counter = 0
    training_log = []

    for epoch in range(num_epochs):
        epoch_loss = AverageMeter()
        model.train()

        # Shuffle API order each epoch
        random.shuffle(api_names)

        pbar = tqdm(
            total=total_steps_per_epoch,
            desc="Epoch %d/%d" % (epoch + 1, num_epochs),
            unit="step",
            bar_format="{l_bar}{bar:30}{r_bar}"
        )

        for api_name in api_names:
            api_requests = api_groups[api_name]
            route_modules = route_maps.get(api_name, [])

            if not route_modules:
                LOG.warning("No route map for API: %s, skipping", api_name)
                continue

            # Activate only this API's routed modules
            activate_route(module_map, original_scaling, route_modules, training=True)

            # Shuffle samples for this API
            random.shuffle(api_requests)
            texts = [r["prompt"] for r in api_requests]
            targets = [r["target_new"] for r in api_requests]

            for txt_batch, tgt_batch in zip(
                chunks(texts, batch_size),
                chunks(targets, batch_size)
            ):
                mask_token = -100
                opt.zero_grad()

                # Build input: prompt + target
                full_prompt = [p + " " + t for p, t in zip(txt_batch, tgt_batch)]
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

                # Mask prompt tokens and pad tokens -> only compute loss on target
                num_pad_toks = [
                    int((ids == tokenizer.pad_token_id).sum()) for ids in tokens["labels"]
                ]
                for j in range(len(txt_batch)):
                    tokens["labels"][j][
                        num_pad_toks[j]:num_pad_toks[j] + num_prompt_toks[j]
                    ] = mask_token
                tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
                tokens = tokens.to(device)

                pred = model(**tokens)
                loss = pred.loss

                if loss is not None:
                    # Add L2 regularization on active adapter weights
                    if lambda_reg > 0:
                        l2_pen = compute_l2_penalty(module_map, route_modules)
                        loss = loss + lambda_reg * l2_pen

                    loss.backward()
                    opt.step()
                    epoch_loss.update(loss.item(), n=bs)

                pbar.set_postfix({
                    'loss': '%.4f' % epoch_loss.avg,
                    'api': api_name.split('.')[-1][:15]
                })
                pbar.update(1)

        pbar.close()

        # Restore all modules after epoch
        activate_all_modules(module_map, original_scaling)

        epoch_info = {
            'epoch': epoch + 1,
            'avg_loss': round(epoch_loss.avg, 6),
        }
        training_log.append(epoch_info)

        print("  Epoch %d/%d -- avg_loss: %.4f" % (epoch + 1, num_epochs, epoch_loss.avg))
        LOG.info("Epoch %d/%d -- avg_loss: %.4f", epoch + 1, num_epochs, epoch_loss.avg)

        # Early stopping check
        if epoch_loss.avg < best_loss:
            best_loss = epoch_loss.avg
            patience_counter = 0
        else:
            patience_counter += 1
            LOG.info("  No improvement. Patience: %d/%d", patience_counter, patience)
            if patience_counter >= patience:
                LOG.info("  Early stopping at epoch %d", epoch + 1)
                print("  Early stopping at epoch %d" % (epoch + 1))
                break

    return model, training_log


# ==============================================================
# 6. Evaluation
# ==============================================================

def evaluate_model(model, tokenizer, requests, route_maps, module_map,
                   original_scaling, config, output_dir):
    """
    Evaluate the edited model with route-aware inference.
    For each test case, activate the API's routed modules before generation.
    """
    model.eval()

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve portability links
    case_lookup = {}
    for r in requests:
        case_lookup[r['case_id']] = r

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
                LOG.warning("Case %s: portability target '%s' not found", request['case_id'], port_id)
                request["portability"] = ""

        # Activate route for this API
        api_name = request['target_api']
        route_modules = route_maps.get(api_name, [])
        if route_modules:
            activate_route(module_map, original_scaling, route_modules, training=False)
        else:
            # Fallback: activate all modules if no route map
            activate_all_modules(module_map, original_scaling)

        start = time()
        torch.cuda.reset_peak_memory_stats("cuda:%d" % config['device'])

        try:
            metric_result = compute_edit_quality(
                model, tokenizer, request,
                test_generation=False,
            )
            mem_mb = torch.cuda.max_memory_allocated("cuda:%d" % config['device']) / (1024 ** 2)
            all_metrics.append({
                'case_id': request['case_id'],
                'target_api': api_name,
                'route_modules': route_modules,
                'time': round(time() - start, 3),
                'max_memory': round(mem_mb, 2),
                'post': metric_result,
            })
        except Exception as e:
            LOG.error("Case %s error: %s", request['case_id'], str(e))
            continue

        # Periodic save
        if (index + 1) % 20 == 0:
            with open(results_dir / "run_000.json", 'w') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    # Restore all modules
    activate_all_modules(module_map, original_scaling)

    # Final save
    with open(results_dir / "run_000.json", 'w') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    # Compute mean metrics
    mean_metrics = {}
    for metric_name in ['efficacy', 'generalization', 'portability', 'specificity']:
        mean_metrics[metric_name] = {}
        for match_metric in MATCH_METRICS:
            vals = [item['post'][metric_name][match_metric] for item in all_metrics
                    if metric_name in item.get('post', {})]
            if vals:
                mean_metrics[metric_name][match_metric] = (
                    round(float(np.mean(vals)) * 100, 2),
                    round(float(np.std(vals)) * 100, 2),
                )
            else:
                mean_metrics[metric_name][match_metric] = (0, 0)

    mean_metrics["time"] = (
        round(float(np.mean([m["time"] for m in all_metrics])), 3),
        round(float(np.std([m["time"] for m in all_metrics])), 3),
    )
    mean_metrics["max_memory"] = (
        round(float(np.mean([m["max_memory"] for m in all_metrics])), 3),
        round(float(np.std([m["max_memory"] for m in all_metrics])), 3),
    )

    with open(results_dir / "mean_run_000.json", 'w') as f:
        json.dump(mean_metrics, f, ensure_ascii=False, indent=2)

    return all_metrics, mean_metrics


# ==============================================================
# 7. Main
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Module-Routed Shared LoRA for API Migration"
    )
    parser.add_argument('--config', type=str, default='config/deepseek-1.3b.yaml',
                        help='Path to config YAML')
    parser.add_argument('--data_path', type=str, default='data/deepseek-1.3b/all.json',
                        help='Path to all.json dataset')
    parser.add_argument('--layer_config', type=str,
                        default='layer_config/deepseek-1.3b/selected_layer_groups.json',
                        help='Path to selected_layer_groups.json')
    parser.add_argument('--route_maps', type=str, default='route_maps.json',
                        help='Path to route_maps.json (from profile_module_sensitivity.py)')
    parser.add_argument('--output_dir', type=str, default='results/deepseek-1.3b',
                        help='Output directory for results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only show config and routing info, skip training')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    seed_everything(42)

    # -- Load config --
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Ensure numeric fields are proper types
    for key in ['lr', 'weight_decay', 'lora_alpha', 'lora_dropout', 'lambda_reg']:
        if key in config and isinstance(config[key], str):
            config[key] = float(config[key])
    LOG.info("Config: %s", str(config))

    # -- Step 1: Select editable layers --
    editable_layers, common_layers, layer_info = select_editable_layers(
        args.layer_config,
        num_layers=config.get('num_editable_layers', 8)
    )

    print("")
    print("=" * 60)
    print("LAYER SELECTION")
    print("=" * 60)
    print("  Common layers (FROZEN):   %s" % sorted(common_layers))
    print("  Editable layers (LoRA):   %s" % editable_layers)
    print("  Total model layers:       24 (deepseek-1.3b)")
    print("  Frozen: %d layers | Editable: %d layers" % (24 - len(editable_layers), len(editable_layers)))

    # Show frequency info
    print("")
    print("  Layer frequency (in api_specific_layers):")
    for layer_id in editable_layers:
        freq = layer_info['layer_frequency'].get(str(layer_id), {})
        print("    Layer %2d: count=%s, ratio=%s" % (
            layer_id, freq.get('count', 'N/A'), freq.get('ratio', 'N/A')))

    # -- Step 2: Load route maps --
    print("")
    print("=" * 60)
    print("ROUTE MAPS")
    print("=" * 60)

    if not os.path.exists(args.route_maps):
        print("  ERROR: route_maps.json not found at: %s" % args.route_maps)
        print("  Run profile_module_sensitivity.py first:")
        print("    python profile_module_sensitivity.py --config %s" % args.config)
        sys.exit(1)

    route_maps, route_config = load_route_maps(args.route_maps)

    print("  Route config:")
    print("    Candidate layers:  %s" % route_config['editable_layers'])
    print("    Candidate modules: %s" % route_config['candidate_modules'])
    print("    Top-k:             %d" % route_config['top_k'])
    print("    Total candidates:  %d" % route_config['total_candidate_modules'])

    # Show a few example route maps
    print("")
    print("  Example route maps (first 5 APIs):")
    for i, (api_name, modules) in enumerate(sorted(route_maps.items())):
        if i >= 5:
            print("    ... (%d more APIs)" % (len(route_maps) - 5))
            break
        print("    %s:" % api_name)
        print("      -> %s" % modules)

    # Compute module usage stats
    module_usage = defaultdict(int)
    for api_name, modules in route_maps.items():
        for mod in modules:
            module_usage[mod] += 1

    print("")
    print("  Module usage (top 10):")
    for mod, count in sorted(module_usage.items(), key=lambda x: -x[1])[:10]:
        print("    %s: %d/%d APIs (%.1f%%)" % (
            mod, count, len(route_maps), 100 * count / len(route_maps)))

    print("=" * 60)

    if args.dry_run:
        print("")
        print("Dry run complete. Config, layers, and routes verified.")
        return

    # -- Step 3: Load model --
    model, tokenizer = load_model_and_tokenizer(
        config['model_name'],
        device=config.get('device', 0)
    )

    # -- Step 4: Setup shared LoRA (on ALL candidate modules) --
    print("")
    print("=" * 60)
    print("SETTING UP SHARED LoRA (Module-Routed)")
    print("=" * 60)
    model = setup_shared_lora(model, editable_layers, config)

    # Build the module map for routing
    module_map, original_scaling = build_lora_module_map(model)
    print("  LoRA modules found: %d" % len(module_map))
    print("  Module keys: %s" % sorted(module_map.keys()))

    # -- Step 5: Load data --
    raw_data = load_data(args.data_path)
    requests = prepare_requests(raw_data, config['model_name'])
    LOG.info("Prepared %d requests for training", len(requests))

    # Count APIs
    apis = set(r['target_api'] for r in requests)
    print("")
    print("  Dataset: %d samples across %d APIs" % (len(requests), len(apis)))

    # Verify all APIs have route maps
    missing_routes = [a for a in apis if a not in route_maps]
    if missing_routes:
        print("  WARNING: %d APIs missing route maps: %s" % (
            len(missing_routes), missing_routes))

    # -- Step 6: Train --
    print("")
    print("=" * 60)
    print("TRAINING (Module-Routed)")
    print("=" * 60)
    start_train = time()
    model, training_log = train(
        model, tokenizer, requests, route_maps,
        module_map, original_scaling, config
    )
    train_time = time() - start_train
    print("")
    print("  Training completed in %.1fs" % train_time)

    # Save training log
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_log.json", 'w') as f:
        json.dump({
            'method': 'module_routed_shared_lora',
            'config': config,
            'editable_layers': editable_layers,
            'common_layers': common_layers,
            'route_config': route_config,
            'route_maps': route_maps,
            'train_time': round(train_time, 2),
            'epochs': training_log,
        }, f, indent=2)

    # Save LoRA adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("  Adapter saved to %s" % adapter_dir)

    # Save route maps alongside adapter for inference
    with open(adapter_dir / "route_maps.json", 'w') as f:
        json.dump(route_maps, f, indent=2)

    # -- Step 7: Evaluate --
    print("")
    print("=" * 60)
    print("EVALUATION (4 EDAPI Metrics, Route-Aware)")
    print("=" * 60)
    all_metrics, mean_metrics = evaluate_model(
        model, tokenizer, requests, route_maps,
        module_map, original_scaling, config, args.output_dir
    )

    # Print summary
    print("")
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for metric_name in ['efficacy', 'generalization', 'portability', 'specificity']:
        print("")
        print("  %s:" % metric_name.upper())
        for match_metric in MATCH_METRICS:
            mean_val, std_val = mean_metrics[metric_name][match_metric]
            print("    %-20s: %.2f%% +/- %.2f%%" % (match_metric, mean_val, std_val))

    print("")
    print("  TIME:   %.3fs +/- %.3fs" % (mean_metrics['time'][0], mean_metrics['time'][1]))
    print("  MEMORY: %.1fMB +/- %.1fMB" % (mean_metrics['max_memory'][0], mean_metrics['max_memory'][1]))
    print("=" * 60)
    print("")
    print("All results saved to %s/" % args.output_dir)


if __name__ == "__main__":
    main()
