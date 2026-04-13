"""
Evaluation utilities for SharedLoRA pipeline.
Adapted from EDAPI codellmeditor/evaluate/evaluate_utils.py with standalone imports.
"""

import torch
import numpy as np
import scipy
import nltk
import typing
import re
from difflib import SequenceMatcher
from evaluate import load as evaluate_load
from .bleu.bleu import Bleu
import logging
import os

MATCH_METRICS = ['api_exact_match', 'exact_match', 'bleu', 'rougeL']

# Determine the directory of this file for loading local metrics
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def format_ratio(pre_datum, post_datum):
    sign_prefix = ('+' if post_datum >= pre_datum else '')
    abs_ratio = sign_prefix + f'{format_score((post_datum - pre_datum) * 100)}%'
    rel_ratio = sign_prefix + f'{format_score((post_datum / pre_datum - 1.) * 100)}%'
    return abs_ratio, rel_ratio

def format_score(datum):
    return round(datum, 3)

class Metric:
    @staticmethod
    def exact_match(gens: list, refs: list, is_specificity=False, is_api=False):
        """Exact Match on the token-level"""
        if not is_specificity:
            score = np.prod([1 if sorted(g) == sorted(r) else 0 for g, r in zip(gens, refs)])
            return format_score(float(score))
        else:
            score = 0
            n = len(gens)
            for i in range(n):
                if is_api and set(gens[i]) == set(refs[i]) and len(gens[i]) > 0:
                    score += 1
                elif not is_api and gens[i].strip() != '' and gens[i].strip() == refs[i].strip():
                    score += 1
            score = score / n if n > 0 else 0.0
            return format_score(float(score))

    @staticmethod
    def bleu_score(predictions: list, references: list):
        """BLEU on the token-level"""
        predictions = [prediction for prediction in predictions]
        references = [[reference] for reference in references]
        metric = Bleu()
        score = metric.compute(predictions=predictions, references=references)['bleu']
        return format_score(score)

    @staticmethod
    def rouge_score(predictions: list, references: list):
        """ROUGE on the token-level"""
        predictions = [prediction for prediction in predictions]
        references = [reference for reference in references]
        rouge_path = os.path.join(_THIS_DIR, 'rouge')
        metric = evaluate_load(rouge_path)
        score = metric.compute(predictions=predictions, references=references)['rougeL']
        return format_score(score)

    @staticmethod
    def api_exact_match_for_efficacy(gens: list, refs: list):
        """Exact Match on the token-level"""
        gens, ref = gens[0], refs[0][0]
        for gen in gens:
            if gen == ref:
                return format_score(float(1))
        return format_score(float(0))

def batch_generate(model, tok, prompts, max_length, sample_generate=False):
    prompt_tok = tok(
        prompts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    gen_args = {
            'inputs': prompt_tok['input_ids'],
            'attention_mask': prompt_tok['input_ids'].ne(tok.pad_token_id),
            'max_new_tokens': max_length,
            'pad_token_id': tok.eos_token_id,
            'eos_token_id': tok.eos_token_id,
            'top_k': None,
            'top_p': None,
            'temperature': 1.0,
            'do_sample': False,
            'num_beams': 1,
            'num_return_sequences': 1,
        }
    with torch.no_grad():
        if sample_generate:
            gen_args.update({
                'do_sample': True,
                'num_beams': 1,
                'top_k': 5,
            })
        gen_tokens = model.generate(**gen_args)
    return tok.batch_decode(gen_tokens[:, prompt_tok['input_ids'].shape[1]:])


def test_generation_quality(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,
    tokenizer_for_fluency=None
):
    gen_texts = batch_generate(
        model,
        tok,
        prefixes,
        max_out_len,
        True
    )
    ngram_entropy = n_gram_entropy(gen_texts, tokenizer_for_fluency=tokenizer_for_fluency)
    ret = {
        "ngram_entropy": ngram_entropy,
        "generated_texts" : gen_texts
    }
    return ret


def n_gram_entropy(gen_texts, agg="arith", tokenizer_for_fluency=None):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt, tokenizer_for_fluency=tokenizer_for_fluency) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith", tokenizer_for_fluency=None):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n, tokenizer_for_fluency)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2, tokenizer_for_fluency=None):
    if tokenizer_for_fluency is not None:
        tokens = tokenizer_for_fluency.encode(sentence)
    else:
        tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)

def clean_pred(pred: str):
    lines = [line for line in pred.split("\n") if not line.strip().startswith("#")]
    return "\n".join(lines)

def extract_first_statement(pred: str, remove_space=True):
    def unclosed(_stmt: str):
        if len(_stmt) == 0:
            return True
        if _stmt.count("(") > _stmt.count(")"):
            return True
        if _stmt.count("[") > _stmt.count("]"):
            return True
        if _stmt.count("{") > _stmt.count("}"):
            return True
        if _stmt.rstrip().endswith("\\"):
            return True
        return False
    
    def normalize(_line: str):
        _line = _line.split("#")[0]
        
        if remove_space:
            _line = _line.strip().rstrip(" \\")
            _line = re.sub(r"\s+", " ", _line)
        else:
            _line = _line.rstrip(" \\")
        return _line

    
    lines = pred.split("\n")
    stmt = normalize(lines.pop(0))
    
    while unclosed(stmt) and len(lines) > 0:
        stmt += normalize(lines.pop(0))
    return stmt

def extract_first_func(code: str):
    lines = code.split("\n")
    while len(lines) > 0 and not lines[0].lstrip().startswith("def "):
        lines.pop(0)
    if len(lines) == 0:
        return code
    indent = len(re.search("^\s*", lines[0]).group(0))
    func = "\n".join(lines)
    func = re.split(r"\n {0,%d}[^\s#]" % indent, func, flags=re.M|re.S)[0]
    return func


def extract_apis_in_first_stmt(pred, ref_dict, alias_dict):
    stmt = extract_first_statement(pred, False)
    pkg_as = dict()
    for alias, name in alias_dict.items():
        alias_parts = alias.split(".")
        name_parts = name.split(".")
        while len(alias_parts) > 0 and len(name_parts) > 0 and alias_parts[-1] == name_parts[-1]:
            alias_parts.pop()
            name_parts.pop()
        pkg_alias, pkg_name = ".".join(alias_parts), ".".join(name_parts)
        if pkg_alias != pkg_name:
            pkg_as[pkg_alias] = pkg_name

    apis = set()
    for mobj in re.finditer(r"([\w\.]+)\s*\(", stmt):
        api = mobj.group(1).strip()
        if api == "":
            continue
        parts = api.split('.')
        if len(parts) == 2 and parts[0] in ref_dict:
            api = f"{ref_dict[parts[0]]}.{parts[1]}"
        if api in alias_dict:
            api = alias_dict[api]
        else:
            for pkg_alias, pkg_name in pkg_as.items():
                if api.startswith(f"{pkg_alias}."):
                    api = api.replace(f"{pkg_alias}.", f"{pkg_name}.")
                    break
        apis.add(api)
    return list(apis)
