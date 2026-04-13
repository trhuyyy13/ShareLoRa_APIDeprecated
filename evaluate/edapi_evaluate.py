"""
EDAPI evaluation: compute edit quality metrics.
Adapted from EDAPI codellmeditor/evaluate/edapi_evaluate.py with standalone imports.
"""

import typing
import torch
import re
import numpy as np
import copy
from .evaluate_utils import (
    batch_generate,
    Metric,
    test_generation_quality,
    MATCH_METRICS,
    clean_pred,
    extract_first_statement,
    extract_first_func,
    extract_apis_in_first_stmt,
)
import logging

LOG = logging.getLogger(__name__)


def compute_edit_quality(
    model,
    tok,
    record: typing.Dict,
    test_generation: bool,
    tokenizer_for_fluency=None,
) -> typing.Dict:
    # First, unpack rewrite evaluation record.
    intent = record['prompt']
    rewritten_intent = record['rephrase_prompt']
    rewritten_target = record['rephrase_target_new']
    target_snippet = record['target_new']
    neighborhoods = record['specificity']
    reference_dict = record['reference_dict']
    alias_dict = record['alias_dict']
    rephrase_reference_dict = record['rephrase_reference_dict']
    new_api = record['new_api']
    portability = record["portability"]

    # for replace_api evaluation
    if 'replace_prompt' in record:
        intent = record['replace_prompt']
        rewritten_intent = record['replace_rephrase_prompt']
    
    ret = {}
    ret['gen_strs'] = []
    ret['gen_apis'] = []
    ret['gen_apis_rephrase'] = []
    ## Test efficacy and generalization
    target_snippet_token = tok.encode(target_snippet)
    if portability != "":
        if 'replace_prompt' in record:
            gen_strs = batch_generate(model, tok, [intent, rewritten_intent, portability['replace_prompt']], max_length=50)
        else:
            gen_strs = batch_generate(model, tok, [intent, rewritten_intent, portability['prompt']], max_length=50)
    else:
        gen_strs = batch_generate(model, tok, [intent, rewritten_intent], max_length=50)
        
    _preds = [clean_pred(p) for p in gen_strs]
    if 'replace_prompt' in record:
        _preds[0] = extract_first_func(record["replace_prompt"] + _preds[0])[len(record["prompt"]):]
        _preds[1] = extract_first_func(record["replace_rephrase_prompt"] + _preds[1])[len(record["rephrase_prompt"]):]
        if portability != "":
            _preds[2] = extract_first_func(record["portability"]["replace_prompt"] + _preds[2])[len(record["portability"]["prompt"]):]
    gen_strs = [extract_first_statement(p, False) for p in _preds]
    gen_apis_prompt = [extract_apis_in_first_stmt(_preds[0], reference_dict, alias_dict)]
    gen_apis_rephrase = [extract_apis_in_first_stmt(_preds[1], rephrase_reference_dict, alias_dict)]
    if portability != "":
        gen_apis_portability = [extract_apis_in_first_stmt(_preds[2], portability['reference_dict'], portability['alias_dict'])]
        
    ret['gen_strs'].append(gen_strs)
    ret['gen_apis'] = gen_apis_prompt
    ret['gen_apis_rephrase'] = gen_apis_rephrase
    ret['gen_apis_portability'] = gen_apis_portability if portability != "" else []
    ret['efficacy'] = {}
    ret['generalization'] = {}
    ret['portability'] = {}  
    for i, func in enumerate([
                Metric.api_exact_match_for_efficacy,
                Metric.exact_match,
                Metric.bleu_score,
                Metric.rouge_score,
        ]):
        if gen_strs[0].strip() == '':
            ret['efficacy'][MATCH_METRICS[i]] = 0
        else:
            if i == 0:
                ret['efficacy'][MATCH_METRICS[i]] = func(gen_apis_prompt, new_api)
            else:
                ret['efficacy'][MATCH_METRICS[i]] = func([gen_strs[0].strip()], [target_snippet.strip()])
        if gen_strs[1].strip() == '':
            ret['generalization'][MATCH_METRICS[i]] = 0
        else:
            if i == 0:
                ret['generalization'][MATCH_METRICS[i]] = func(gen_apis_rephrase, new_api)
            else:
                ret['generalization'][MATCH_METRICS[i]] = func([gen_strs[1].strip()], [rewritten_target.strip()])
        if portability == "" or gen_strs[2].strip() == '':
            ret['portability'][MATCH_METRICS[i]] = 0
        else:
            if i == 0:
                ret['portability'][MATCH_METRICS[i]] = func(gen_apis_portability, new_api)
            else:
                ret['portability'][MATCH_METRICS[i]] = func([gen_strs[2].strip()], [portability['target_new'].strip()])
    
    ## Test specificity
    ret['specificity'] = {}
    gen_strs = []
    for prompt in neighborhoods['prompts']:
        gen_strs += batch_generate(model, tok, prompt, max_length=50)
    _preds = [clean_pred(p) for p in gen_strs]
    gen_strs = [extract_first_statement(p, False) for p in _preds]
    gen_apis = [extract_apis_in_first_stmt(p, {}, alias_dict) for p in gen_strs]
    ret['spec_output'] = gen_strs
    for i, func in enumerate([
            Metric.exact_match,
            Metric.exact_match,
            Metric.bleu_score,
            Metric.rouge_score,
    ]):
        if i == 0:
            score = func(gen_apis, neighborhoods['pred-api'], True, True)
            ret['specificity'][MATCH_METRICS[i]] = np.mean(score)
        elif i ==1:
            score = func(gen_strs, neighborhoods['ground_truth'], True)
            ret['specificity'][MATCH_METRICS[i]] = np.mean(score)
        else:
            score = func(gen_strs, neighborhoods['ground_truth'])
            ret['specificity'][MATCH_METRICS[i]] = np.mean(score)
    if test_generation:
        try:
            res = test_generation_quality(model, tok, [intent, rewritten_intent], max_out_len=50)
            ret.update(res)
        except Exception as e:
            LOG.warn(f"Case {record['case_id']} test generation raise error {e}")
    return ret
