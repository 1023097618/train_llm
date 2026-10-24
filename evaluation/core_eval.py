"""CORE metric (DCLM paper, https://arxiv.org/abs/2406.11794), ported from nanochat/core_eval.py

Each task is few-shot prompted:
- multiple_choice: same context, different continuations -> pick the lowest mean loss
- schema: different contexts, same continuation -> pick the lowest mean loss
- language_modeling: greedy prediction must reproduce the continuation exactly
Accuracy is centered by the random baseline and averaged over tasks.
"""
import csv
import json
import os
import random
import time

import torch
import torch.distributed as dist
from jinja2 import Template

from common.data_paths import EVAL_BUNDLE


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    context = {'fewshot_examples': fewshot_examples or [], 'continuation_delimiter': continuation_delimiter, 'item': item}
    return [template.render(choice=choice, **context) for choice in item['choices']]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    context = {'fewshot_examples': fewshot_examples or [], 'continuation_delimiter': continuation_delimiter, 'item': item}
    return [template.render(context=context_option, **context) for context_option in item['context_options']]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    context = {'fewshot_examples': fewshot_examples or [], 'continuation_delimiter': continuation_delimiter, 'item': item}
    # strip so the prompt without continuation is a clean token prefix of the one with it
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {'left': range(min_len), 'right': range(-1, -min_len - 1, -1)}[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    input_ids = torch.full((len(tokens), max(len(x) for x in tokens)), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def encode(tokeniz, prompts):
    bos = tokeniz.special_token['<|doc|>']
    return [[bos] + t for t in tokeniz.encode_ordinary(prompts)]


def batch_sequences(tokeniz, prompts, task_type):
    tokens = encode(tokeniz, prompts)
    if task_type == 'multiple_choice':
        start = find_common_length(tokens, direction='left')
        return tokens, [start] * len(tokens), [len(x) for x in tokens]
    if task_type == 'schema':
        suffix = find_common_length(tokens, direction='right')
        ends = [len(x) for x in tokens]
        return tokens, [e - suffix for e in ends], ends
    # language_modeling: only the prompt with the continuation is needed
    tokens_without, tokens_with = tokens
    start, end = len(tokens_without), len(tokens_with)
    assert start < end and tokens_without == tokens_with[:start], "prompt without is supposed to be a prefix of prompt with"
    return [tokens_with], [start], [end]


@torch.no_grad()
def forward_model(model, input_ids):
    """BxT token ids -> BxT losses (last column nan) and argmax predictions"""
    B, T = input_ids.shape
    logits = model(input_ids).logit
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(logits.reshape(B * T, -1).float(), target_ids.reshape(B * T), reduction='none').view(B, T)
    losses[:, -1] = float('nan')
    return losses, logits.argmax(dim=-1)


@torch.no_grad()
def evaluate_example(idx, model, tokeniz, data, device, task_meta, max_seq_len):
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        fewshot_examples = [data[i] for i in rng.sample([i for i in range(len(data)) if i != idx], num_fewshot)]
    render = {'multiple_choice': render_prompts_mc, 'schema': render_prompts_schema, 'language_modeling': render_prompts_lm}[task_type]
    prompts = render(item, task_meta['continuation_delimiter'], fewshot_examples)
    tokens, start_idxs, end_idxs = batch_sequences(tokeniz, prompts, task_type)
    # crop from the left to the model context, shifting the answer indices
    cropped = []
    for i, t in enumerate(tokens):
        if len(t) > max_seq_len:
            crop = len(t) - max_seq_len
            t, start_idxs[i], end_idxs[i] = t[-max_seq_len:], start_idxs[i] - crop, end_idxs[i] - crop
            assert start_idxs[i] >= 1
        cropped.append(t)
    input_ids = stack_sequences(cropped, tokeniz.special_token['<|doc|>']).to(device)
    losses, predictions = forward_model(model, input_ids)
    if task_type == 'language_modeling':
        si, ei = start_idxs[0], end_idxs[0]
        return torch.all(predictions[0, si - 1:ei - 1] == input_ids[0, si:ei]).item()
    mean_losses = [losses[i, si - 1:ei - 1].mean().item() for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
    return mean_losses.index(min(mean_losses)) == item['gold']


def evaluate_task(model, tokeniz, data, device, task_meta, max_seq_len):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        correct[idx] = float(evaluate_example(idx, model, tokeniz, data, device, task_meta, max_seq_len))
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


def evaluate_core(model, tokeniz, device, max_per_task=-1, bundle_dir=EVAL_BUNDLE, log=print):
    """returns {"results": {task: acc}, "centered_results": {task: centered}, "core_metric": float}"""
    import yaml
    if not os.path.exists(os.path.join(bundle_dir, "core.yaml")):
        raise FileNotFoundError(f"{bundle_dir}/core.yaml not found, run `python -m script.download_eval` first")
    with open(os.path.join(bundle_dir, "core.yaml"), encoding='utf-8') as f:
        tasks = yaml.safe_load(f)['icl_tasks']
    with open(os.path.join(bundle_dir, "eval_meta_data.csv"), encoding='utf-8') as f:
        random_baselines = {row['Eval Task']: float(row['Random baseline']) for row in csv.DictReader(f)}
    max_seq_len = model.config.seq_len
    results, centered_results = {}, {}
    was_training = model.training
    model.eval()
    for task in tasks:
        t0 = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' '),
        }
        with open(os.path.join(bundle_dir, "eval_data", task['dataset_uri']), encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        random.Random(1337).shuffle(data)  # consistent subsampling with max_per_task
        if max_per_task > 0:
            data = data[:max_per_task]
        accuracy = evaluate_task(model, tokeniz, data, device, task_meta, max_seq_len)
        baseline = 0.01 * random_baselines[label]
        results[label] = accuracy
        centered_results[label] = (accuracy - baseline) / (1.0 - baseline)
        log(f"{label:<32} ({task_meta['num_fewshot']}-shot {task_meta['task_type']}): "
            f"accuracy {accuracy:.4f} | centered {centered_results[label]:.4f} | {time.time() - t0:.1f}s")
    model.train(was_training)
    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}
