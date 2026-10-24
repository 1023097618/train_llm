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


_TEMPLATES = {}


def _template(template_str):
    # compiling a jinja template is much slower than rendering it, do it once per template
    if template_str not in _TEMPLATES:
        _TEMPLATES[template_str] = Template(template_str)
    return _TEMPLATES[template_str]


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = _template(template_str)
    context = {'fewshot_examples': fewshot_examples or [], 'continuation_delimiter': continuation_delimiter, 'item': item}
    return [template.render(choice=choice, **context) for choice in item['choices']]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = _template(template_str)
    context = {'fewshot_examples': fewshot_examples or [], 'continuation_delimiter': continuation_delimiter, 'item': item}
    return [template.render(context=context_option, **context) for context_option in item['context_options']]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = _template(template_str)
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


def prepare_example(idx, tokeniz, data, task_meta, max_seq_len):
    """render + tokenize one example -> (sequences, answer starts, answer ends)"""
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
    for i, t in enumerate(tokens):
        if len(t) > max_seq_len:
            crop = len(t) - max_seq_len
            tokens[i], start_idxs[i], end_idxs[i] = t[-max_seq_len:], start_idxs[i] - crop, end_idxs[i] - crop
            assert start_idxs[i] >= 1
    return tokens, start_idxs, end_idxs


@torch.no_grad()
def score_sequences(model, sequences, pad_token_id, device, batch_tokens):
    """
    sequences: list of (tokens, start, end)
    returns per sequence (mean loss of tokens[start:end], whether greedy argmax reproduces tokens[start:end])
    sequences are sorted by length and packed into right padded batches of <= batch_tokens tokens
    """
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i][0]))
    results = [None] * len(sequences)
    i = 0
    while i < len(order):
        longest = len(sequences[order[i]][0])
        n = max(1, batch_tokens // longest)
        batch = order[i:i + n]
        i += n
        input_ids = stack_sequences([sequences[j][0] for j in batch], pad_token_id).to(device)
        logits = model(input_ids).logit
        B, T = input_ids.shape
        target_ids = torch.roll(input_ids, shifts=-1, dims=1)
        losses = torch.nn.functional.cross_entropy(logits.reshape(B * T, -1), target_ids.reshape(B * T), reduction='none').view(B, T)
        correct = logits.argmax(dim=-1) == target_ids
        del logits
        for row, j in enumerate(batch):
            _, s, e = sequences[j]
            results[j] = (losses[row, s - 1:e - 1].mean().item(), bool(correct[row, s - 1:e - 1].all().item()))
    return results


def evaluate_task(model, tokeniz, data, device, task_meta, max_seq_len, batch_tokens=8192):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    indices = list(range(rank, len(data), world_size))
    sequences, owners = [], []
    for idx in indices:
        tokens, starts, ends = prepare_example(idx, tokeniz, data, task_meta, max_seq_len)
        for t, s, e in zip(tokens, starts, ends):
            sequences.append((t, s, e))
            owners.append(idx)
    scores = score_sequences(model, sequences, tokeniz.special_token['<|doc|>'], device, batch_tokens)
    per_example = {}
    for idx, score in zip(owners, scores):
        per_example.setdefault(idx, []).append(score)
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx, sc in per_example.items():
        if task_meta['task_type'] == 'language_modeling':
            correct[idx] = float(sc[0][1])
        else:  # multiple_choice / schema: the option with the lowest mean loss
            mean_losses = [loss for loss, _ in sc]
            correct[idx] = float(mean_losses.index(min(mean_losses)) == data[idx]['gold'])
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


def evaluate_core(model, tokeniz, device, max_per_task=-1, bundle_dir=EVAL_BUNDLE, log=print, batch_tokens=8192):
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
        accuracy = evaluate_task(model, tokeniz, data, device, task_meta, max_seq_len, batch_tokens)
        baseline = 0.01 * random_baselines[label]
        results[label] = accuracy
        centered_results[label] = (accuracy - baseline) / (1.0 - baseline)
        log(f"{label:<32} ({task_meta['num_fewshot']}-shot {task_meta['task_type']}): "
            f"accuracy {accuracy:.4f} | centered {centered_results[label]:.4f} | {time.time() - t0:.1f}s")
    model.train(was_training)
    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}
