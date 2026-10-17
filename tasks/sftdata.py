"""SFT 数据：nanochat 的对话数据混合 + best-fit 打包加载器。

三层结构：
- Task：一个数据源，get_example(i) 返回 {"source", "messages"}
- TaskMixture：把多个 Task 按固定 seed 打乱混在一起，同一个 Task 传多次即多个 epoch
- SFTDataset：从 TaskMixture 里按 (rank, worker) 步长取对话，tokenize 后 best-fit 打包成 batch
"""

import os
import glob
import json
import re
import numpy as np
import pyarrow as pa
import torch
from pyarrow import parquet as pq
from torch.utils.data import get_worker_info

from tokenizer import tokenizer
from tasks.basedata import BaseDataset
from common import data_paths
from evaluation.sample import calculate


class Task:
    name = "task"

    def __init__(self, start=0, stop=None):
        self.start = start
        self.stop = stop

    def num_rows(self):
        raise NotImplementedError

    def get_messages(self, index):
        raise NotImplementedError

    def __len__(self):
        total = self.num_rows()
        stop = total if self.stop is None else min(self.stop, total)
        return max(0, stop - self.start)

    def get_example(self, index):
        return {"source": self.name, "messages": self.get_messages(self.start + index)}


class ParquetTask(Task):
    """只读需要的列，按行随机访问。"""

    def __init__(self, data_dir, split, columns, start=0, stop=None):
        super().__init__(start, stop)
        files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"no {split}-*.parquet under {data_dir}, run `python -m script.download_sft` first"
            )
        table = pa.concat_tables([pq.read_table(f, columns=columns) for f in files])
        self.columns = {c: table.column(c) for c in columns}
        self.rows = table.num_rows

    def num_rows(self):
        return self.rows

    def row(self, index):
        return {c: col[index].as_py() for c, col in self.columns.items()}


class SmolTalk(ParquetTask):
    """HuggingFaceTB/smol-smoltalk：通用多轮对话，train 460K / test 24K。"""
    name = "smoltalk"

    def __init__(self, split, **kwargs):
        super().__init__(data_paths.SFT_SMOLTALK, split, ["messages"], **kwargs)

    def get_messages(self, index):
        return self.row(index)["messages"]


def render_mc(question, letters, choices):
    # 与 nanochat 的 tasks/common.py 保持一致
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


class MMLU(ParquetTask):
    """cais/mmlu all：auxiliary_train 100K 选择题，test 14K。"""
    name = "mmlu"
    letters = ("A", "B", "C", "D")

    def __init__(self, split, **kwargs):
        super().__init__(data_paths.SFT_MMLU, split, ["question", "choices", "answer"], **kwargs)

    def get_messages(self, index):
        row = self.row(index)
        return [
            {"role": "user", "content": render_mc(row["question"], self.letters, row["choices"])},
            {"role": "assistant", "content": self.letters[row["answer"]]},
        ]


GSM8K_TOOL_PATTERN = re.compile(r"(<<[^>]+>>)")


# 和推理时完全一致：system 里给出工具 schema（同 Tokenizer.render_prompt_for_completion），
# 调用格式对应 inference/engine.py 的 safe_execute_function，返回值就是 calculate 的真实输出
CALCULATOR_TOOLS = "[" + calculate.schema + "]"


def calculator_call(expression):
    return json.dumps(
        [{"function": {"name": "calculate", "arguments": json.dumps({"expression": expression})}}]
    )


def _to_number(text):
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def calculator_output(expression, result):
    output = json.loads(calculate(expression))
    expected = _to_number(result)
    value = output.get("result")
    # GSM8K 的表达式偶尔带千分位逗号或笔误，算出来和标注不一致时用标注值，保证和后文一致
    if expected is not None and (not isinstance(value, (int, float)) or abs(value - expected) > 1e-6 * max(1, abs(expected))):
        output = {"expression": expression, "result": expected}
    return json.dumps(output, ensure_ascii=False)


class GSM8K(ParquetTask):
    """openai/gsm8k main：train 7.5K / test 1.3K，答案里的 <<expr=result>> 转成计算器调用。"""
    name = "gsm8k"

    def __init__(self, split, **kwargs):
        super().__init__(data_paths.SFT_GSM8K, split, ["question", "answer"], **kwargs)

    def get_messages(self, index):
        row = self.row(index)
        messages = [{"role": "system", "content": "", "tools": CALCULATOR_TOOLS},
                    {"role": "user", "content": row["question"]}]
        text = ""
        for part in GSM8K_TOOL_PATTERN.split(row["answer"]):
            if part.startswith("<<") and part.endswith(">>"):
                expression, result = part[2:-2].rsplit("=", 1)
                messages.append({"role": "assistant", "content": text,
                                 "tool_calls": calculator_call(expression)})
                messages.append({"role": "tool", "content": calculator_output(expression, result)})
                text = ""
            else:
                text += part
        messages.append({"role": "assistant", "content": text})
        return messages


HERMES_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
HERMES_TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def convert_hermes(row):
    """hermes 的 system/human/gpt/tool 对话 -> 和推理时一致的 tools / tool_calls 格式；格式不对返回 None。"""
    # 推理时 system 里是 "[" + fn.schema + "]"，schema 只有 {"function": {...}}，没有 "type"
    tools = json.loads(row["tools"]) if row.get("tools") else None
    messages = []
    if tools:
        tools = json.dumps([{"function": t.get("function", t)} for t in tools], ensure_ascii=False)
        messages.append({"role": "system", "content": "", "tools": tools})
    for turn in row["conversations"][1:]:
        speaker, value = turn["from"], turn["value"]
        if speaker == "human":
            messages.append({"role": "user", "content": value})
        elif speaker == "gpt":
            calls = HERMES_TOOL_CALL.findall(value)
            if not calls:
                messages.append({"role": "assistant", "content": value.strip()})
                continue
            calls = [json.loads(c) for c in calls]
            messages.append({"role": "assistant", "content": "", "tool_calls": json.dumps(
                [{"function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                 for c in calls], ensure_ascii=False)})
        elif speaker == "tool":
            if not messages or not messages[-1].get("tool_calls"):
                return None
            # engine 把每个函数的 json 输出用换行拼成一个 tool output
            responses = [json.loads(r)["content"] for r in HERMES_TOOL_RESPONSE.findall(value)]
            messages.append({"role": "tool", "content": "\n".join(
                r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in responses)})
        else:
            return None
    while messages and (messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls")):
        messages.pop()
    return messages if any(m["role"] == "user" for m in messages) else None


class HermesFunctionCalling(Task):
    """NousResearch/hermes-function-calling-v1：func-calling 1.9K + glaive 5.2K 条多轮函数调用对话。"""
    name = "toolcall"
    files = ("func-calling.json", "glaive-function-calling-5k.json")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = []
        for name in self.files:
            path = os.path.join(data_paths.SFT_TOOLCALL, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found, run `python -m script.download_sft` first")
            with open(path, "r", encoding="utf-8") as f:
                for row in json.load(f):
                    try:
                        messages = convert_hermes(row)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        messages = None
                    if messages:
                        self.rows.append(messages)

    def num_rows(self):
        return len(self.rows)

    def get_messages(self, index):
        return self.rows[index]


class CustomJSON(Task):
    """每行一个 {"messages": [...]} 的 jsonl，比如自己合成的身份认知对话。"""
    name = "custom"

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line)["messages"] for line in f if line.strip()]

    def num_rows(self):
        return len(self.rows)

    def get_messages(self, index):
        return self.rows[index]


class TaskMixture:
    """确定性地打乱所有 Task 的样本，让各个任务均匀分布在整个训练过程里。"""

    def __init__(self, tasks, seed=42):
        if not tasks:
            raise ValueError("TaskMixture requires at least one task")
        self.tasks = list(tasks)
        task_ids = np.concatenate([np.full(len(t), i, dtype=np.int32) for i, t in enumerate(self.tasks)])
        local_ids = np.concatenate([np.arange(len(t), dtype=np.int64) for t in self.tasks])
        perm = np.random.default_rng(seed).permutation(len(task_ids))
        self.task_ids = task_ids[perm]
        self.local_ids = local_ids[perm]

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, index):
        return self.tasks[self.task_ids[index]].get_example(int(self.local_ids[index]))


def preprocess_conversation(conversation):
    messages = list(conversation["messages"])
    if len(messages) > 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'user':
        messages[0] = {**messages[0], 'role': 'system'}
    if messages and messages[-1]['role'] == 'user':
        messages.pop()
    return {"source": conversation["source"], "messages": messages}


BANNED_PATTERN = re.compile(
    r"(?:作为|身为|我是)(?:一个|一名)?\s*(?:ai|人工智能|大?语言模型)|"
    r"(?:as\s+an?|i\s+am\s+an?|i['’]m\s+an?)\s+"
    r"(?:ai|artificial\s+intelligence|(?:large\s+)?language\s+model)",
    re.IGNORECASE,
)


def _text_of(value):
    """把 content / reasoning_content / tool_calls 统一摊平成字符串列表。"""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for part in value:
            if isinstance(part, dict) and isinstance(part.get('content'), str):
                out.append(part['content'])
            elif isinstance(part, str):
                out.append(part)
        return out
    return []


def contains_banned_words(conversation):
    for message in conversation['messages']:
        for field in ('content', 'reasoning_content', 'tool_calls', 'tools'):
            for text in _text_of(message.get(field)):
                if BANNED_PATTERN.search(text):
                    return True
    return False


class SFTDataset(BaseDataset):
    """和 PretrainDataset 一样配合 DataLoader 多 worker 使用：shard_id = rank*num_workers + worker_id，
    每个 shard 依次取 mixture 的第 shard_id, shard_id+num_shards, ... 条对话，打包成 CPU 上的 (x, y, segment_ids)。
    resume_state_dict 以 worker id 为 key。"""

    def __init__(self, mixture: TaskMixture, tokeniz: tokenizer.Tokenizer,
                 batch_size, seq_len, rank=0, world_size=1,
                 resume_state_dict=None, buffer_capacity=1000):
        self.mixture = mixture
        self.tokeniz = tokeniz
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.resume_state_dict = resume_state_dict
        self.buffer_capacity = buffer_capacity

    def __len__(self):
        return len(self.mixture)

    def _tokenize(self, example):
        """返回 (token, mask)；被过滤或超长的对话返回 None。"""
        conversation = preprocess_conversation(example)
        if not conversation["messages"] or contains_banned_words(conversation):
            return None
        token, mask = self.tokeniz.render_conversation(conversation)
        if len(token) > self.seq_len + 1:
            return None
        return token, mask

    def _get_conversation(self, epoch, cursor, shard_id, num_shards):
        while True:
            while cursor < len(self.mixture):
                example = self.mixture[cursor]
                cursor += num_shards
                rendered = self._tokenize(example)
                if rendered is not None:
                    yield *rendered, epoch, cursor
            cursor = shard_id
            epoch += 1

    def __iter__(self):
        info = get_worker_info()
        (wid, nw) = (info.id, info.num_workers) if info is not None else (0, 1)
        shard_id = self.rank * nw + wid
        num_shards = self.world_size * nw
        tokeniz, seq_len, batch_size = self.tokeniz, self.seq_len, self.batch_size
        epoch, cursor = 0, shard_id
        resume = (self.resume_state_dict or {}).get(wid)
        if resume:
            epoch = resume['epoch']
            cursor = resume['cursor']
        conversation_loader = self._get_conversation(epoch, cursor, shard_id, num_shards)
        eos = tokeniz.special_token['<|/doc|>']
        token_buffer = []
        mask_buffer = []

        def refill_buffer():
            nonlocal epoch, cursor
            while len(token_buffer) < self.buffer_capacity:
                token, mask, epoch, cursor = next(conversation_loader)
                token_buffer.append(token)
                mask_buffer.append(mask)

        while True:
            token_batch = []
            mask_batch = []
            segment_batch = []
            for _ in range(batch_size):
                token = []
                mask = []
                segment = []
                remaining = seq_len + 1
                segment_id = 0
                while remaining > 0:
                    refill_buffer()
                    best_idx = -1
                    best_len = -1
                    for j, buffer in enumerate(token_buffer):
                        if len(buffer) <= remaining and len(buffer) > best_len:
                            best_len = len(buffer)
                            best_idx = j
                    if best_idx == -1:
                        token.extend([eos] * remaining)
                        mask.extend([tokenizer.Ignored.YES] * remaining)
                        segment.extend([segment_id] * remaining)
                        remaining = 0
                    else:
                        token.extend(token_buffer[best_idx])
                        token_buffer[best_idx] = token_buffer[-1]
                        token_buffer.pop()
                        mask.extend(mask_buffer[best_idx])
                        mask_buffer[best_idx] = mask_buffer[-1]
                        mask_buffer.pop()
                        segment.extend([segment_id] * best_len)
                        remaining -= best_len
                    segment_id += 1
                token_batch.append(token)
                mask_batch.append(mask)
                segment_batch.append(segment)
            token_batch = torch.tensor(token_batch)
            mask_batch = torch.tensor(mask_batch)
            segment_batch = torch.tensor(segment_batch)
            x = token_batch[:, :-1].contiguous()
            y = token_batch[:, 1:].contiguous()
            y[mask_batch[:, :-1] == tokenizer.Ignored.YES] = -1
            segment_batch = segment_batch[:, :-1].contiguous()
            state_dict = {'worker': wid, 'epoch': epoch, 'cursor': cursor}
            yield x, y, segment_batch, state_dict

    def estimate_total_token(self, sample_num=2000):
        # mixture 已经打乱，取前 sample_num 条估平均长度，再乘总条数（所有 rank 合计）
        total_row = len(self.mixture)
        sample_num = min(sample_num, total_row)
        token_sum = 0
        for index in range(sample_num):
            rendered = self._tokenize(self.mixture[index])
            if rendered is not None:
                token_sum += len(rendered[0])
        return int(token_sum / sample_num * total_row)
