"""Chat evaluation tasks (nanochat tasks/{arc,mmlu,gsm8k,humaneval}.py)

Each task gives conversations {"messages": [...], ...} whose last message is the reference assistant answer,
and evaluate(conversation, completion_text) -> bool.
- categorical (ARC, MMLU): the answer letter is read from the logits right after <|assistant|>
- generative (GSM8K, HumanEval): the model samples a completion (GSM8K may call the calculator)
"""
import os
import re
import glob

import numpy as np
from pyarrow import parquet as pq

from common import data_paths
from evaluation.execution import execute_code
from tasks.sftdata import render_mc, GSM8K as SFTGSM8K, MMLU as SFTMMLU


def _read_parquet(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} not found, run `python -m script.download_eval` and `python -m script.download_sft` first")
    rows = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    return rows


class EvalTask:
    name = "task"
    eval_type = "categorical"
    baseline = 0.0  # random-guess accuracy, used by the ChatCORE metric

    def __init__(self):
        # fixed shuffle so --max_problems takes a random but reproducible subset
        self.order = np.random.default_rng(42).permutation(self.num_rows())

    def num_rows(self):
        raise NotImplementedError

    def get(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        conversation = self.get(int(self.order[index]))
        conversation.setdefault("source", self.name)
        return conversation

    @staticmethod
    def prompt_messages(conversation):
        """messages up to (and including) the first user message, the model completes the assistant turn"""
        messages = conversation["messages"]
        first_user = next(i for i, m in enumerate(messages) if m["role"] == "user")
        return messages[:first_user + 1]


class ARC(EvalTask):
    baseline = 0.25

    def __init__(self, subset):
        assert subset in ("ARC-Easy", "ARC-Challenge")
        self.name = subset
        self.rows = _read_parquet(os.path.join(data_paths.EVAL_ARC, subset, "test-*.parquet"))
        super().__init__()

    def num_rows(self):
        return len(self.rows)

    def get(self, index):
        row = self.rows[index]
        letters = row["choices"]["label"]
        assert row["answerKey"] in letters
        return {"messages": [{"role": "user", "content": render_mc(row["question"], letters, row["choices"]["text"])},
                             {"role": "assistant", "content": row["answerKey"]}],
                "letters": letters}

    def evaluate(self, conversation, response):
        return response == conversation["messages"][-1]["content"]


class MMLU(EvalTask):
    name = "MMLU"
    baseline = 0.25

    def __init__(self):
        self.task = SFTMMLU("test")
        super().__init__()

    def num_rows(self):
        return len(self.task)

    def get(self, index):
        return {"messages": self.task.get_messages(index), "letters": list(SFTMMLU.letters)}

    def evaluate(self, conversation, response):
        return response == conversation["messages"][-1]["content"]


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_gsm8k_answer(text):
    match = GSM_RE.search(text)
    return match.group(1).strip().replace(",", "") if match else None


class GSM8K(EvalTask):
    name = "GSM8K"
    eval_type = "generative"

    def __init__(self, split="test"):
        self.task = SFTGSM8K(split)
        super().__init__()

    def num_rows(self):
        return len(self.task)

    def get(self, index):
        # system (calculator schema) + user + assistant/tool turns, same format as SFT
        return {"messages": self.task.get_messages(index)}

    def evaluate(self, conversation, response):
        reference = extract_gsm8k_answer(conversation["messages"][-1]["content"])
        predicted = extract_gsm8k_answer(response)
        return predicted is not None and predicted == reference

    def reward(self, conversation, response):
        return float(self.evaluate(conversation, response))


def _extract_imports(prompt):
    imports = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def _extract_program(completion):
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", completion, re.DOTALL)
    return matches[0].strip() if matches else completion.strip()


class HumanEval(EvalTask):
    name = "HumanEval"
    eval_type = "generative"

    def __init__(self):
        self.rows = _read_parquet(os.path.join(data_paths.EVAL_HUMANEVAL, "openai_humaneval", "test-*.parquet"))
        super().__init__()

    def num_rows(self):
        return len(self.rows)

    def get(self, index):
        row = self.rows[index]
        return {"messages": [{"role": "user", "content": row["prompt"]},
                             {"role": "assistant", "content": f"{row['prompt']}\n{row['canonical_solution']}"}],
                "entry_point": row["entry_point"], "test": row["test"]}

    def evaluate(self, conversation, response):
        program = (_extract_imports(conversation["messages"][0]["content"]) + "\n\n" + _extract_program(response)
                   + "\n\n" + conversation["test"] + "\n" + f"check({conversation['entry_point']})")
        return execute_code(program).success


ALL_TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]
EVAL_TYPE = {"ARC-Easy": "categorical", "ARC-Challenge": "categorical", "MMLU": "categorical", "GSM8K": "generative", "HumanEval": "generative"}
BASELINE = {"ARC-Easy": 0.25, "ARC-Challenge": 0.25, "MMLU": 0.25, "GSM8K": 0.0, "HumanEval": 0.0}


def build_task(name):
    if name in ("ARC-Easy", "ARC-Challenge"):
        return ARC(name)
    return {"MMLU": MMLU, "GSM8K": GSM8K, "HumanEval": HumanEval}[name]()
