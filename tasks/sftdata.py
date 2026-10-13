import os
import glob
import re
import torch
from typing import Literal
from pyarrow import parquet as pq

from tokenizer import tokenizer
from tasks.basedata import BaseDataset


def preprocess_conversation(conversation):
    if len(conversation) > 2 and conversation[0]['role'] == 'user' and conversation[1]['role'] == 'user':
        conversation[0]['role'] = 'system'
    if conversation[len(conversation) - 1]['role'] == 'user':
        conversation.pop()
    return {"source": "sft", "messages": conversation}


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
    def __init__(self, data_dir, split: Literal["train", "test"], tokeniz: tokenizer.Tokenizer,
                 batch_size, seq_len, device, resume_state_dict=None,buffer_capacity=1000,
                 file_pattern="sft_t2t-*.parquet", holdout_last=True):
        self.tokeniz = tokeniz
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.resume_state_dict = resume_state_dict
        self.buffer_capacity=buffer_capacity
        data_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if holdout_last:
            self.data_files = data_files[:-1] if split == 'train' else data_files[-1:]
        else:
            # Small specialist datasets commonly have only one shard.  In that
            # case use all rows; callers can disable evaluation or provide a
            # separate validation run instead of silently creating an empty set.
            self.data_files = data_files

    def __len__(self):
        return sum(pq.ParquetFile(f).metadata.num_rows for f in self.data_files)

    def _get_file_row(self, batch_size, epoch, f_idx, rg_idx):
        while True:
            while f_idx < len(self.data_files):
                pq_file = pq.ParquetFile(self.data_files[f_idx])
                while rg_idx < pq_file.num_row_groups:
                    row_group = pq_file.read_row_group(rg_idx)['conversations'].to_pylist()
                    for start in range(0, len(row_group), batch_size):
                        conversations = row_group[start:start + batch_size]
                        conversations = [preprocess_conversation(c) for c in conversations]
                        conversations = [c for c in conversations if not contains_banned_words(c)]
                        yield conversations, epoch, f_idx, rg_idx
                    rg_idx += 1
                rg_idx = 0
                f_idx += 1
            f_idx = 0
            epoch += 1

    def __iter__(self):
        tokeniz, seq_len, batch_size, device = self.tokeniz, self.seq_len, self.batch_size, self.device
        epoch, f_idx, rg_idx = 0, 0, 0
        if self.resume_state_dict:
            epoch = self.resume_state_dict['epoch']
            f_idx = self.resume_state_dict['f_idx']
            rg_idx = self.resume_state_dict['rg_idx']
        text_loader = self._get_file_row(128, epoch, f_idx, rg_idx)
        eos = tokeniz.special_token['<|/doc|>']
        token_buffer = []
        mask_buffer = []

        def refill_buffer():
            nonlocal epoch, f_idx, rg_idx
            while len(token_buffer) < self.buffer_capacity:
                conversations, epoch, f_idx, rg_idx = next(text_loader)
                for conversation in conversations:
                    token, mask = tokeniz.render_conversation(conversation)
                    if len(token) > seq_len + 1:
                        continue
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
            x = token_batch[:, :-1].contiguous().pin_memory().to(device=device, non_blocking=True)
            y = token_batch[:, 1:].contiguous().pin_memory().to(device=device, non_blocking=True)
            mask_batch = mask_batch[:, :-1].contiguous().pin_memory().to(device=device, non_blocking=True)
            y[mask_batch == tokenizer.Ignored.YES] = -1
            segment_batch = segment_batch[:, :-1].contiguous().pin_memory().to(device=device, non_blocking=True)
            state_dict = {'epoch': epoch, 'f_idx': f_idx, 'rg_idx': rg_idx}
            yield x, y, segment_batch, state_dict
    
    def _count_tokens(self, conversations):
        total = 0
        for conversation in conversations:
            token, _ = self.tokeniz.render_conversation(conversation)
            if len(token) > self.seq_len + 1:
                continue
            total += len(token)
        return total
