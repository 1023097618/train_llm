import os
import glob
import torch
from typing import Literal
from pyarrow import parquet as pq

from tokenizer import tokenizer
from tasks.basedata import BaseDataset


class PretrainDataset(BaseDataset):
    def __init__(self, data_dir, split: Literal["train", "test"], tokeniz: tokenizer.Tokenizer,
                 batch_size, seq_len, device, resume_state_dict=None,buffer_capacity=1000):
        self.tokeniz = tokeniz
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.resume_state_dict = resume_state_dict
        self.buffer_capacity=buffer_capacity
        data_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.parquet")))
        self.data_files = data_files[:-1] if split == 'train' else data_files[-1:]

    def __len__(self):
        return sum(pq.ParquetFile(f).metadata.num_rows for f in self.data_files)

    def _get_file_row(self, batch_size, epoch, f_idx, rg_idx):
        while True:
            while f_idx < len(self.data_files):
                pq_file = pq.ParquetFile(self.data_files[f_idx])
                while rg_idx < pq_file.num_row_groups:
                    row_group = pq_file.read_row_group(rg_idx)['text'].to_pylist()
                    for start in range(0, len(row_group), batch_size):
                        yield row_group[start:start + batch_size], epoch, f_idx, rg_idx
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
        bos = tokeniz.special_token['<|doc|>']
        eos = tokeniz.special_token['<|/doc|>']
        token_buffer = []

        def refill_buffer():
            nonlocal epoch, f_idx, rg_idx
            while len(token_buffer) < self.buffer_capacity:
                texts, epoch, f_idx, rg_idx = next(text_loader)
                tokens = tokeniz.encode_ordinary(texts)
                tokens = [[bos] + token + [eos] for token in tokens]
                token_buffer.extend(tokens)

        while True:
            token_batch = []
            segment_batch = []
            for _ in range(batch_size):
                token = []
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
                        best_idx = min(range(len(token_buffer)), key=lambda k: len(token_buffer[k]))
                        temp = token_buffer[best_idx]
                        token.extend(temp[:remaining])
                        segment.extend([segment_id] * remaining)
                        token_buffer[best_idx] = token_buffer[-1]
                        token_buffer.pop()
                        remaining = 0
                    else:
                        temp = token_buffer[best_idx]
                        token.extend(temp)
                        segment.extend([segment_id] * best_len)
                        token_buffer[best_idx] = token_buffer[-1]
                        token_buffer.pop()
                        remaining -= best_len
                    segment_id += 1
                token_batch.append(token)
                segment_batch.append(segment)
            token_batch = torch.tensor(token_batch)
            segment_batch = torch.tensor(segment_batch)
            x = token_batch[:, :-1].contiguous().pin_memory().to(device=device, non_blocking=True)
            y = token_batch[:, 1:].contiguous().pin_memory().to(device=device, non_blocking=True)
            segment_batch = segment_batch[:, :-1].contiguous().pin_memory().to(device=device, non_blocking=True)
            state_dict = {'epoch': epoch, 'f_idx': f_idx, 'rg_idx': rg_idx}
            yield x, y, segment_batch, state_dict
