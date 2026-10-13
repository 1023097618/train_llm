import os
import glob
import json
import torch
from typing import Literal
from pyarrow import parquet as pq

from tokenizer import tokenizer
from tasks.basedata import BaseDataset
from torch.utils.data import get_worker_info


def count_file_tokens(data_file, tokeniz: tokenizer.Tokenizer, num_threads=8):
    # 与 __iter__ 一致：每篇文档额外加 <|doc|> 和 <|/doc|> 两个 token
    texts = pq.read_table(data_file, columns=['text'])['text'].to_pylist()
    tokens = tokeniz.tokenizer.encode_ordinary_batch(texts, num_threads=num_threads)
    return sum(len(t) for t in tokens) + 2 * len(texts)


def select_train_files(data_dir, tokeniz: tokenizer.Tokenizer, target_token, cache_name="token_count.json"):
    """按顺序累加训练文件的 token 数，返回刚好满足 target_token 的最短前缀文件列表；不足时报错。
    统计结果缓存在 data_dir/cache_name，文件大小/修改时间/词表大小变化时会重新统计。"""
    data_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.parquet")))[:-1]  # 最后一个文件留给 test
    cache_path = os.path.join(data_dir, cache_name)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    selected, total = [], 0
    for data_file in data_files:
        st = os.stat(data_file)
        key = os.path.basename(data_file)
        meta = {'size': st.st_size, 'mtime': st.st_mtime, 'n_vocab': tokeniz.n_vocab}
        entry = cache.get(key)
        if entry is None or any(entry.get(k) != v for k, v in meta.items()):
            entry = {**meta, 'tokens': count_file_tokens(data_file, tokeniz)}
            cache[key] = entry
            with open(cache_path, 'w') as f:
                json.dump(cache, f, indent=1)
        selected.append(data_file)
        total += entry['tokens']
        print(f"[token count] {key}: {entry['tokens']:,} (total {total:,}/{target_token:,})")
        if total >= target_token:
            return selected, total
    raise RuntimeError(
        f"训练数据不足：{len(data_files)} 个训练文件共 {total:,} token，"
        f"需要 {target_token:,} token（还差 {target_token - total:,}）"
    )


class PretrainDataset(BaseDataset):
    def __init__(self, data_dir, split: Literal["train", "test"], tokeniz: tokenizer.Tokenizer,
                 rank_batch_size, seq_len, resume_state_dict=None,buffer_capacity=1000,rank=0,world_size=1,
                 data_files=None):
        self.tokeniz = tokeniz
        self.rank_batch_size = rank_batch_size
        self.seq_len = seq_len
        self.resume_state_dict = resume_state_dict
        self.buffer_capacity=buffer_capacity
        if data_files is None:
            data_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.parquet")))
            data_files = data_files[:-1] if split == 'train' else data_files[-1:]
        self.data_files = data_files
        self.rank=rank
        self.world_size=world_size

    def __len__(self):
        return sum(pq.ParquetFile(f).metadata.num_rows for f in self.data_files)

    def _get_file_row(self, batch_size, epoch, f_idx, rg_idx,global_rg_idx):
        while True:
            while f_idx < len(self.data_files):
                pq_file = pq.ParquetFile(self.data_files[f_idx])
                while rg_idx < pq_file.num_row_groups:
                    if global_rg_idx % self.num_shards!= self.shard_id:
                        global_rg_idx+=1
                        rg_idx+=1
                        continue
                    row_group = pq_file.read_row_group(rg_idx)['text'].to_pylist()
                    for start in range(0, len(row_group), batch_size):
                        yield row_group[start:start + batch_size], epoch, f_idx, rg_idx,global_rg_idx
                    global_rg_idx+=1
                    rg_idx += 1
                rg_idx = 0
                f_idx += 1
            f_idx = 0
            epoch += 1

    def __iter__(self):
        info=get_worker_info()
        (wid,nw)=(info.id,info.num_workers) if info is not None else (0,1)
        self.shard_id=self.rank*nw+wid
        self.num_shards=self.world_size*nw
        tokeniz, seq_len, rank_batch_size= self.tokeniz, self.seq_len, self.rank_batch_size
        epoch, f_idx, rg_idx, global_rg_idx = 0, 0, 0,0
        resume=(self.resume_state_dict or {}).get(wid)
        if resume:
            epoch = resume['epoch']
            f_idx = resume['f_idx']
            rg_idx = resume['rg_idx']
            global_rg_idx= resume['global_rg_idx']
        text_loader = self._get_file_row(128, epoch, f_idx, rg_idx,global_rg_idx)
        bos = tokeniz.special_token['<|doc|>']
        eos = tokeniz.special_token['<|/doc|>']
        token_buffer = []

        def refill_buffer():
            nonlocal epoch, f_idx, rg_idx,global_rg_idx
            while len(token_buffer) < self.buffer_capacity:
                texts, epoch, f_idx, rg_idx,global_rg_idx = next(text_loader)
                tokens = tokeniz.encode_ordinary(texts)
                tokens = [[bos] + token + [eos] for token in tokens]
                token_buffer.extend(tokens)

        while True:
            token_batch = []
            segment_batch = []
            for _ in range(rank_batch_size):
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
            x = token_batch[:, :-1].contiguous()
            y = token_batch[:, 1:].contiguous()
            segment_batch = segment_batch[:, :-1].contiguous()
            state_dict = {'worker':wid,'epoch': epoch, 'f_idx': f_idx, 'rg_idx': rg_idx,'global_rg_idx':global_rg_idx}
            yield x, y, segment_batch, state_dict
