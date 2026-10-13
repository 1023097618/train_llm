import os
import json
import glob
import torch
from typing import Literal

from tokenizer import tokenizer
from tasks.basedata import BaseDataset
from tasks.sftdata import SFTDataset, preprocess_conversation


class CustomJson(SFTDataset):
    # 复用 SFTDataset.__iter__，只换数据源。
    # 注意：父类里的 rg_idx 在这里表示「文件内的行偏移」(下一条要读的行号)。
    def __init__(self, data_path, tokeniz: tokenizer.Tokenizer,
                 batch_size, seq_len, device, resume_state_dict=None,buffer_capacity=1000):
        self.tokeniz = tokeniz
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.resume_state_dict = resume_state_dict
        self.buffer_capacity=buffer_capacity

        if os.path.isdir(data_path):
            self.data_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        else:
            self.data_files = sorted(glob.glob(data_path))

    def __len__(self):
        total = 0
        for f in self.data_files:
            with open(f, 'r', encoding='utf-8') as fp:
                total += sum(1 for line in fp if line.strip())
        return total

    def _get_file_row(self, batch_size, epoch, f_idx, line_idx):
        while True:
            while f_idx < len(self.data_files):
                #the jsonl file is small, so load it at once in the memory is ok
                with open(self.data_files[f_idx], 'r', encoding='utf-8') as fp:
                    lines = [line for line in fp if line.strip()]
                start = line_idx
                while start < len(lines):
                    end = min(start + batch_size, len(lines))
                    conversations = [preprocess_conversation(json.loads(lines[k])['messages'])
                                     for k in range(start, end)]
                    start = end
                    yield conversations, epoch, f_idx, start
                line_idx = 0
                f_idx += 1
            f_idx = 0
            epoch += 1
    
    def _count_tokens(self, conversations):
        total = 0
        for conversation in conversations:
            token, _ = self.tokeniz.render_conversation(conversation)
            if len(token) > self.seq_len + 1:
                continue
            total += len(token)
        return total