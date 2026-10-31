"""Caption + VQ token batches for script/train_imagegen.py (shards of script/prepare_t2i_data.py). Each epoch visits the
shards and rows of this worker in a new order; the position is resumable."""
import glob
import os
import random

import numpy as np
import torch
from pyarrow import parquet as pq
from torch.utils.data import IterableDataset, get_worker_info


def shard_files(data_dir, split):
    files = sorted(glob.glob(os.path.join(data_dir, "en256-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no shards under {data_dir}, run `python -m script.prepare_t2i_data` first")
    return files[:-1] if split == "train" else files[-1:]


class T2IDataset(IterableDataset):
    """yields (prefix ids (B, capacity), lengths (B), visual ids (B, 256), state)"""

    def __init__(self, files, tokeniz, batch_size, capacity, rank=0, world_size=1, resume_state_dict=None, seed=0):
        self.files, self.tokeniz, self.batch_size, self.capacity = files, tokeniz, batch_size, capacity
        self.rank, self.world_size, self.resume_state_dict, self.seed = rank, world_size, resume_state_dict, seed

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        shard_id, num_shards = self.rank * nw + wid, self.world_size * nw
        by_file = len(self.files) >= num_shards
        my_files = self.files[shard_id::num_shards] if by_file else self.files
        c = dict((self.resume_state_dict or {}).get(wid) or {"epoch": 0, "file": 0, "row": 0})
        user, end = self.tokeniz.special_token["<|user|>"], self.tokeniz.special_token["<|/user|>"]
        pad = self.tokeniz.special_token["<|/doc|>"]
        pending = []
        while True:
            files = list(my_files)
            random.Random(self.seed * 1000 + c["epoch"]).shuffle(files)
            while c["file"] < len(files):
                table = pq.read_table(files[c["file"]])
                captions, tokens = table.column("caption").to_pylist(), table.column("vq_token_bytes").to_pylist()
                order = list(range(len(captions)))
                random.Random(self.seed * 1000 + c["epoch"] * 31 + c["file"]).shuffle(order)
                while c["row"] < len(order):
                    r = order[c["row"]]
                    c["row"] += 1
                    if not by_file and r % num_shards != shard_id:
                        continue
                    ids = [user] + self.tokeniz.encode_ordinary(captions[r]) + [end]
                    if len(ids) > self.capacity:
                        continue
                    pending.append((ids, np.frombuffer(tokens[r], dtype=np.uint16).astype(np.int64)))
                    if len(pending) == self.batch_size:
                        prefix = torch.tensor([p + [pad] * (self.capacity - len(p)) for p, _ in pending])
                        lengths = torch.tensor([len(p) for p, _ in pending])
                        visual = torch.from_numpy(np.stack([v for _, v in pending]))
                        pending = []
                        yield prefix, lengths, visual, {"worker": wid, **c}
                c["file"] += 1
                c["row"] = 0
            c["epoch"] += 1
            c["file"] = 0
