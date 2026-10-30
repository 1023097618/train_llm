"""Image-text training data for script/train_vlm.py.

Sources are sampled by weight: image sources stream the shuffled shards of script/prepare_vlm_data.py
(conversations with "<image>" placeholders + JPEG bytes), the optional text source replays the text SFT mixture so
the language ability is kept. Conversations are packed into seq_len + 1 token rows like SFTDataset (best fit), the
images of a row are returned in the order of their <|image pad|> runs.
"""
import glob
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from pyarrow import parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from model.vision import preprocess
from tasks.sftdata import contains_banned_words, preprocess_conversation
from tokenizer import tokenizer


def shard_files(data_dir, name, split):
    files = sorted(glob.glob(os.path.join(data_dir, name, f"{name}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no {name} shards under {data_dir}, run `python -m script.prepare_vlm_data` first")
    return files[:-1] if split == "train" else files[-1:]  # last shard = validation


def to_messages(conversations, tokens_per_image):
    """MiniMind-V/ALLaVA [{"role", "content" with <image>}] -> our messages with image content parts"""
    messages = []
    for m in json.loads(conversations):
        content = m["content"]
        if m["role"] == "user" and "<image>" in content:
            parts = []
            for i, text in enumerate(content.split("<image>")):
                if i > 0:
                    parts.append({"type": "image", "length": tokens_per_image})
                if text.strip():
                    parts.append({"type": "text", "content": text.strip()})
            content = parts
        messages.append({"role": m["role"], "content": content})
    return messages


class ImageTextStream:
    """endless (messages, [image bytes]) of one image source for one shard of the workers"""

    def __init__(self, files, shard_id, num_shards, tokens_per_image, cursor=None):
        self.files, self.shard_id, self.num_shards = files, shard_id, num_shards
        self.tokens_per_image = tokens_per_image
        self.cursor = dict(cursor or {"epoch": 0, "file": 0, "row": 0})

    def __iter__(self):
        # with at least as many files as shards every worker reads its own files, otherwise every num_shards-th row
        by_file = len(self.files) >= self.num_shards
        my_files = self.files[self.shard_id::self.num_shards] if by_file else self.files
        while True:
            while self.cursor["file"] < len(my_files):
                table = pq.read_table(my_files[self.cursor["file"]], columns=["conversations", "image_bytes"])
                convs = table.column("conversations").to_pylist()
                images = table.column("image_bytes").to_pylist()
                while self.cursor["row"] < len(convs):
                    row = self.cursor["row"]
                    self.cursor["row"] += 1
                    if not by_file and row % self.num_shards != self.shard_id:
                        continue
                    yield to_messages(convs[row], self.tokens_per_image), [images[row]]
                self.cursor["file"] += 1
                self.cursor["row"] = 0
            self.cursor["epoch"] += 1
            self.cursor["file"] = 0


class TextStream:
    """endless text-only conversations of a sftdata.TaskMixture"""

    def __init__(self, mixture, shard_id, num_shards, cursor=None):
        self.mixture, self.shard_id, self.num_shards = mixture, shard_id, num_shards
        self.cursor = dict(cursor or {"epoch": 0, "index": shard_id})

    def __iter__(self):
        while True:
            while self.cursor["index"] < len(self.mixture):
                example = self.mixture[self.cursor["index"]]
                self.cursor["index"] += self.num_shards
                yield example["messages"], []
            self.cursor["epoch"] += 1
            self.cursor["index"] = self.shard_id


class VLMDataset(IterableDataset):
    """yields CPU batches (x, y, segment_ids, pixels uint8 (N, 3, S, S), state); state is resumable per worker"""

    def __init__(self, image_sources, tokeniz, batch_size, seq_len, tokens_per_image, image_size,
                 text_mixture=None, text_weight=0.0, rank=0, world_size=1, resume_state_dict=None,
                 buffer_capacity=256, seed=0):
        """image_sources: [(name, files, weight)]"""
        self.image_sources = image_sources
        self.text_mixture, self.text_weight = text_mixture, text_weight
        self.tokeniz = tokeniz
        self.batch_size, self.seq_len = batch_size, seq_len
        self.tokens_per_image, self.image_size = tokens_per_image, image_size
        self.rank, self.world_size = rank, world_size
        self.resume_state_dict = resume_state_dict
        self.buffer_capacity = buffer_capacity
        self.seed = seed

    def render(self, messages, images):
        """(tokens, mask, pixels) or None when filtered / too long"""
        conversation = preprocess_conversation({"source": "vlm", "messages": messages})
        if not conversation["messages"] or contains_banned_words(conversation):
            return None
        image_parts = sum(isinstance(m["content"], list) and sum(p["type"] == "image" for p in m["content"])
                          for m in conversation["messages"])
        if image_parts != len(images):  # e.g. an ALLaVA row with "<image>" in the reply text
            return None
        tokens, mask = self.tokeniz.render_conversation(conversation)
        if len(tokens) > self.seq_len + 1:
            return None
        pixels = preprocess([Image.open(io.BytesIO(b)) for b in images], self.image_size)
        return tokens, mask, pixels

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        shard_id, num_shards = self.rank * nw + wid, self.world_size * nw
        resume = (self.resume_state_dict or {}).get(wid) or {}
        cursors = resume.get("cursors", {})
        streams, weights, names = [], [], []
        for name, files, weight in self.image_sources:
            if weight > 0:
                stream = ImageTextStream(files, shard_id, num_shards, self.tokens_per_image, cursors.get(name))
                streams.append((stream, iter(stream)))
                weights.append(weight)
                names.append(name)
        if self.text_mixture is not None and self.text_weight > 0:
            stream = TextStream(self.text_mixture, shard_id, num_shards, cursors.get("text"))
            streams.append((stream, iter(stream)))
            weights.append(self.text_weight)
            names.append("text")
        probs = np.array(weights) / sum(weights)
        draws = resume.get("draws", 0)
        rng = np.random.default_rng([self.seed, shard_id, draws])  # a resumed worker continues with a new stream
        eos = self.tokeniz.special_token["<|/doc|>"]
        buffer = []

        def refill():
            nonlocal draws
            while len(buffer) < self.buffer_capacity:
                draws += 1
                _, it = streams[rng.choice(len(streams), p=probs)]
                rendered = self.render(*next(it))
                if rendered is not None:
                    buffer.append(rendered)

        while True:
            token_batch, mask_batch, segment_batch, pixel_batch = [], [], [], []
            for _ in range(self.batch_size):
                tokens, mask, segment = [], [], []
                remaining, segment_id = self.seq_len + 1, 0
                while remaining > 0:
                    refill()
                    best = max((j for j in range(len(buffer)) if len(buffer[j][0]) <= remaining),
                               key=lambda j: len(buffer[j][0]), default=-1)
                    if best == -1:
                        tokens += [eos] * remaining
                        mask += [tokenizer.Ignored.YES] * remaining
                        segment += [segment_id] * remaining
                        remaining = 0
                    else:
                        t, m, p = buffer[best]
                        buffer[best] = buffer[-1]
                        buffer.pop()
                        tokens += t
                        mask += m
                        segment += [segment_id] * len(t)
                        pixel_batch.append(p)
                        remaining -= len(t)
                    segment_id += 1
                token_batch.append(tokens)
                mask_batch.append(mask)
                segment_batch.append(segment)
            token_batch, mask_batch = torch.tensor(token_batch), torch.tensor(mask_batch)
            x = token_batch[:, :-1].contiguous()
            y = token_batch[:, 1:].contiguous()
            y[mask_batch[:, :-1] == tokenizer.Ignored.YES] = -1
            segments = torch.tensor(segment_batch)[:, :-1].contiguous()
            pixels = torch.cat(pixel_batch) if pixel_batch else torch.zeros((0, 3, self.image_size, self.image_size), dtype=torch.uint8)
            state = {"worker": wid, "draws": draws,
                     "cursors": {n: dict(s.cursor) for n, (s, _) in zip(names, streams)}}
            yield x, y, segments, pixels, state
