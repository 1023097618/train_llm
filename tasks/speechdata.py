"""Speech training data for script/train_speech.py (shards of script/prepare_audio_data.py).

Layout of one assistant turn on the 9 parallel streams (text + 8 codebooks), starting at the turn's <|assistant|>
position s, with T audio frames:

    text:        ... <|assistant|> reply text ... <|/assistant|> eos eos ...
    codebook j:  pad  ... pad  (s + j: start)  (s + j + 1 + i: code i of frame i)  ...  codebook 0 also: stop

The audio runs in lockstep with the text and may be longer than it (the text stream continues with eos). Codebook j
is delayed by j positions; only codebook 0 decides when to stop. Before s the codebook streams may hold a reference
voice (ref_audios) and a speaker token (spk_emb), which condition the voice and are not trained.
Text mask semantics as in tokenizer.render_conversation: mask[i] == NO trains the prediction of token i + 1.
"""
import glob
import json
import os
import random

import numpy as np
import torch
from pyarrow import parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from model.speech import AUDIO_EMB_DIM, NUM_CODEBOOKS, SPK_EMB_DIM, TalkerConfig
from tasks.sftdata import contains_banned_words, preprocess_conversation
from tokenizer import tokenizer

YES, NO = int(tokenizer.Ignored.YES), int(tokenizer.Ignored.NO)


def shard_files(data_dir, name, split):
    files = sorted(glob.glob(os.path.join(data_dir, name, f"{name}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no {name} shards under {data_dir}, run `python -m script.prepare_audio_data` first")
    return files[:-1] if split == "train" else files[-1:]  # last shard = validation


def spoken_messages(conversations, embs, lens, rng, text_prob=0.1):
    """replace each user turn by its SenseVoice features (text_prob of the turns stay text, so typing still works);
    returns the messages and the features in the order of their <|audio pad|> runs"""
    messages, used, turn = [], [], 0
    for m in conversations:
        if m["role"] == "user" and turn < len(embs):
            feats = np.frombuffer(embs[turn], dtype=np.float16).reshape(lens[turn], AUDIO_EMB_DIM)
            turn += 1
            if rng.random() >= text_prob:
                messages.append({"role": "user", "content": [{"type": "audio", "content": feats}]})
                used.append(feats)
                continue
        messages.append(m)
    return messages, used


def layout_turn(tokens, mask, start, end, codes, config: TalkerConfig, eos, ref_codes=None, has_spk=False):
    """the 9 streams of the conversation up to the assistant turn (start, end) with its answer codes (T * 8 flat).
    Returns (ids, masks) as lists of 9 lists, or None if the turn has no complete frame"""
    frames = len(codes) // NUM_CODEBOOKS
    if frames == 0:
        return None
    length = max(start + frames + NUM_CODEBOOKS, end) + 1
    ids = [[eos] * length] + [[config.audio_pad_token] * length for _ in range(NUM_CODEBOOKS)]
    masks = [[YES] * length for _ in range(NUM_CODEBOOKS + 1)]
    ids[0][:end + 1] = tokens[:end + 1]
    masks[0][start:end + 1] = mask[start:end + 1]  # only this turn's text is trained
    for j in range(NUM_CODEBOOKS):
        stream, m = ids[1 + j], masks[1 + j]
        stream[start + j] = config.audio_start_token
        m[start + j] = NO
        for i in range(frames):
            stream[start + j + 1 + i] = codes[i * NUM_CODEBOOKS + j]
            m[start + j + 1 + i] = NO
        if j == 0:
            stream[start + frames + 1] = config.audio_stop_token  # codebook 0 predicts the end of speech
        else:
            m[start + j + frames] = YES  # nothing to predict after the last code
    ref_start = start
    if ref_codes:
        ref_frames = len(ref_codes) // NUM_CODEBOOKS
        ref_start = max(1 if has_spk else 0, start - ref_frames)
        window = start - ref_start
        if window > 0:
            first = ref_frames - window
            for j in range(NUM_CODEBOOKS):
                for i in range(window):
                    ids[1 + j][ref_start + i] = ref_codes[(first + i) * NUM_CODEBOOKS + j]
    spk_used = False
    if has_spk and ref_start > 0:
        for j in range(NUM_CODEBOOKS):
            ids[1 + j][ref_start - 1] = config.audio_spk_token
        spk_used = True
    return ids, masks, spk_used


class SpeechStream:
    """endless turns of one set for one shard of the workers: (ids, masks, feats list, spk list)"""

    def __init__(self, files, shard_id, num_shards, tokeniz, config, seq_len, cursor=None, seed=0):
        self.files, self.shard_id, self.num_shards = files, shard_id, num_shards
        self.tokeniz, self.config, self.seq_len = tokeniz, config, seq_len
        self.cursor = dict(cursor or {"epoch": 0, "file": 0, "row": 0})
        self.seed = seed

    def turns(self, row, row_id):
        rng = random.Random(self.seed * 1000003 + row_id)
        conversations = json.loads(row["conversations"])
        feats = []
        if row.get("question_audio_embs"):
            conversations, feats = spoken_messages(conversations, row["question_audio_embs"], row["question_audio_lens"], rng)
        conversation = preprocess_conversation({"source": "speech", "messages": conversations})
        if not conversation["messages"] or contains_banned_words(conversation):
            return
        tokens, mask = self.tokeniz.render_conversation(conversation)
        special = self.tokeniz.special_token
        start_id, end_id, audio_id = special["<|assistant|>"], special["<|/assistant|>"], special["<|audio|>"]
        ranges, s = [], None
        for i, t in enumerate(tokens):
            if t == start_id:
                s = i
            elif t == end_id:
                ranges.append((s, i))
        answers = row["answer_audios"] or []
        if len(ranges) != len(answers):
            return  # e.g. tool turns: no speech for them in this data
        eos = special["<|/doc|>"]
        spk = row.get("spk_emb")
        for (start, end), codes in zip(ranges, answers):
            # half of the turns get the reference voice as well, like mymultigpt: the voice can be conditioned either way
            ref = row.get("ref_audios") if rng.random() < 0.5 else None
            laid = layout_turn(tokens, mask, start, end, codes, self.config, eos, ref, bool(spk))
            if laid is None or len(laid[0][0]) > self.seq_len + 1:
                continue
            ids, masks, spk_used = laid
            n_questions = sum(1 for t in tokens[:end + 1] if t == audio_id)
            yield ids, masks, feats[:n_questions], [spk] if spk_used else []

    def __iter__(self):
        by_file = len(self.files) >= self.num_shards
        my_files = self.files[self.shard_id::self.num_shards] if by_file else self.files
        while True:
            while self.cursor["file"] < len(my_files):
                rows = pq.read_table(my_files[self.cursor["file"]]).to_pylist()
                while self.cursor["row"] < len(rows):
                    r = self.cursor["row"]
                    self.cursor["row"] += 1
                    if not by_file and r % self.num_shards != self.shard_id:
                        continue
                    yield from self.turns(rows[r], self.cursor["epoch"] * 10000000 + self.cursor["file"] * 100000 + r)
                self.cursor["file"] += 1
                self.cursor["row"] = 0
            self.cursor["epoch"] += 1
            self.cursor["file"] = 0


class SpeechDataset(IterableDataset):
    """yields CPU batches (ids (B,9,T), targets (B,9,T), segments (B,T), features (N,512), speakers (M,192), state)"""

    def __init__(self, sources, tokeniz, config: TalkerConfig, batch_size, seq_len, rank=0, world_size=1,
                 resume_state_dict=None, buffer_capacity=256, seed=0):
        """sources: [(name, files, weight)]"""
        self.sources, self.tokeniz, self.config = sources, tokeniz, config
        self.batch_size, self.seq_len = batch_size, seq_len
        self.rank, self.world_size = rank, world_size
        self.resume_state_dict, self.buffer_capacity, self.seed = resume_state_dict, buffer_capacity, seed

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        shard_id, num_shards = self.rank * nw + wid, self.world_size * nw
        resume = (self.resume_state_dict or {}).get(wid) or {}
        cursors = resume.get("cursors", {})
        streams = [SpeechStream(files, shard_id, num_shards, self.tokeniz, self.config, self.seq_len,
                                cursors.get(name), self.seed) for name, files, _ in self.sources]
        iters = [iter(s) for s in streams]
        names = [name for name, _, _ in self.sources]
        probs = np.array([w for _, _, w in self.sources], dtype=float)
        probs /= probs.sum()
        draws = resume.get("draws", 0)
        rng = np.random.default_rng([self.seed, shard_id, draws])
        eos = self.tokeniz.special_token["<|/doc|>"]
        pad = self.config.audio_pad_token
        buffer = []

        def refill():
            nonlocal draws
            while len(buffer) < self.buffer_capacity:
                draws += 1
                buffer.append(next(iters[rng.choice(len(iters), p=probs)]))

        while True:
            id_rows, mask_rows, seg_rows, feats, spks = [], [], [], [], []
            for _ in range(self.batch_size):
                ids = [[] for _ in range(NUM_CODEBOOKS + 1)]
                masks = [[] for _ in range(NUM_CODEBOOKS + 1)]
                segment = []
                remaining, segment_id = self.seq_len + 1, 0
                while remaining > 0:
                    refill()
                    best = max((j for j in range(len(buffer)) if len(buffer[j][0][0]) <= remaining),
                               key=lambda j: len(buffer[j][0][0]), default=-1)
                    if best == -1:
                        for k in range(NUM_CODEBOOKS + 1):
                            ids[k] += [eos if k == 0 else pad] * remaining
                            masks[k] += [YES] * remaining
                        segment += [segment_id] * remaining
                        remaining = 0
                    else:
                        item_ids, item_masks, item_feats, item_spks = buffer[best]
                        buffer[best] = buffer[-1]
                        buffer.pop()
                        n = len(item_ids[0])
                        for k in range(NUM_CODEBOOKS + 1):
                            ids[k] += item_ids[k]
                            masks[k] += item_masks[k]
                        segment += [segment_id] * n
                        feats += item_feats
                        spks += item_spks
                        remaining -= n
                    segment_id += 1
                id_rows.append(ids)
                mask_rows.append(masks)
                seg_rows.append(segment)
            ids, masks = torch.tensor(id_rows), torch.tensor(mask_rows)
            x = ids[:, :, :-1].contiguous()
            y = ids[:, :, 1:].contiguous()
            y[masks[:, :, :-1] == YES] = -1
            segments = torch.tensor(seg_rows)[:, :-1].contiguous()
            features = torch.from_numpy(np.concatenate(feats)).float() if feats else torch.zeros((0, AUDIO_EMB_DIM))
            speakers = torch.tensor(spks, dtype=torch.float32) if spks else torch.zeros((0, SPK_EMB_DIM))
            state = {"worker": wid, "draws": draws, "cursors": {n: dict(s.cursor) for n, s in zip(names, streams)}}
            yield x, y, segments, features, speakers, state
