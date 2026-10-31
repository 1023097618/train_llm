"""Speech training data for train_speech.py from MiniMind-O's English mini sets (script/download_audio.py):

- t2a: text question -> spoken answer (answer_audios: Mimi codes, 8 per 12.5 Hz frame, interleaved)
- a2a: spoken question -> spoken answer; the question audio (mp3 bytes) is encoded once here with the frozen
       SenseVoice encoder into (frames, 512) float16 features (question_audio_embs / question_audio_lens)
Each set is shuffled and written as data/processed/audio/<set>/<set>-NNNNN.parquet, the last shard is the
validation split.

usage:
python -m script.prepare_audio_data --device=cuda:0
"""
import argparse
import io
import os

import numpy as np
import pyarrow as pa
import soundfile as sf
import torch
import torchaudio
from pyarrow import parquet as pq
from torch.utils.data import DataLoader, Dataset

from common.data_paths import AUDIO_DATA, AUDIO_MINIMIND_O, AUDIO_SENSEVOICE

parser = argparse.ArgumentParser(description="speech data for train_speech.py")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--rows_per_shard", type=int, default=5000)
parser.add_argument("--val_rows", type=int, default=1000)
parser.add_argument("--max_rows", type=int, default=0, help="per set, for a quick test")
parser.add_argument("--sets", type=str, default="t2a,a2a")
parser.add_argument("--num_workers", type=int, default=16, help="a2a: processes decoding the spoken questions")
parser.add_argument("--encode_batch", type=int, default=64, help="a2a: spoken questions per SenseVoice forward")
args = parser.parse_args()

AUDIO_EMB_DIM = 512


class QuestionFeatures(Dataset):
    """decodes the spoken questions of one shard and computes their fbank features in DataLoader workers (the CPU part)"""

    def __init__(self, questions, frontend):
        self.questions, self.frontend = questions, frontend

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        out = []
        for audio_bytes in self.questions[i] or []:
            wav, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            x = torch.from_numpy(wav)
            if rate != 16000:
                x = torchaudio.functional.resample(x, rate, 16000)
            x = x.unsqueeze(0)
            out.append(self.frontend(x, torch.tensor([x.shape[1]])))
        return out


def write_a2a_shards(table):
    """the features are too large for one Arrow column (2GB offsets), so each shard is encoded and written on its own;
    finished shards are kept, a rerun continues with the next one"""
    import funasr
    wrapper = funasr.AutoModel(model=str(AUDIO_SENSEVOICE), disable_update=True, device=args.device)
    frontend = wrapper.kwargs["frontend"]
    encoder = wrapper.model.encoder.to(args.device).eval()
    out_dir = AUDIO_DATA / "a2a"
    os.makedirs(out_dir, exist_ok=True)
    shards = shard_indices(table.num_rows)
    autocast = dict(device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.device.startswith("cuda"))
    for n, idx in enumerate(shards):
        path = out_dir / f"a2a-{n:05d}.parquet"
        if path.exists():
            print(f"a2a: shard {n} exists", flush=True)
            continue
        rows = table.take(pa.array(idx))
        data = QuestionFeatures(rows.column("question_audios").to_pylist(), frontend)
        # (row, question) -> feature; encoded in batches of similar length, one at a time spends the time in Python, not on the GPU
        features = {}
        pending = []

        @torch.inference_mode()
        def flush():
            pending.sort(key=lambda item: item[1].shape[0])
            for b0 in range(0, len(pending), args.encode_batch):
                batch = pending[b0:b0 + args.encode_batch]
                fbank = torch.nn.utils.rnn.pad_sequence([f for _, f in batch], batch_first=True).to(args.device)
                fbank_lens = torch.tensor([f.shape[0] for _, f in batch], device=args.device)
                with torch.autocast(**autocast):
                    emb, emb_lens = encoder(fbank, fbank_lens)
                emb = emb.float().cpu().numpy().astype(np.float16)
                assert emb.shape[2] == AUDIO_EMB_DIM, emb.shape
                for (key, _), e, n in zip(batch, emb, emb_lens.tolist()):
                    features[key] = e[:n]
            pending.clear()

        questions_per_row = []
        loader = DataLoader(data, batch_size=None, num_workers=args.num_workers, prefetch_factor=8 if args.num_workers else None)
        for r, row in enumerate(loader):
            questions_per_row.append(len(row))
            pending += [((r, q), fbank[0]) for q, (fbank, _) in enumerate(row)]
            if len(pending) >= args.encode_batch * 16:
                flush()
        flush()
        embs = [[features[(r, q)].tobytes() for q in range(n)] for r, n in enumerate(questions_per_row)]
        lens = [[features[(r, q)].shape[0] for q in range(n)] for r, n in enumerate(questions_per_row)]
        rows = rows.drop_columns(["question_audios"])
        rows = rows.append_column("question_audio_embs", pa.array(embs, type=pa.list_(pa.large_binary())))
        rows = rows.append_column("question_audio_lens", pa.array(lens, type=pa.list_(pa.int32())))
        pq.write_table(rows, str(path) + ".tmp", row_group_size=500)
        os.replace(str(path) + ".tmp", path)
        print(f"a2a: shard {n + 1}/{len(shards)} written", flush=True)
    print(f"a2a: {table.num_rows} rows in {len(shards)} shards -> {out_dir}")


def shard_indices(num_rows):
    """shuffled rows split into shards, the last shard is the validation split"""
    order = np.random.default_rng(0).permutation(num_rows)
    val, train = order[-args.val_rows:], order[:-args.val_rows]
    return [train[i:i + args.rows_per_shard] for i in range(0, len(train), args.rows_per_shard)] + [val]


def write_shards(name, table):
    out_dir = AUDIO_DATA / name
    os.makedirs(out_dir, exist_ok=True)
    shards = shard_indices(table.num_rows)
    val, train = shards[-1], np.concatenate(shards[:-1])
    for n, idx in enumerate(shards):
        pq.write_table(table.take(pa.array(idx)), out_dir / f"{name}-{n:05d}.parquet", row_group_size=500)
    print(f"{name}: {len(train)} train + {len(val)} val rows in {len(shards)} shards -> {out_dir}")


for name in args.sets.split(","):
    table = pq.read_table(AUDIO_MINIMIND_O / f"sft_{name}_mini.parquet")
    if args.max_rows:
        table = table.slice(0, args.max_rows)
    if name == "a2a":
        write_a2a_shards(table)
        continue
    table = table.drop_columns([c for c in ("question_audios",) if c in table.column_names])
    write_shards(name, table)
