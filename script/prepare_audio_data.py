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
import torch
from pyarrow import parquet as pq

from common.data_paths import AUDIO_DATA, AUDIO_MINIMIND_O, AUDIO_SENSEVOICE

parser = argparse.ArgumentParser(description="speech data for train_speech.py")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--rows_per_shard", type=int, default=5000)
parser.add_argument("--val_rows", type=int, default=1000)
parser.add_argument("--max_rows", type=int, default=0, help="per set, for a quick test")
parser.add_argument("--sets", type=str, default="t2a,a2a")
args = parser.parse_args()

AUDIO_EMB_DIM = 512


def build_sensevoice(model_dir, device):
    """mp3/wav bytes -> (frames, 512) float16 SenseVoice encoder features"""
    import funasr
    import soundfile as sf
    import torchaudio.functional as F
    wrapper = funasr.AutoModel(model=str(model_dir), disable_update=True, device=device)
    frontend = wrapper.kwargs["frontend"]
    encoder = wrapper.model.encoder.to(device).eval()

    @torch.inference_mode()
    def encode(audio_bytes):
        wav, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        x = torch.from_numpy(wav)
        if rate != 16000:
            x = F.resample(x, rate, 16000)
        x = x.unsqueeze(0)
        fbank, fbank_lens = frontend(x, torch.tensor([x.shape[1]]))
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            emb, _ = encoder(fbank.to(device), fbank_lens.to(device))
        emb = emb.squeeze(0).float().cpu().numpy().astype(np.float16)
        assert emb.ndim == 2 and emb.shape[1] == AUDIO_EMB_DIM, emb.shape
        return emb.tobytes(), emb.shape[0]

    return encode


def write_shards(name, table):
    out_dir = AUDIO_DATA / name
    os.makedirs(out_dir, exist_ok=True)
    order = np.random.default_rng(0).permutation(table.num_rows)
    val, train = order[-args.val_rows:], order[:-args.val_rows]
    shards = [train[i:i + args.rows_per_shard] for i in range(0, len(train), args.rows_per_shard)] + [val]
    for n, idx in enumerate(shards):
        pq.write_table(table.take(pa.array(idx)), out_dir / f"{name}-{n:05d}.parquet", row_group_size=500)
    print(f"{name}: {len(train)} train + {len(val)} val rows in {len(shards)} shards -> {out_dir}")


for name in args.sets.split(","):
    table = pq.read_table(AUDIO_MINIMIND_O / f"sft_{name}_mini.parquet")
    if args.max_rows:
        table = table.slice(0, args.max_rows)
    if name == "a2a":
        encode = build_sensevoice(AUDIO_SENSEVOICE, args.device)
        embs, lens = [], []
        for i, questions in enumerate(table.column("question_audios").to_pylist()):
            encoded = [encode(q) for q in (questions or [])]
            embs.append([e for e, _ in encoded])
            lens.append([n for _, n in encoded])
            if i % 2000 == 0:
                print(f"a2a: encoded {i}/{table.num_rows} rows", flush=True)
        table = table.drop_columns(["question_audios"])
        table = table.append_column("question_audio_embs", pa.array(embs, type=pa.list_(pa.binary())))
        table = table.append_column("question_audio_lens", pa.array(lens, type=pa.list_(pa.int32())))
    else:
        table = table.drop_columns([c for c in ("question_audios",) if c in table.column_names])
    write_shards(name, table)
