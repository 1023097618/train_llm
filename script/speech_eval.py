"""Does the talker say what the thinker wrote? Word error rate of SenseVoice transcripts against the reply text,
for generated speech and, as the upper bound of this measure, for the dataset's real speech decoded by Mimi.

usage:
python -m script.speech_eval --thinker_ckpt=weight/d24/vlm_rl.ckpt --speech_ckpt=weight/d24/speech_t2a.ckpt \
    --tokenizer_path=weight/d24/tokenizer.pkl --out_dir=weight/d24/report_speech/t2a
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import json
import os
import re

import soundfile as sf
import torch
from pyarrow import parquet as pq

from common import common
from common.data_paths import AUDIO_DATA, AUDIO_MIMI, AUDIO_SENSEVOICE
from inference.speech import MimiDecoder, generate_speech
from model import gpt
from model.speech import NUM_CODEBOOKS, Talker, TalkerConfig
from tasks import speechdata
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="word error rate of generated speech")
parser.add_argument("--thinker_ckpt", type=str, required=True)
parser.add_argument("--speech_ckpt", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=30)
parser.add_argument("--max_chars", type=int, default=300, help="only questions with a short reference answer")
parser.add_argument("--audio_temperature", type=float, default=0.8)
parser.add_argument("--audio_top_k", type=int, default=50)
args = parser.parse_args()


def words(text):
    text = re.sub(r"<\|[^|]*\|>", " ", text).lower()
    return re.findall(r"[a-z0-9']+", text)


def wer(reference, hypothesis):
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        cur = [i]
        for j, h in enumerate(hypothesis, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(reference), 1)


device = "cuda:0"
os.makedirs(args.out_dir, exist_ok=True)
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
ckpt = torch.load(args.thinker_ckpt, map_location="cpu", weights_only=False)
thinker = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), device, tokeniz)
thinker.load_state_dict(common.strip_prefix(ckpt["model_state"]))
thinker.requires_grad_(False).eval()
del ckpt
speech = torch.load(args.speech_ckpt, map_location="cpu", weights_only=False)
talker = Talker(TalkerConfig(**speech["talker_config"]), thinker.config.model_dim, device)
talker.load_state_dict(speech["talker_state"])
talker.eval()
print(f"talker step {speech['step']}, bridge from block {speech['hidden_layer']}")
mimi = MimiDecoder(AUDIO_MIMI, device)

import funasr
asr = funasr.AutoModel(model=str(AUDIO_SENSEVOICE), disable_update=True, device=device)


def transcribe(path):
    return asr.generate(input=path, language="en", use_itn=True)[0]["text"]


rows = pq.read_table(speechdata.shard_files(str(AUDIO_DATA), "t2a", "val")[0]).to_pylist()
results = []
for row in rows:
    if len(results) >= args.num_samples:
        break
    messages = json.loads(row["conversations"])
    if len(messages) != 2 or len(messages[1]["content"]) > args.max_chars or not row["answer_audios"]:
        continue
    n = len(results)
    question, reference = messages[0]["content"], messages[1]["content"]
    # real speech: how well SenseVoice can do at all
    codes = torch.tensor(row["answer_audios"][0], dtype=torch.long).view(-1, NUM_CODEBOOKS).t()
    real_path = os.path.join(args.out_dir, f"{n:02d}_real.wav")
    sf.write(real_path, mimi.decode(codes), mimi.sample_rate)
    real_text = transcribe(real_path)
    real_wer = wer(words(reference), words(real_text))
    # generated speech for the thinker's own reply
    prompt, _ = tokeniz.render_conversation({"source": "eval", "messages": messages[:1]}, last_is_user=True)
    text, gen = generate_speech(thinker, talker, tokeniz, prompt, speech["hidden_layer"],
                                audio_temperature=args.audio_temperature, audio_top_k=args.audio_top_k)
    gen_path = os.path.join(args.out_dir, f"{n:02d}_generated.wav")
    gen_text = ""
    if gen.shape[1] > 0:
        sf.write(gen_path, mimi.decode(gen), mimi.sample_rate)
        gen_text = transcribe(gen_path)
    gen_wer = wer(words(text), words(gen_text))
    real_rate = codes.shape[1] / 12.5 / max(len(reference), 1)
    gen_rate = gen.shape[1] / 12.5 / max(len(text), 1)
    results.append({"question": question, "reply": text, "generated_transcript": gen_text, "generated_wer": gen_wer,
                    "generated_seconds": gen.shape[1] / 12.5, "generated_sec_per_char": gen_rate, "reference": reference,
                    "real_transcript": real_text, "real_wer": real_wer, "real_sec_per_char": real_rate})
    print(f"{n:02d} WER generated {gen_wer:.2f} real {real_wer:.2f} | sec/char generated {gen_rate:.3f} real {real_rate:.3f}")
    if n < 3:
        print("   reply:     ", text[:150])
        print("   generated: ", re.sub(r"<\|[^|]*\|>", "", gen_text)[:150])
        print("   real:      ", re.sub(r"<\|[^|]*\|>", "", real_text)[:150])

mean = lambda key: sum(r[key] for r in results) / len(results)
summary = {k: mean(k) for k in ("generated_wer", "real_wer", "generated_sec_per_char", "real_sec_per_char")}
print(summary)
with open(os.path.join(args.out_dir, "speech_eval.json"), "w", encoding="utf-8") as f:
    json.dump({"step": speech["step"], "summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
