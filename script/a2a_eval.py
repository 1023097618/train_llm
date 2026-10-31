"""Does the model understand speech? Spoken validation questions (SenseVoice features through the audio projector) vs
the same questions as text: how close the replies are, and how well a reply answers against the dataset's reference.
Word overlap only, a rough measure of understanding, not of correctness.

usage:
python -m script.a2a_eval --thinker_ckpt=weight/d24/vlm_rl.ckpt --speech_ckpt=weight/d24/speech_a2a_v2.ckpt \
    --tokenizer_path=weight/d24/tokenizer.pkl --out=weight/d24/report_speech/a2a_v2.json
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import json
import os
import re

import numpy as np
import torch
from pyarrow import parquet as pq

from common import common
from common.data_paths import AUDIO_DATA
from model import gpt
from model.speech import AUDIO_EMB_DIM, AudioProjector
from tasks import speechdata
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="understanding of spoken questions")
parser.add_argument("--thinker_ckpt", type=str, required=True)
parser.add_argument("--speech_ckpt", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--max_tokens", type=int, default=120)
args = parser.parse_args()

device = "cuda:0"
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
ckpt = torch.load(args.thinker_ckpt, map_location="cpu", weights_only=False)
model = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), device, tokeniz)
model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
model.requires_grad_(False).eval()
del ckpt
speech = torch.load(args.speech_ckpt, map_location="cpu", weights_only=False)
proj = AudioProjector(model.config.model_dim).to(device)
proj.load_state_dict(speech["audio_proj_state"])
proj.eval()
audio_pad = tokeniz.special_token["<|audio pad|>"]
end = tokeniz.special_token["<|/assistant|>"]


def words(text):
    return set(re.findall(r"[a-z0-9']+", text.lower())) - set("the a an of to and in is it for on with that this are as be by or at".split())


def overlap(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(len(a | b), 1)


@torch.no_grad()
def reply(content, features=None):
    tokens, _ = tokeniz.render_conversation({"source": "eval", "messages": [{"role": "user", "content": content}]}, last_is_user=True)
    out = []
    # no KV cache: the features only go in on the first forward, rerunning the short prompt is simple and fast enough
    for _ in range(args.max_tokens):
        x = torch.tensor([tokens + out], device=device)
        media = None if features is None else proj.media_embeds(x, features, model.config.model_dim, audio_pad)
        nxt = int(model(x, media_embeds=media).logit[0, -1].argmax())
        if nxt == end:
            break
        out.append(nxt)
    return tokeniz.decode(out)


rows = pq.read_table(speechdata.shard_files(str(AUDIO_DATA), "a2a", "val")[0]).to_pylist()
results = []
for row in rows:
    if len(results) >= args.num_samples:
        break
    messages = json.loads(row["conversations"])
    if len(messages) != 2 or not row.get("question_audio_embs"):
        continue
    question, reference = messages[0]["content"], messages[1]["content"]
    if not isinstance(question, str) or len(question) > 200:
        continue
    feats = np.frombuffer(row["question_audio_embs"][0], dtype=np.float16).reshape(row["question_audio_lens"][0], AUDIO_EMB_DIM)
    spoken = reply([{"type": "audio", "content": feats}], torch.from_numpy(feats.copy()).float().to(device))
    typed = reply(question)
    results.append({"question": question, "reference": reference, "spoken_reply": spoken, "typed_reply": typed,
                    "spoken_vs_typed": overlap(spoken, typed), "spoken_vs_reference": overlap(spoken, reference),
                    "typed_vs_reference": overlap(typed, reference)})
    if len(results) <= 6:
        print("Q:", question[:110]); print("  spoken:", spoken[:130]); print("  typed: ", typed[:130])

# chance: a spoken reply against another question's typed reply
chance = np.mean([overlap(r["spoken_reply"], results[(i + 1) % len(results)]["typed_reply"]) for i, r in enumerate(results)])
summary = {k: float(np.mean([r[k] for r in results])) for k in ("spoken_vs_typed", "spoken_vs_reference", "typed_vs_reference")}
summary["spoken_vs_other_question_typed"] = float(chance)
print(summary)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
