"""Update the identity conversations to the current model specs (datagen/facts.py) with minimal edits.

usage:
DEEPSEEK_API_KEY=... python -m datagen.update_identity --variant=d20 \
    --src=data/raw/custom/identity_conversations_en_d20.jsonl --dst=data/raw/custom/identity_conversations_en_d20.jsonl
Only conversations that talk about the model's specs / architecture / training are sent to the API.
"""
import argparse
import json
import re

from datagen.deepseek import run_jobs, is_failed
from datagen.facts import facts, ALLOWED_NUMBERS

SELECT = re.compile(r"param|layer|hidden|head|ReLU|activation|architecture|pre-?train|SFT|fine-tun|train(?:ed|ing)|"
                    r"\bGPU\b|32 ?GB|cost|RL\b|reinforcement|DPO|context|token|optimi[sz]er|Muon|embedding", re.I)
OLD_FACTS = {
    # the d24 file is made from the d20 one: its specs and the single GPU are outdated
    "d24": r"0\.56|561|560 ?million|0\.1 ?B\b|0\.1 billion|100 ?million|1\.4 ?(?:B|billion)|0\.48|\b20[- ]layers?\b|\b1280\b|\b10 attention heads|single 32 ?GB|(?<!squared )\bReLU\b(?!²| ?squared|-squared|\^2)",
    "d20": r"0\.56|561|560 ?million|0\.1 ?B\b|0\.1 billion|100 ?million|(?<!squared )\bReLU\b(?!²| ?squared|-squared|\^2)|\b12[- ]layers?\b",
    "d12": r"0\.56|561|560 ?million|0\.1 ?B\b|0\.1 billion|100 ?million|(?<!squared )\bReLU\b(?!²| ?squared|-squared|\^2)|\b20[- ]layers?\b|\b1280\b|32 ?GB",
}
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

PROMPT = """You make MINIMAL edits to identity training conversations of a language model named Guangmingxing.
Some facts in them are outdated. Make every statement consistent with these current facts:

{facts}
Rules:
- Only change sentences that contradict the facts above. Keep everything else verbatim: wording, style, tone,
  analogies, length, structure. If a message needs no change, return it verbatim.
- When a parameter count is mentioned, say the total and add that most of it is embedding lookup tables.
- When the training pipeline is listed, it is pretraining -> SFT -> reinforcement learning on math word problems.
- When the activation is mentioned, it is ReLU² (squared ReLU), not plain ReLU.
- If the user states an outdated fact as something they read, update the user message to the current fact;
  if the user states a wrong premise that the assistant corrects, keep the premise and correct it with the current fact.
- Do not add facts that the conversation did not talk about. Same number of messages, same roles and order, English only.
Return JSON: {{"messages": [{{"role": ..., "content": ...}}, ...]}}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["d12", "d20", "d24"], required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()]
    old = re.compile(OLD_FACTS[args.variant], re.I)
    allowed = ALLOWED_NUMBERS[args.variant]
    selected = [i for i, r in enumerate(rows) if SELECT.search("\n".join(m["content"] for m in r["messages"]))]
    if args.limit:
        selected = selected[:args.limit]
    print(f"{len(rows)} conversations, {len(selected)} sent for update")
    system = PROMPT.format(facts=facts(args.variant))

    def make_messages(i):
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"messages": rows[i]["messages"]}, ensure_ascii=False)}]

    def validate(i, new):
        orig = rows[i]["messages"]
        msgs = new.get("messages")
        if not isinstance(msgs, list) or len(msgs) != len(orig):
            return None, "length"
        if any(not isinstance(b, dict) or b.get("role") != a["role"] or not str(b.get("content", "")).strip() for a, b in zip(orig, msgs)):
            return None, "role"
        text = "\n".join(m["content"] for m in msgs)
        if re.search(r"[一-鿿]", text):
            return None, "cjk"
        for m in msgs:
            if m["role"] == "assistant" and old.search(m["content"]):
                return None, "old_fact:" + old.search(m["content"]).group(0)
        orig_text = "\n".join(m["content"] for m in orig)
        for num in NUMBER.findall(text):
            if num not in orig_text and num not in allowed:
                return None, "new_number:" + num
        return [{"role": m["role"], "content": m["content"].strip()} for m in msgs], None

    results = run_jobs([(i, i) for i in selected], make_messages, validate,
                       args.cache or args.dst + f".{args.variant}.cache.jsonl", temperature=0.3)
    out, failed = [], 0
    sel = set(selected)
    for i, r in enumerate(rows):
        if i not in sel:
            if old.search("\n".join(m["content"] for m in r["messages"] if m["role"] == "assistant")):
                failed += 1  # untouched but outdated (only happens with --limit)
                continue
            out.append(r)
        elif is_failed(results[i]):
            failed += 1
        else:
            out.append({**r, "messages": results[i]})
    with open(args.dst, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print(f"written {len(out)} conversations -> {args.dst}, dropped {failed}")


if __name__ == "__main__":
    main()
