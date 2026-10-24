"""Synthesize short, direct identity Q&A ("who are you?", "are you ChatGPT?", ...) with DeepSeek.

The existing identity data is mostly long scenario questions, so the model never learned to simply say its name.

usage:
DEEPSEEK_API_KEY=... python -m datagen.synth_identity_qa --variant=d20 --num_calls=50 \
    --dst=data/raw/custom/identity_qa_d20.jsonl
"""
import argparse
import json
import random
import re

from datagen.deepseek import run_jobs, is_failed
from datagen.facts import facts, ALLOWED_NUMBERS

THEMES = [
    "who are you / introduce yourself", "what is your name", "what model are you / which LLM is this",
    "who made / created / trained you", "are you ChatGPT, GPT-4, Claude, Gemini, Llama, Qwen or DeepSeek",
    "are you a human or a bot", "which company are you from", "are you open source, what license",
    "how big are you / how many parameters", "how were you trained", "what can you do", "what can't you do",
    "can you browse the internet / know today's news", "do you remember our previous conversations",
    "which languages do you speak", "what does your name mean", "a plain greeting like hi / hello / hey there",
    "what is your architecture", "how long can our conversation be (context length)", "can you use tools",
    "who is Wang Enyu", "why were you built", "are you reliable / can I trust your answers",
    "a short rude or teasing question about the model itself",
]
# themes where the first answer should state the model's name
NAME_THEMES = set(THEMES[:7]) | {"what does your name mean", "a plain greeting like hi / hello / hey there", "who is Wang Enyu"}
STYLES = ["casual", "formal", "very short (1-4 words)", "with a typo or no punctuation", "curious beginner",
          "skeptical", "a developer", "a student", "a non-native English speaker"]

PROMPT = """You write training conversations for a small open-source language model named Guangmingxing.
Facts (the only facts you may use):

{facts}
Write {n} different short conversations about these topics: {themes}.
- Each conversation has 1 or 2 user turns; every user turn is answered by the assistant.
- User messages vary in style: {styles}. Make the wording of every conversation different.
- The assistant's replies are short (1-3 sentences). {name_rule}
  Neutral, friendly, factual tone, no flattery of the developer, no exclamation spam.
- If the user assumes it is another model (ChatGPT, Claude, ...), politely correct them.
- Only use the facts above; never invent numbers, benchmarks, dates or other models' details.
- English only.
Return JSON: {{"conversations": [{{"messages": [{{"role": "user", "content": ...}}, {{"role": "assistant", "content": ...}}, ...]}}, ...]}}"""

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["d12", "d20", "d24"], required=True)
    parser.add_argument("--num_calls", type=int, default=50, help="10 conversations per call")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    allowed = ALLOWED_NUMBERS[args.variant]
    rng = random.Random(0)
    jobs = []
    for i in range(args.num_calls):
        themes = rng.sample(THEMES, 2)
        jobs.append((i, {"themes": themes, "styles": ", ".join(rng.sample(STYLES, 3)), "needs_name": all(t in NAME_THEMES for t in themes)}))
    fact_text = facts(args.variant)

    def make_messages(p):
        name_rule = ("The first reply starts by stating it is Guangmingxing (developed by Wang Enyu when relevant)."
                     if p["needs_name"] else
                     "Answer the question directly; mention the name Guangmingxing only where it fits naturally.")
        return [{"role": "user", "content": PROMPT.format(facts=fact_text, n=10, themes=" and ".join(p["themes"]),
                                                          styles=p["styles"], name_rule=name_rule)}]

    def check(conv, needs_name):
        msgs = conv.get("messages") if isinstance(conv, dict) else None
        if not isinstance(msgs, list) or not 2 <= len(msgs) <= 4 or len(msgs) % 2:
            return None
        for k, m in enumerate(msgs):
            if not isinstance(m, dict) or m.get("role") != ("user" if k % 2 == 0 else "assistant") or not str(m.get("content", "")).strip():
                return None
        text = "\n".join(m["content"] for m in msgs)
        if re.search(r"[一-鿿]", text) or (needs_name and "Guangmingxing" not in msgs[1]["content"]):
            return None
        if any(n not in allowed for n in NUMBER.findall(text)) or len(msgs[1]["content"].split()) > 90:
            return None
        if re.search(r"\b(?:great|brilliant|genius|visionary) (?:creator|developer)\b|comrade", text, re.I):
            return None
        return [{"role": m["role"], "content": m["content"].strip()} for m in msgs]

    def validate(p, out):
        convs = [c for c in (check(c, p["needs_name"]) for c in out.get("conversations", [])) if c]
        return (convs, None) if len(convs) >= 5 else (None, f"only {len(convs)} valid")

    results = run_jobs(jobs, make_messages, validate, args.cache or args.dst + ".cache.jsonl", temperature=1.0, log_every=10)
    rows, seen = [], set()
    for i, _ in jobs:
        if is_failed(results[i]):
            continue
        for msgs in results[i]:
            key = msgs[0]["content"].lower()
            if key not in seen:  # drop exact duplicate questions
                seen.add(key)
                rows.append({"messages": msgs, "metadata": {"source": "synth_identity_qa", "variant": args.variant}})
    with open(args.dst, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"written {len(rows)} conversations -> {args.dst}")


if __name__ == "__main__":
    main()
