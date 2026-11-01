"""Conversations that teach the chat model when to draw and how to write the description for the image decoder.

- templates: a plain request around a caption of the image generation data, the description repeats it (no API)
- request:   a natural, often vague request, the assistant writes a concrete description
- edit:      the user keeps changing the image, the assistant rewrites the whole description each time
- mixed:     normal chat before or after drawing, questions about the image are answered from its description
- no_draw:   talks about pictures or art without asking for one, the assistant answers and does not draw

The description goes into a {"type": "t2i_prompt"} part, rendered as <|visual prompt|>...<|/visual prompt|>. An edit is a new
full description, not <|edit prompt|>: the decoder only reads text and cannot look at the previous image.

usage:
python -m datagen.synth_t2i_chat --per_kind=0 --templates=60000
DEEPSEEK_API_KEY=... python -m datagen.synth_t2i_chat --per_kind=2000 --templates=60000
"""
import argparse
import glob
import json
import os
import random
import re

from pyarrow import parquet as pq

from common.data_paths import SFT_T2I_CHAT, T2I_DATA
from datagen.deepseek import is_failed, run_jobs

REQUESTS = ["Draw {c}", "Draw me {c}", "Please draw {c}", "Can you draw {c}?", "Could you draw {c}?",
            "Generate an image of {c}", "Generate a picture of {c}", "Can you generate an image of {c}?",
            "Create an image of {c}", "Create a picture of {c}", "Make an image of {c}", "Make me a picture of {c}",
            "I want a picture of {c}", "I'd like an image of {c}", "Show me a picture of {c}", "Show me {c}",
            "Paint {c}", "Sketch {c}", "Picture of {c}", "image of {c}", "draw {c}", "generate {c}"]
REPLIES = ["Here you go.", "Sure, here it is.", "Here is the image.", "Sure.", "Of course.", "Here's your picture.",
           "Okay, drawing it now.", "Here's what I came up with."]

KINDS = {
    "request": ("The user asks for one image in natural words, often vaguely (for example only a subject or a mood). "
                "The assistant draws it right away, without asking questions, choosing sensible concrete details."),
    "edit": ("The user asks for an image, then asks for {edits} changes one after the other, each referring to the previous "
             "image in natural words (\"make it darker\", \"now with two of them\", \"actually, a dog instead\"). Each assistant "
             "turn draws again; its description is complete on its own and keeps everything the user did not ask to change."),
    "mixed": ("Mix drawing with normal chat: {mix}. Answers about an image only use what its description says."),
    "no_draw": ("The user talks about images or art but does not want one generated ({topic}). The assistant answers "
                "helpfully in plain text and does not draw."),
}
MIXES = ["the user chats about a topic first, then asks for a related image",
         "the assistant draws, then the user asks what is in the picture",
         "the assistant draws, then the user asks a general question about the subject",
         "the user asks for an image, comments that they like it, and asks for another different one"]
NO_DRAW = ["how to draw something by hand", "photography tips", "the meaning of a famous painting", "image file formats",
           "how image generators work", "what color goes well with another", "the history of an art movement",
           "a book or movie description"]
STYLES = ["casual", "short and direct", "polite and detailed", "busy, a bit impatient", "non-native English speaker", "lowercase, no punctuation"]

PROMPT = """You write one training conversation for a small assistant model that can generate images.
Scenario: {scenario}
For inspiration only, the subject may loosely relate to: "{seed}"
Rules:
- The user writes like a real person ({style}). English only.
- An assistant turn that draws has a very short "content" (at most one short sentence) and an "image_prompt": one sentence of
  8 to 25 words describing only what is visible (subject, setting, colors, lighting). No camera jargon or quality tags.
- Other assistant turns have only "content" and no "image_prompt".
- Never mention being an AI or a language model. Never apologize for being unable to draw.
Return JSON:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "...", "image_prompt": "..."}}]}}
(the example only shows the format; follow the scenario for the exact turns)"""


def load_captions(rng, limit):
    """first sentences of the English captions, short enough to be typed or written as a description"""
    files = sorted(glob.glob(os.path.join(T2I_DATA, "en256-*.parquet")))[:-1]
    if not files:
        raise FileNotFoundError(f"no shards under {T2I_DATA}, run `python -m script.prepare_t2i_data` first")
    captions = []
    for f in files:
        for caption in pq.read_table(f, columns=["caption"]).column(0).to_pylist():
            sentence = re.split(r"(?<=[.!?])\s", caption.strip(), maxsplit=1)[0]
            if 20 <= len(sentence) <= 160 and not re.match(r"(?i)the (first|second|image|photo|picture)\b", sentence):
                captions.append(sentence)
    rng.shuffle(captions)
    return captions[:limit] if limit else captions


def drawing_turn(text, prompt):
    return {"role": "assistant", "content": [{"type": "text", "content": text}, {"type": "t2i_prompt", "content": prompt}]}


def template_row(caption, rng):
    subject = caption.rstrip(".!? ")
    if not re.match(r"[A-Z]{2}|[A-Z][a-z]+ [A-Z]", subject):  # keeps names and acronyms capitalized
        subject = subject[0].lower() + subject[1:]
    return {"messages": [{"role": "user", "content": rng.choice(REQUESTS).format(c=subject)},
                         drawing_turn(rng.choice(REPLIES), caption)],
            "metadata": {"source": "t2i_chat", "kind": "template"}}


def validate(payload, out):
    msgs = out.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2:
        return None, "structure"
    converted, prompts = [], []
    for i, m in enumerate(msgs):
        role, text = m.get("role"), str(m.get("content") or "").strip()
        if role != ("user" if i % 2 == 0 else "assistant") or not text:
            return None, "turn order"
        prompt = str(m.get("image_prompt") or "").strip()
        if role == "user" or not prompt:
            converted.append({"role": role, "content": text})
            continue
        if not 5 <= len(prompt.split()) <= 35 or len(text.split()) > 20:
            return None, "length"
        prompts.append(prompt)
        converted.append(drawing_turn(text, prompt))
    if re.search(r"[一-鿿]", json.dumps(converted, ensure_ascii=False)):
        return None, "cjk"
    kind = payload["kind"]
    if kind == "no_draw" and prompts:
        return None, "no_draw drew"
    if kind == "request" and len(prompts) != 1:
        return None, "request must draw once"
    if kind == "mixed" and not prompts:
        return None, "mixed must draw"
    if kind == "edit" and (len(prompts) < 2 or len(set(prompts)) < len(prompts) or len(prompts) != len(msgs) // 2):
        return None, "edit must draw a new description each turn"
    return converted, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_kind", type=int, default=2000, help="DeepSeek conversations of each kind, 0 = no API calls")
    parser.add_argument("--templates", type=int, default=60000)
    parser.add_argument("--dst", default=str(SFT_T2I_CHAT))
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    rng = random.Random(0)
    captions = load_captions(rng, 0)
    print(f"{len(captions)} usable captions")
    rows = [template_row(c, rng) for c in captions[:args.templates]]
    jobs = []
    for kind in KINDS:
        for k in range(args.per_kind):
            scenario = KINDS[kind].format(edits=rng.randint(1, 3), mix=rng.choice(MIXES), topic=rng.choice(NO_DRAW))
            jobs.append((f"{kind}-{k}", {"kind": kind, "scenario": scenario, "style": rng.choice(STYLES), "seed": rng.choice(captions)}))
    stats = {"template": len(rows)}
    if jobs:
        def make_messages(p):
            return [{"role": "user", "content": PROMPT.format(scenario=p["scenario"], style=p["style"], seed=p["seed"])}]

        results = run_jobs(jobs, make_messages, validate, args.cache or args.dst + ".cache.jsonl", temperature=1.0)
        for job_id, p in jobs:
            if not is_failed(results[job_id]):
                rows.append({"messages": results[job_id], "metadata": {"source": "t2i_chat", "kind": p["kind"]}})
                stats[p["kind"]] = stats.get(p["kind"], 0) + 1
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"written {len(rows)} conversations -> {args.dst}, per kind {stats}")


if __name__ == "__main__":
    main()
