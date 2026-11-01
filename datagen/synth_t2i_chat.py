"""Conversations that teach the chat model when to draw and how to write the description for the image decoder.

- templates: a plain request around a caption of the image generation data, the description repeats it (no API)
- request2:  a natural, often vague request, the assistant writes a concrete description that always has a setting
- edit2:     the user keeps changing the image, usually without saying that the rest stays; the description is the previous
             one with only the requested change, checked by the words the two have in common
- vague2:    the user asks for a vague change ("cuter", "more dramatic"), the description changes in concrete visible ways
- bare:      only a subject is asked for; each conversation is given another look and place, so the first picture is not always the same
- faithful:  the request has specific details, all of which must reach the description
- replace:   the user changes one attribute; the old value must leave the description, not stand next to the new one
- revert:    after a change the user asks to go back; the description returns to the earlier one
- helpful:   everyday questions (names, lists, advice, explanations) answered usefully, without drawing
- mixed:     normal chat before or after drawing, questions about the image are answered from its description
- no_draw:   talks about pictures or art without asking for one, the assistant answers and does not draw

The description goes into a {"type": "t2i_prompt"} part, rendered as <|visual prompt|>...<|/visual prompt|>. An edit is a new
full description, not <|edit prompt|>: the decoder only reads text and cannot look at the previous image.

usage:
python -m datagen.synth_t2i_chat --per_kind=0
DEEPSEEK_API_KEY=... python -m datagen.synth_t2i_chat --per_kind=3000
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
FULL_SENTENCE_REQUESTS = ["Draw this: {c}", "Generate an image: {c}", "Image prompt: {c}", "Make a picture: {c}",
                          "Can you draw this? {c}", "{c}. Draw that please.", "Please generate this scene: {c}"]
NOUN_PHRASE = re.compile(r"(?i)(a|an|the|two|three|four|five|several|many|some)\s")
REPLIES = ["Here you go.", "Sure, here it is.", "Here is the image.", "Sure.", "Of course.", "Here's your picture.",
           "Okay, drawing it now.", "Here's what I came up with."]

KINDS = {
    "request2": ("The user asks for one image in natural words, often vaguely (for example only a subject or a mood). The "
                 "assistant draws it right away, without asking questions. Even for a bare subject like \"a cat\", its "
                 "description names a specific subject, what it is doing, a concrete setting or background, and the light."),
    "edit2": ("The user asks for an image, then asks for {edits} changes one after the other in their own words, each referring "
              "to the previous image. The kinds of change, in this order: {changes}. {keep} Each assistant turn draws again: "
              "copy the previous description word for word and replace only the words the change is about. The setting and "
              "every detail the user did not mention stay exactly the same, unless the user asks for a new scene."),
    "vague2": ("The user asks for an image, then asks for a vague or subjective change in their own words: {vague}. The assistant "
              "decides what that means visibly and draws again: the description is the previous one, setting included, with a few "
              "concrete new visible details that achieve it. Its reply says in at most 15 words what actually changed."),
    "bare": ("The user asks for a picture of {subject} and gives no other detail. The assistant draws it right away. Its "
             "description shows {subject} in a look and place that suit it and read naturally. Take \"{look}\" and \"{place}\" as inspiration, "
             "and pick something similar if they do not make sense for it. It also says what the subject is doing and what the light is like."),
    "faithful": ("The user asks for one image and gives two to four specific details, such as what the subject wears or holds, "
                 "colors, numbers, the action and the place. The assistant draws it right away. Its description contains every "
                 "detail the user gave, in the user's words, and adds only a little light and setting."),
    "replace": ("The user asks for an image of {old} {thing}, then says to make the {thing} {new}. The assistant draws again: the "
                "description is the previous one word for word, with \"{old}\" replaced by \"{new}\". The word \"{old}\" no longer appears."),
    "revert": ("The user asks for an image, then asks for one change: {changes}. Then the user changes their mind and asks in their own words to "
               "go back to how it was. The assistant draws a third time: its third description is the first description again, word for word."),
    "helpful": ("The user asks about: {helpful}. This has nothing to do with images. The assistant answers directly and usefully, with "
                "real, sensible, specific content, in two to five sentences or a short list, and does not draw."),
    "mixed": ("Mix drawing with normal chat: {mix}. Answers about an image only use what its description says."),
    "no_draw": ("The user talks about images or art but does not want one generated ({topic}). The assistant answers "
                "helpfully in plain text and does not draw."),
}
VAGUE = ["cuter", "more dramatic", "more elegant", "scarier", "more fun", "more peaceful", "more realistic", "fancier", "more colorful",
         "more modern", "older looking", "happier", "sadder", "more mysterious", "more romantic", "cooler", "nicer", "more cozy", "more epic", "less boring"]
SUBJECTS = ["a cat", "a dog", "a bird", "a horse", "a car", "a house", "a tree", "a flower", "a robot", "a boat", "a mountain", "a city", "a person",
            "a rabbit", "a fish", "a castle", "a bicycle", "a cake", "a dragon", "a forest", "a beach", "a lion", "a butterfly", "a train"]
LOOKS = ["black", "white", "gray", "brown", "spotted", "striped", "golden", "very small", "very large", "old", "young", "blue", "red", "green", "pale cream", "dark", "silver", "patterned"]
PLACES = ["a sunny meadow", "a city street", "a snowy yard", "a kitchen", "a garden", "a rocky shore", "a library", "a market", "a forest path", "a rooftop", "a farm",
          "a living room", "a desert", "a riverbank", "a train station", "a park", "a hillside", "a small village", "an old barn", "a harbor", "a cafe", "a beach"]
REQUEST_WORDS = {"draw", "paint", "sketch", "generate", "create", "make", "image", "picture", "photo", "please", "can", "could", "you", "me", "want", "like", "need", "would", "show", "hey", "hi",
                 "some", "that", "this", "it", "for", "to", "i", "id", "love", "see", "something", "get", "be", "who", "which", "are", "has", "have", "their", "his", "her", "there", "so", "very", "thanks", "looks", "look", "just", "one", "scene"}
COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray", "silver", "gold"]
THINGS = ["car", "dress", "hat", "sunglasses", "umbrella", "house", "door", "bicycle", "boat", "shirt", "scarf", "balloon", "chair", "cat", "flower", "backpack", "kite", "sofa"]
HELPFUL = ["suggesting good names for a pet, a band, a shop or a project", "a short list of ideas for a weekend or a gift", "practical advice about cooking, studying or sleep",
           "explaining a simple science or history question", "comparing two everyday options and recommending one", "a friendly greeting and small talk, then a simple question",
           "fixing a small everyday problem at home", "planning a short trip", "explaining a word or a saying", "writing a short message, such as a thank-you note"]
NEW_SCENE = "the user wants a completely new picture of something else"
CHANGES = ["a different color", "a different time of day", "different weather", "a different background", "another location",
           "one more object added", "an object removed", "a different number of the subject", "the subject replaced by another",
           "a different season", "a different mood", "a different art style", "a different pose or action", "a clothing change",
           "a size change of something", "the lighting changed"]
IMPLICIT = "The user just says what to change, the way people do, and never says that the rest should stay."
EXPLICIT = "In one of the requests the user also says that something should stay as it is, in their own words."
SAYS_KEEP = re.compile(r"(?i)\b(keep|keeping|same|unchanged|stay|stays|as it is|as is|don't change|do not change|everything else|the rest)\b")
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
- The user writes like a real person ({style}); the assistant always writes in clear, normal sentences. English only.
- The assistant speaks about "the image" or "the picture"; the word "description" never appears in its "content".
- Vary the wording of the "content" of drawing turns; do not always say "Here you go".
- An assistant turn that draws has a very short "content" (at most one short sentence) and an "image_prompt": one sentence of
  14 to 35 words describing only what is visible: the subject, what it does, the setting or background, colors, lighting.
  No camera jargon or quality tags.
- Other assistant turns have only "content" and no "image_prompt".
- A description has only the subject the user asked for: no extra animals or people they did not ask for.
- A drawing reply is at most 15 words and only states what the description actually says.
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
    # "picture of ..." only reads well in front of something like "a red car", not "in a forest, a man walks"
    requests = REQUESTS if NOUN_PHRASE.match(subject) else FULL_SENTENCE_REQUESTS
    return {"messages": [{"role": "user", "content": rng.choice(requests).format(c=subject)},
                         drawing_turn(rng.choice(REPLIES), caption)],
            "metadata": {"source": "t2i_chat", "kind": "template"}}


def shared_words(previous, new):
    """the share of the previous description's words still in the new one"""
    words = lambda text: set(re.findall(r"[a-z]+", text.lower())) - {"a", "an", "the", "and", "of", "with", "in", "on", "at", "under", "by", "its", "is"}
    a, b = words(previous), words(new)
    return len(a & b) / max(1, len(a))


def keep_cached(kind, messages):
    """conversations generated under earlier rules: a bare subject must stay alone"""
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            if kind == "bare":
                words = set(re.findall(r"[a-z]+", m["content"][1]["content"].lower()))
                subject = re.findall(r"[a-z]+", messages[0]["content"].lower())
                if {w.split()[-1] for w in SUBJECTS} & words - set(subject) & {"cat", "dog", "bird", "horse", "person", "rabbit", "fish", "lion", "butterfly", "dragon"}:
                    return False
    return True


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
        if not 10 <= len(prompt.split()) <= 45 or len(text.split()) > 20:
            return None, "length"
        prompts.append(prompt)
        converted.append(drawing_turn(text, prompt))
    if re.search(r"[一-鿿]", json.dumps(converted, ensure_ascii=False)):
        return None, "cjk"
    kind = payload["kind"]
    if kind == "no_draw" and prompts:
        return None, "no_draw drew"
    if kind == "request2" and len(prompts) != 1:
        return None, "request must draw once"
    if kind == "mixed" and not prompts:
        return None, "mixed must draw"
    if kind != "no_draw" and any(m["role"] == "assistant" and isinstance(m["content"], str) and "description" in m["content"].lower()
                                 for m in converted):
        return None, "mentions the description"
    if kind == "vague2":
        if len(prompts) < 2 or len(prompts) != len(msgs) // 2:
            return None, "vague must draw again"
        for a, b in zip(prompts, prompts[1:]):
            if shared_words(a, b) < 0.5 or len(set(re.findall(r"[a-z]+", b.lower())) - set(re.findall(r"[a-z]+", a.lower()))) < 3:
                return None, "vague must keep the setting and add concrete details"
    if kind == "bare" and (len(prompts) != 1 or payload["subject"].split()[-1] not in prompts[0].lower()):
        return None, "bare must draw the subject"
    if kind == "faithful":
        if len(prompts) != 1:
            return None, "faithful must draw once"
        asked = set(re.findall(r"[a-z]+", msgs[0]["content"].lower())) - REQUEST_WORDS - {"a", "an", "the", "and", "of", "with", "in", "on", "at", "under", "by", "its", "is"}
        if len(asked) < 4 or shared_words(" ".join(asked), prompts[0]) < 0.8:
            return None, "faithful lost what the user asked for"
    if kind == "replace":
        old, new = payload["old"], payload["new"]
        words = [set(re.findall(r"[a-z]+", p.lower())) for p in prompts]
        if len(prompts) != 2 or old not in words[0] or old in words[1] or new not in words[1] or shared_words(prompts[0], prompts[1]) < 0.7:
            return None, "replace must swap the old value for the new"
    if kind == "revert" and (len(prompts) != 3 or shared_words(prompts[0], prompts[2]) < 0.85 or shared_words(prompts[2], prompts[0]) < 0.85 or prompts[1] == prompts[0]):
        return None, "revert must return to the first description"
    if kind == "helpful" and (prompts or len(msgs) != 2 and len(msgs) != 4):
        return None, "helpful does not draw"
    if kind == "edit2":
        if payload["implicit"] and any(m["role"] == "user" and SAYS_KEEP.search(m["content"]) for m in converted):
            return None, "the user said that the rest stays"
        if len(prompts) < 2 or len(set(prompts)) < len(prompts) or len(prompts) != len(msgs) // 2:
            return None, "edit must draw a new description each turn"
        if not payload["new_scene"] and min(shared_words(a, b) for a, b in zip(prompts, prompts[1:])) < 0.6:
            return None, "edit lost too much of the previous description"
    return converted, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_kind", type=int, default=2000, help="DeepSeek conversations of each kind, 0 = no API calls")
    parser.add_argument("--templates", type=int, default=8000, help="few: they teach to repeat the request, not to describe")
    parser.add_argument("--new_per_kind", type=int, default=1700, help="conversations of each of vague, bare, faithful")
    parser.add_argument("--replace", type=int, default=2000)
    parser.add_argument("--revert", type=int, default=1500)
    parser.add_argument("--helpful", type=int, default=4000)
    parser.add_argument("--dst", default=str(SFT_T2I_CHAT))
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    rng = random.Random(0)
    captions = load_captions(rng, 0)
    print(f"{len(captions)} usable captions")
    rows = [template_row(c, rng) for c in captions[:args.templates]]
    jobs = []
    for kind in KINDS:
        counts = {"vague2": args.new_per_kind, "bare": args.new_per_kind, "faithful": args.new_per_kind, "replace": args.replace, "revert": args.revert, "helpful": args.helpful}
        for k in range(counts.get(kind, args.per_kind)):
            edits = rng.randint(1, 3)
            new_scene = kind == "edit2" and rng.random() < 0.07
            changes = ", ".join(rng.sample(CHANGES, edits - 1) + [NEW_SCENE] if new_scene else rng.sample(CHANGES, edits))
            implicit = rng.random() < 0.85
            subject = rng.choice(SUBJECTS)
            scenario = KINDS[kind].format(edits=edits, changes=changes, keep=IMPLICIT if implicit else EXPLICIT, mix=rng.choice(MIXES), topic=rng.choice(NO_DRAW),
                                          vague=rng.choice(VAGUE), subject=subject, look=rng.choice(LOOKS), place=rng.choice(PLACES),
                                          old=(old := rng.choice(COLORS)), new=(new := rng.choice([c for c in COLORS if c != old])), thing=rng.choice(THINGS), helpful=rng.choice(HELPFUL))
            jobs.append((f"{kind}-{k}", {"kind": kind, "scenario": scenario, "style": rng.choice(STYLES), "seed": rng.choice(captions), "new_scene": new_scene, "implicit": implicit, "subject": subject, "old": old, "new": new}))
    stats = {"template": len(rows)}
    if jobs:
        def make_messages(p):
            return [{"role": "user", "content": PROMPT.format(scenario=p["scenario"], style=p["style"], seed=p["seed"])}]

        results = run_jobs(jobs, make_messages, validate, args.cache or args.dst + ".cache.jsonl", temperature=1.0)
        for job_id, p in jobs:
            if not is_failed(results[job_id]) and keep_cached(p["kind"], results[job_id]):
                rows.append({"messages": results[job_id], "metadata": {"source": "t2i_chat", "kind": p["kind"]}})
                stats[p["kind"]] = stats.get(p["kind"], 0) + 1
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"written {len(rows)} conversations -> {args.dst}, per kind {stats}")


if __name__ == "__main__":
    main()
