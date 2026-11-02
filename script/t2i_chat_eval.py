"""Does the chat model draw when asked, stay in text otherwise, and keep the image when changing it? Written by hand, none of
these are in the training data. An edit passes when the new description has the requested words, still has the kept ones and
no longer has the replaced ones. The requests do not say that the rest stays: keeping it is the default.

usage:
CUDA_VISIBLE_DEVICES=1 python -m script.t2i_chat_eval --ckpt_path=weight/d24/chat_t2i.ckpt --tokenizer_path=weight/d24/tokenizer.pkl
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import re

import numpy as np

from common.report import default_report_dir, write_report
from datagen.synth_t2i_chat import shared_words
from inference.chat_t2i import ChatT2I
from inference.image_backend import ZImageBackend

DRAW = ["Draw a red bicycle leaning against a brick wall.", "can you make a picture of a lighthouse at night", "I need an image of two penguins on an iceberg.",
        "generate a castle on a hill", "Please paint a vase of sunflowers.", "show me a picture of a rainy street in Tokyo", "draw me a dragon",
        "Could you create an image of a cozy cabin in the woods?", "a cat wearing sunglasses, make it", "I'd love a picture of a hot air balloon over a desert.",
        "sketch an old steam train", "make me something with a big whale and a tiny boat", "Create a poster-like image of a mountain sunrise.", "picture of a bowl of ramen",
        "can u draw a robot playing guitar", "Generate a photo of a white horse running on a beach.", "draw something peaceful", "I want to see a futuristic city, can you draw it?",
        "Illustrate a fox sleeping under a tree.", "Make an image of a snowy village at Christmas."]
NO_DRAW = ["What is the capital of France?", "How do I draw a cat step by step?", "What's the difference between JPEG and PNG?", "Who painted the Mona Lisa?",
           "Give me three tips for taking better photos.", "What color do you get when you mix blue and yellow?", "Explain how a rainbow forms.", "Write a short poem about the sea.",
           "What is 12 times 12?", "Why do cats purr?", "How does an image generator work?", "Recommend a good book about art history.", "What does impressionism mean?",
           "Hello, how are you today?", "Translate 'good morning' into Spanish."]
# (first request, change, words the new description must have, words it must keep, words that must be gone)
EDIT = [("Draw a red car on a beach.", "make it blue", ["blue"], ["car", "beach"], ["red"]),
        ("Draw a dog in a park.", "now make it a cat", ["cat"], ["park"], ["dog"]),
        ("Draw a cabin in a forest during the day.", "change it to night time", ["night"], ["cabin", "forest"], []),
        ("Generate a picture of a boat on a lake.", "add a mountain in the background", ["mountain"], ["boat", "lake"], []),
        ("Draw a woman with a yellow umbrella.", "make the umbrella green", ["green", "umbrella"], ["woman"], ["yellow"]),
        ("Make an image of a castle in summer.", "make it winter with snow", ["snow"], ["castle"], ["summer"]),
        ("Create an image of an apple on a table.", "put three apples instead", ["three", "apple"], ["table"], []),
        ("Draw an old man reading a book.", "make him sit on a bench in a garden", ["bench", "garden"], ["man", "book"], []),
        ("Draw a house by a river.", "make it look like a watercolor painting", ["watercolor"], ["house", "river"], []),
        ("Draw a yellow bird on a branch.", "remove the branch and put it on a fence", ["fence"], ["bird"], ["branch"]),
        ("draw a picture of a cat", "change the cat to orange", ["orange", "cat"], [], []),
        ("draw a cat in a kitchen", "make the cat more cute", ["cat", "kitchen"], [], []),
        ("Draw a man in a blue shirt standing in a library.", "give him a hat", ["hat", "blue", "library"], ["man"], []),
        ("Draw a white horse in a field at sunset.", "make it a black horse", ["black", "horse"], ["field", "sunset"], ["white"]),
        ("Draw a bowl of soup on a wooden table.", "add a spoon", ["spoon", "soup"], ["table", "wooden"], []),
        ("Draw a cozy cabin in the snow.", "make it night with northern lights", ["northern", "night"], ["cabin", "snow"], []),
        ("Draw a lake with mountains behind it.", "add a rainbow", ["rainbow"], ["lake", "mountain"], []),
        ("Draw a city street at dusk.", "add fireworks in the sky", ["firework"], ["street"], [])]

# a vague change must change the description; the rest of it must stay
VAGUE = [("Draw a dog on a sofa in a living room.", "make it cuter", ["sofa"]), ("Draw a house on a hill.", "make it scarier", ["house"]),
         ("Draw a woman walking in a city at night.", "make it more elegant", ["woman"]), ("Draw a cat in a kitchen.", "make the cat more cute", ["kitchen"]),
         ("Draw a car on a road.", "make it cooler", ["car"]), ("Draw a garden with a bench.", "make it more peaceful", ["bench"]),
         ("Draw a castle by a lake.", "make it more dramatic", ["castle", "lake"]), ("Draw a bowl of soup on a table.", "make it look nicer", ["soup"])]
# (request, words the first description must have)
FAITHFUL = [("Draw a cat wearing sunglasses.", ["cat", "sunglasses"]), ("Draw a dog in a park.", ["dog", "park"]), ("Draw three red balloons above a green field.", ["three", "red", "balloon", "green", "field"]),
            ("Draw a boy holding a yellow kite on a beach.", ["boy", "yellow", "kite", "beach"]), ("Draw a whale next to a tiny boat.", ["whale", "tiny", "boat"]),
            ("Draw an astronaut riding a horse on the moon.", ["astronaut", "horse", "moon"]), ("Draw a cat in a kitchen.", ["cat", "kitchen"]), ("Draw a rainy street in Tokyo at night.", ["rain", "street", "tokyo", "night"]),
            ("Draw a robot playing guitar on a stage.", ["robot", "guitar", "stage"]), ("Draw a chef in a white hat cooking pasta.", ["chef", "white", "hat", "pasta"])]
# a replaced attribute must leave the description: (request, change, word that must appear, word that must be gone)
REPLACE = [("Draw a cat wearing black sunglasses on a beach.", "make the sunglasses red", "red", "black"), ("Draw a green car on a road.", "make the car yellow", "yellow", "green"),
           ("Draw a girl in a blue dress in a garden.", "make the dress white", "white", "blue"), ("Draw a white house on a hill.", "paint the house pink", "pink", "white"),
           ("Draw a brown horse in a field.", "make it gray", "gray", "brown"), ("Draw a red umbrella on a beach.", "make the umbrella purple", "purple", "red")]
REVERT = [("Draw a red fox in a forest.", "change the fox to a white wolf", "actually go back to the fox", "fox", "wolf"), ("Draw a cabin in the snow.", "make it a beach house", "no, go back to the cabin", "cabin", "beach"),
          ("Draw a dog on a sofa.", "make it a cat", "I liked the dog better, change it back", "dog", "cat"), ("Draw a boat on a lake.", "make it a submarine", "undo that", "boat", "submarine")]
HELPFUL = [("What's a good name for a golden retriever?", ["buddy", "max", "sunny", "goldie", "honey", "bailey", "charlie", "cooper", "daisy", "luna", "bella", "rusty", "amber", "duke", "sandy", "maple", "ginger"]),
           ("What are the three primary colors?", ["red", "blue"]), ("How many days are in a leap year?", ["366"]), ("What should I pack for a beach trip? Give me a few items.", ["sunscreen", "towel"]),
           ("What does a bee make?", ["honey"]), ("What is the largest planet in our solar system?", ["jupiter"])]
# does it see the picture? The description in the conversation hides the answer, the picture shows it; a gray picture is the baseline
SEEING = [("draw a car", "A car parked on a quiet road.", "A bright red car parked on a quiet road.", "What color is the car?", "red"),
          ("draw a bird", "A bird on a branch.", "A bright blue bird on a branch.", "What color is the bird?", "blue"),
          ("draw a door", "A door in a stone wall.", "A bright green door in a stone wall.", "What color is the door?", "green"),
          ("draw a balloon", "A balloon in the sky.", "A bright yellow balloon in the sky.", "What color is the balloon?", "yellow"),
          ("draw a flower", "A flower in a pot.", "A purple flower in a pot.", "What color is the flower?", "purple"),
          ("draw an umbrella", "An umbrella on a beach.", "A bright orange umbrella on a beach.", "What color is the umbrella?", "orange"),
          ("draw a house", "A house on a hill.", "A pink house on a hill.", "What color is the house?", "pink"),
          ("draw some animal on a farm", "An animal standing on a farm.", "A cow standing on a farm.", "What animal is in the picture?", "cow"),
          ("draw an animal in the forest", "An animal in a forest.", "A deer in a forest.", "What animal is in the picture?", "deer"),
          ("draw a vehicle on a street", "A vehicle on a street.", "A yellow school bus on a street.", "What vehicle is in the picture?", "bus")]
BARE = ["draw a cat", "draw a dog", "draw a house", "draw a car", "draw a bird", "draw a flower"]



def FIRST(request):
    """stands in when the model did not draw the first picture: the later turn still needs one to refer to"""
    return {"role": "assistant", "content": [{"type": "text", "content": "Here you go."}, {"type": "t2i_prompt", "content": request},
                                             {"type": "image", "length": 64, "image_ref": "0"}]}


def edited_ref(message):
    """the picture an <|edit prompt|> changes, or None when the turn drew from scratch"""
    return next((int(p["image_ref"]) for p in message["content"] if p["type"] == "i2i_prompt"), None)


parser = argparse.ArgumentParser(description="behaviour of conversational image generation")
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--stage", type=str, default="t2i_chat_behaviour")
parser.add_argument("--top_k", type=int, default=2, help="1 is greedy; above it every case runs --repeats times, since sampling differs per run")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--report_dir", type=str, default=None)
args = parser.parse_args()

chat = ChatT2I(args.ckpt_path, args.tokenizer_path)
back = ZImageBackend()
drawn = {}


def picture_of(prompt):
    if prompt not in drawn:
        drawn[prompt] = back.draw(prompt, seed=0, size=384, steps=6)
    return drawn[prompt]


def pictures_of(messages):
    """the pictures of a conversation, drawn from their descriptions"""
    out = {}
    for m in messages:
        if isinstance(m["content"], list):
            parts = {p["type"]: p for p in m["content"]}
            if "image" in parts:
                out[int(parts["image"]["image_ref"])] = picture_of((parts.get("i2i_prompt") or parts["t2i_prompt"])["content"])
    return out


def ask(messages):
    return chat.reply(messages, pictures_of(messages), top_k=args.top_k)


if args.top_k > 1:
    DRAW, NO_DRAW, EDIT, VAGUE, FAITHFUL, REPLACE, REVERT, HELPFUL = (x * args.repeats for x in (DRAW, NO_DRAW, EDIT, VAGUE, FAITHFUL, REPLACE, REVERT, HELPFUL))
drew = 0
first_prompts = []
for text in DRAW:
    _, prompt = ask([{"role": "user", "content": text}])
    drew += prompt is not None
    if prompt:
        first_prompts.append(prompt)
    print(f"[draw {'ok ' if prompt else 'NO '}] {text} -> {prompt}")
quiet = 0
for text in NO_DRAW:
    reply, prompt = ask([{"role": "user", "content": text}])
    quiet += prompt is None
    print(f"[text {'ok ' if prompt is None else 'NO '}] {text} -> {prompt or reply['content'][:1]}")
edit_drew = edit_ok = used_edit = 0
retained = []
for first, change, need, keep, gone in EDIT:
    messages = [{"role": "user", "content": first}]
    reply, prompt = ask(messages)
    if prompt is None:  # the edit needs a first image: give it the plain request as its description
        reply = FIRST(first)
    messages += [reply, {"role": "user", "content": change}]
    answer, new = ask(messages)
    edit_drew += new is not None
    used_edit += edited_ref(answer) == 0
    words = set(re.findall(r"[a-z]+", new.lower())) if new else set()
    ok = new is not None and all(any(x.startswith(w) for x in words) for w in need + keep) and not words & set(gone)
    edit_ok += ok
    if new and prompt:
        retained.append(shared_words(prompt, new))
    print(f"[edit {'ok ' if ok else 'NO '}] {prompt} + {change!r} -> {new}")
vague_changed = vague_ok = 0
for first, change, keep in VAGUE:
    messages = [{"role": "user", "content": first}]
    reply, prompt = ask(messages)
    if prompt is None:
        reply, prompt = FIRST(first), first
    messages += [reply, {"role": "user", "content": change}]
    _, new = ask(messages)
    changed = bool(new) and len(set(re.findall(r"[a-z]+", new.lower())) - set(re.findall(r"[a-z]+", prompt.lower()))) >= 2
    vague_changed += changed
    ok = changed and all(w in new.lower() for w in keep)
    vague_ok += ok
    print(f"[vague {'ok ' if ok else 'NO '}] {prompt} + {change!r} -> {new}")
faithful = 0
for text, need in FAITHFUL:
    _, prompt = ask([{"role": "user", "content": text}])
    ok = bool(prompt) and all(w in prompt.lower() for w in need)
    faithful += ok
    print(f"[faith {'ok ' if ok else 'NO '}] {text} -> {prompt}")
distinct = []
for text in BARE:  # with the page's sampling
    prompts = {chat.reply([{"role": "user", "content": text}], top_k=max(args.top_k, 2))[1] for _ in range(5)}
    distinct.append(len(prompts) / 5)
    print(f"[bare] {text}: " + " || ".join(str(p)[:70] for p in list(prompts)[:3]))
replaced = 0
for first, change, new_word, old_word in REPLACE:
    messages = [{"role": "user", "content": first}]
    reply, prompt = ask(messages)
    if prompt is None:
        reply = FIRST(first)
    messages += [reply, {"role": "user", "content": change}]
    _, new = ask(messages)
    words = set(re.findall(r"[a-z]+", new.lower())) if new else set()
    ok = new_word in words and old_word not in words
    replaced += ok
    print(f"[swap {'ok ' if ok else 'NO '}] {change!r} -> {new}")
reverted = reverted_to_first = 0
for first, change, back, want, gone in REVERT:
    messages = [{"role": "user", "content": first}]
    for text in (change, back):
        reply, prompt = ask(messages)
        if prompt is None:
            reply = FIRST(first)
        messages += [reply, {"role": "user", "content": text}]
    answer, new = ask(messages)
    ok = bool(new) and want in new.lower() and gone not in new.lower()
    reverted += ok
    reverted_to_first += edited_ref(answer) == 0
    print(f"[back {'ok ' if ok else 'NO '}] {back!r} -> {new}")
helpful = 0
for text, any_of in HELPFUL:
    reply, prompt = ask([{"role": "user", "content": text}])
    said = " ".join(p["content"] for p in reply["content"] if p["type"] == "text").lower()
    ok = prompt is None and any(w in said for w in any_of)
    helpful += ok
    print(f"[help {'ok ' if ok else 'NO '}] {text} -> {said[:170]}")
seen = {"picture": 0, "gray": 0}
for case in SEEING * (args.repeats if args.top_k > 1 else 1):
    request, shown, real, question, answer = case
    messages = [{"role": "user", "content": request},
                {"role": "assistant", "content": [{"type": "text", "content": "Here you go."}, {"type": "t2i_prompt", "content": shown}, {"type": "image", "length": 64, "image_ref": "0"}]},
                {"role": "user", "content": question}]
    for name, picture in (("picture", picture_of(real)), ("gray", np.full((256, 256, 3), 128, dtype=np.uint8))):
        reply, _ = chat.reply(messages, {0: picture}, top_k=args.top_k)
        said = " ".join(p["content"] for p in reply["content"] if p["type"] == "text")
        hit = answer in said.lower()
        seen[name] += hit
        print(f"[see {name:7s} {'ok ' if hit else 'NO '}] {question} ({answer}) -> {said[:110]}")
lengths = [len(p.split()) for p in first_prompts]
n_seeing = len(SEEING) * (args.repeats if args.top_k > 1 else 1)
results = {"sees_with_picture": seen["picture"] / n_seeing, "sees_with_gray": seen["gray"] / n_seeing, "edit_uses_edit_prompt": used_edit / len(EDIT), "revert_refers_to_first": reverted_to_first / len(REVERT), "replace": replaced / len(REPLACE), "revert": reverted / len(REVERT), "helpful": helpful / len(HELPFUL), "vague_changed": vague_changed / len(VAGUE), "vague_correct": vague_ok / len(VAGUE), "faithful": faithful / len(FAITHFUL), "bare_distinct": sum(distinct) / len(distinct),
           "mean_first_description_words": sum(lengths) / max(1, len(lengths)), "words_retained_by_edit": sum(retained) / max(1, len(retained)),
           "draws_when_asked": drew / len(DRAW), "stays_in_text": quiet / len(NO_DRAW), "edit_draws_again": edit_drew / len(EDIT), "edit_correct": edit_ok / len(EDIT)}
print(results)
path = write_report(args.report_dir or default_report_dir(args.ckpt_path), args.stage, {"ckpt": args.ckpt_path, "results": results})
print(f"-> {path}")
