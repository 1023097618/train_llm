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

from common.report import default_report_dir, write_report
from datagen.synth_t2i_chat import shared_words
from inference.chat_t2i import ChatT2I

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
        ("Draw a bowl of soup on a wooden table.", "add a spoon", ["spoon", "soup"], ["table", "wooden"], [])]

parser = argparse.ArgumentParser(description="behaviour of conversational image generation")
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--stage", type=str, default="t2i_chat_behaviour")
parser.add_argument("--report_dir", type=str, default=None)
args = parser.parse_args()

chat = ChatT2I(args.ckpt_path, args.tokenizer_path)


def ask(messages):
    return chat.reply(messages, top_k=1)  # greedy, the score does not depend on a seed


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
edit_drew = edit_ok = 0
retained = []
for first, change, need, keep, gone in EDIT:
    messages = [{"role": "user", "content": first}]
    reply, prompt = ask(messages)
    if prompt is None:  # the edit needs a first image: give it the plain request as its description
        reply = {"role": "assistant", "content": [{"type": "text", "content": "Here you go."}, {"type": "t2i_prompt", "content": first}]}
    messages += [reply, {"role": "user", "content": change}]
    _, new = ask(messages)
    edit_drew += new is not None
    words = set(re.findall(r"[a-z]+", new.lower())) if new else set()
    ok = new is not None and all(any(x.startswith(w) for x in words) for w in need + keep) and not words & set(gone)
    edit_ok += ok
    if new and prompt:
        retained.append(shared_words(prompt, new))
    print(f"[edit {'ok ' if ok else 'NO '}] {prompt} + {change!r} -> {new}")
lengths = [len(p.split()) for p in first_prompts]
results = {"mean_first_description_words": sum(lengths) / max(1, len(lengths)), "words_retained_by_edit": sum(retained) / max(1, len(retained)),
           "draws_when_asked": drew / len(DRAW), "stays_in_text": quiet / len(NO_DRAW), "edit_draws_again": edit_drew / len(EDIT), "edit_correct": edit_ok / len(EDIT)}
print(results)
path = write_report(args.report_dir or default_report_dir(args.ckpt_path), args.stage, {"ckpt": args.ckpt_path, "results": results})
print(f"-> {path}")
