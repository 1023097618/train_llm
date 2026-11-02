"""Draw the pictures of the conversations of datagen/synth_t2i_chat.py with the image model, so that the chat model trains on seeing what
it drew, and write them for script/train_vlm.py as the "chat" source. A turn that changes a picture gets it from the earlier one like on
the chat page, going back reuses it. The conversations that draw nothing go to t2i_chat_text.jsonl for the text replay.

--merge shuffles the finished chunks into data/processed/vlm/chat/chat-NNNNN.parquet, the last being the validation split, together with
"draw then look" conversations from ALLaVA captions: the first sentence is the description, the question is answered by the whole
caption. Those answers hold details the description lacks, so they need the picture.

usage:
PYTHONPATH=~/train_llm_pylibs/image CUDA_VISIBLE_DEVICES=1 python -m script.render_chat_images --shard=0 --num_shards=2
PYTHONPATH=~/train_llm_pylibs/image CUDA_VISIBLE_DEVICES=2 python -m script.render_chat_images --shard=1 --num_shards=2
python -m script.render_chat_images --merge=1
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import glob
import io
import json
import os
import random
import re
import time

import pyarrow as pa
from PIL import Image
from pyarrow import parquet as pq

from common.data_paths import RAW_CUSTOM, SFT_T2I_CHAT, VLM_DATA

TOKENS_PER_IMAGE = 64
CHUNKS = VLM_DATA / "chat_chunks"
OUT = VLM_DATA / "chat"
QUESTIONS = ["What is in the picture?", "Describe the picture.", "what do you see in this image?", "Can you describe what you drew?", "Tell me about the image in detail.",
             "what's in it?", "Describe it for me.", "What does the picture show?"]
DRAW = ["Draw {c}", "Please draw this: {c}", "Generate an image: {c}", "Make a picture: {c}", "Can you draw this? {c}"]
REPLIES = ["Here it is.", "Here you go.", "Sure, here's the picture.", "Here is the image."]

parser = argparse.ArgumentParser(description="pictures of the drawing conversations")
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--size", type=int, default=384, help="pixels drawn; the chat model sees them at 256")
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--chunk", type=int, default=250)
parser.add_argument("--templates", type=int, default=3000, help="plain one picture conversations to draw")
parser.add_argument("--max_rows", type=int, default=0, help="for a test")
parser.add_argument("--merge", type=int, default=0)
parser.add_argument("--look", type=int, default=20000, help="draw then look conversations")
parser.add_argument("--rows_per_shard", type=int, default=2000)
args = parser.parse_args()


def jpeg(array):
    buffer = io.BytesIO()
    Image.fromarray(array).resize((256, 256), resample=Image.BICUBIC).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def split_rows():
    """drawing conversations to render, conversations without pictures"""
    rng = random.Random(0)
    rows = [json.loads(line) for line in open(SFT_T2I_CHAT, encoding="utf-8")]
    has_picture = lambda r: any(isinstance(m["content"], list) for m in r["messages"])
    templates = [r for r in rows if r["metadata"]["kind"] == "template"]
    rng.shuffle(templates)
    drawing = [r for r in rows if has_picture(r) and r["metadata"]["kind"] != "template"] + templates[:args.templates]
    rng.shuffle(drawing)
    return drawing, [r for r in rows if not has_picture(r)]


def render(row, back, seed):
    """the messages with image parts, and the picture bytes in the order of those parts"""
    messages, pictures, prompts, images = [], {}, {}, []
    for m in row["messages"]:
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            messages.append(m)
            continue
        parts, number = [], None
        for p in m["content"]:
            if p["type"] == "image_id":
                number = int(p["image_ref"])
            else:
                parts.append(p)
        draw = next(p for p in parts if p["type"] in ("t2i_prompt", "i2i_prompt"))
        prompt = draw["content"]
        if draw["type"] == "i2i_prompt":
            source = int(draw["image_ref"])
            if prompts[source] == prompt:
                array = pictures[source]
            else:
                array = back.edit(pictures[source], prompt, prompts[source], seed=seed, size=args.size, steps=args.steps, check=False)
        else:
            array = back.draw(prompt, seed=seed, size=args.size, steps=args.steps)
        pictures[number], prompts[number] = array, prompt
        images.append(jpeg(array))
        messages.append({"role": "assistant", "content": parts + [{"type": "image", "length": TOKENS_PER_IMAGE, "image_ref": str(number)}]})
    return messages, images


def write_chunk(path, conversations, images):
    table = pa.table({"conversations": pa.array(conversations, pa.large_string()), "image_bytes": pa.array(images, pa.list_(pa.large_binary()))})
    pq.write_table(table, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def draw_all():
    from inference.image_backend import ZImageBackend
    back = ZImageBackend()
    drawing, _ = split_rows()
    if args.max_rows:
        drawing = drawing[:args.max_rows]
    os.makedirs(CHUNKS, exist_ok=True)
    chunks = list(range(0, len(drawing), args.chunk))
    t0, done = time.time(), 0
    for c in chunks[args.shard::args.num_shards]:
        path = CHUNKS / f"chunk-{c:07d}.parquet"
        if path.exists():
            continue
        conversations, images = [], []
        for i, row in enumerate(drawing[c:c + args.chunk]):
            messages, pictures = render(row, back, c + i)
            conversations.append(json.dumps(messages, ensure_ascii=False))
            images.append(pictures)
            done += len(pictures)
        write_chunk(path, conversations, images)
        print(f"{path.name}: {done} pictures, {(time.time() - t0) / max(1, done):.2f}s each, chunk {c // args.chunk + 1}/{len(chunks)}")
    print("done")


def caption_rows(rng, limit):
    """draw then look: description = first sentence of an ALLaVA caption, answer = the caption"""
    conversations, images = [], []
    for f in sorted(glob.glob(str(VLM_DATA / "caption" / "caption-*.parquet")))[:-1]:
        table = pq.read_table(f, columns=["conversations", "image_bytes"])
        order = list(range(table.num_rows))
        rng.shuffle(order)
        convs, pics = table.column("conversations"), table.column("image_bytes")
        for r in order[:max(1, limit // 10)]:
            caption = next((m["content"].strip() for m in json.loads(convs[r].as_py()) if m["role"] == "assistant"), "")
            first = re.split(r"(?<=[.!?])\s", caption, maxsplit=1)[0]
            if not 30 <= len(first) <= 200 or len(caption) <= len(first) + 40:
                continue
            messages = [{"role": "user", "content": rng.choice(DRAW).format(c=first.rstrip("."))},
                        {"role": "assistant", "content": [{"type": "text", "content": rng.choice(REPLIES)}, {"type": "t2i_prompt", "content": first},
                                                          {"type": "image", "length": TOKENS_PER_IMAGE, "image_ref": "0"}]},
                        {"role": "user", "content": rng.choice(QUESTIONS)}, {"role": "assistant", "content": caption}]
            conversations.append(json.dumps(messages, ensure_ascii=False))
            images.append([pics[r].as_py()])
            if len(conversations) >= limit:
                return conversations, images
    return conversations, images


def merge():
    rng = random.Random(0)
    drawing, text = split_rows()
    with open(RAW_CUSTOM / "t2i_chat_text.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in text)
    conversations, images = caption_rows(rng, args.look)
    looks = len(conversations)
    files = sorted(glob.glob(str(CHUNKS / "chunk-*.parquet")))
    labelled = 0
    for f in files:
        table = pq.read_table(f)
        chunk = table.column("conversations").to_pylist()
        labels_path = VLM_DATA / "chat_labels" / os.path.basename(f).replace(".parquet", ".json")
        labels = json.load(open(labels_path, encoding="utf-8")) if os.path.exists(labels_path) else {}
        for i, label in labels.items():  # datagen/label_chat_images.py looked at the conversation's last picture
            messages = json.loads(chunk[int(i)])
            messages += [{"role": "user", "content": label["question"]}, {"role": "assistant", "content": label["answer"]}]
            chunk[int(i)] = json.dumps(messages, ensure_ascii=False)
            labelled += 1
        conversations += chunk
        images += table.column("image_bytes").to_pylist()
    order = list(range(len(conversations)))
    rng.shuffle(order)
    os.makedirs(OUT, exist_ok=True)
    shards = [order[i:i + args.rows_per_shard] for i in range(0, len(order), args.rows_per_shard)]
    for n, idx in enumerate(shards):
        write_chunk(OUT / f"chat-{n:05d}.parquet", [conversations[i] for i in idx], [images[i] for i in idx])
    print(f"{len(conversations)} conversations ({looks} draw then look, {labelled} with a question about their picture, {len(files)} chunks of {len(drawing)} drawing), {sum(map(len, images))} pictures, "
          f"{len(shards)} shards -> {OUT}; {len(text)} without pictures -> t2i_chat_text.jsonl")


if args.merge:
    merge()
else:
    draw_all()
