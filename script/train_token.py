import glob
import pyarrow.parquet as pq
import os
import logging
import rustbpe
import argparse
from common import common
from common.data_paths import PRETRAIN_T2T
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
parser=argparse.ArgumentParser(description="train tokenizer")
parser.add_argument("--vocab_size",type=int,default=65500)
parser.add_argument("--save_path",type=str,default="weight/tokenizer.pkl")
parser.add_argument("--seed",type=int,default=42)
parser.add_argument(
    "--data_path", type=str, default=str(PRETRAIN_T2T),
    help="parquet file, glob, or directory containing tokenizer training data",
)
pbar=common.get_pbar()
pbar.total=1
args=parser.parse_args()
SPECIAL_TOKEN=[
    '<|doc|>', #<system> in sft
    '<|/doc|>', #</system> in sft
    '<|user|>',
    '<|/user|>',
    '<|assistant|>',
    '<|/assistant|>',
    '<|tool|>',
    '<|/tool|>',
    '<|tool call|>',
    '<|/tool call|>',
    '<|tool output|>',
    '<|/tool output|>',
    '<|think|>',
    '<|/think|>',
    "<|image|>",
    "<|image pad|>",
    "<|/image|>",
    "<|audio|>",
    "<|audio pad|>",
    "<|/audio|>",
    "<|visual prompt|>",
    "<|/visual prompt|>",
    "<|edit prompt|>",
    "<|/edit prompt|>",
    "<|image ref|>",
    "<|/image ref|>",
    "<|generate image|>",
    "<|/generate image|>"
]
# text to image
# <user>...</user><assistant>...<visual prompt>...</visual prompt><generate image>h w</generate image><image><image ref>0</image ref><image pad>*n</image></assistant>
# image to image
# <user><image><image ref>0</image ref><image pad>*n</image>...</user><assistant>...<edit prompt><image ref>0</image ref>...</edit prompt><generate image>h w</generate image><image><image ref>1</image ref><image pad>*n</image></assistant>
# image to image
# <user>...</user><assistant>...<visual prompt>...</visual prompt><generate image>h w</generate image><image><image ref>0</image ref><image pad>*n</image></assistant>
# <user>...</user><assistant>...<edit prompt><image ref>0</image ref>...</edit prompt><generate image>h w</generate image><image><image ref>1</image ref><image pad>*n</image></assistant>

# image generate model only can see (<visual prompt>...</visual prompt><generate image>h w</generate image>) or (<edit prompt><image ref>0</image ref>...</edit prompt><generate image>h w</generate image> and the original image)
TEXT_CROP=10_000
# BATCH_CROP=2_000_000_000
BATCH_CROP=2_000_000_000
VOCAB_SIZE=args.vocab_size-len(SPECIAL_TOKEN)

if VOCAB_SIZE <= 0:
    raise ValueError("--vocab_size must be larger than the number of special tokens")


def resolve_data_files(data_path):
    path = os.path.expanduser(data_path)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    elif glob.has_magic(path):
        files = sorted(glob.glob(path))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"tokenizer training data does not exist: {data_path}")
    if not files:
        raise FileNotFoundError(f"no parquet files matched: {data_path}")
    unsupported = [p for p in files if not p.endswith(".parquet")]
    if unsupported:
        raise ValueError(f"unsupported tokenizer data file: {unsupported[0]}")
    return files


DATA_FILES = resolve_data_files(args.data_path)
logging.info("Tokenizer training data: %d file(s) matched %s", len(DATA_FILES), args.data_path)

def token_iteration():
    train_len=0
    for path in DATA_FILES:
        pf=pq.ParquetFile(path)
        if "text" not in pf.schema_arrow.names:
            raise ValueError(f"parquet file has no 'text' column: {path}")
        batches = (
            pf.read_row_group(i, columns=["text"]).column("text").to_pylist()
            for i in range(pf.num_row_groups)
        )
        for text_batch in batches:
            for text in text_batch:
                if not isinstance(text, str):
                    continue
                s = text[:TEXT_CROP]
                yield s
                train_len+=len(s.encode("utf-8"))
                pbar.n=min(train_len/BATCH_CROP,1.)
                pbar.refresh()
                if train_len>BATCH_CROP:
                    return



tokenizer=rustbpe.Tokenizer()
# CJK_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}{1,16}|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# 只允许前导“空格”黏词，标点不黏
CJK_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}{1,16}|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
tokenizer.train_from_iterator(iterator=token_iteration(),vocab_size=VOCAB_SIZE,pattern=CJK_PATTERN)
import tiktoken
enc=tiktoken.Encoding(name="mytoken",
                      pat_str=tokenizer.get_pattern(),
                      mergeable_ranks={bytes(k): v for k, v in tokenizer.get_mergeable_ranks()},
                      special_tokens={k:i+VOCAB_SIZE for i,k in enumerate(SPECIAL_TOKEN)})
import pickle
enc.encode("Hello")
save_dir = os.path.dirname(os.path.abspath(args.save_path))
os.makedirs(save_dir, exist_ok=True)
with open(args.save_path,"wb") as f:
    pickle.dump(enc,f)
