import argparse
import json
import os
import pickle
import torch
from safetensors.torch import save_file
from common import common

parser=argparse.ArgumentParser()
parser.add_argument("--ckpt_path",type=str,default="weight/sft_model.ckpt",help="path of model checkpoint")
parser.add_argument("--out_dir",type=str,default="weight_vllm",help="output folder (default: <ckpt name> next to the ckpt)")
parser.add_argument("--tokenizer_path",type=str,default="weight/tokenizer.pkl",help="path of tiktoken tokenizer.pkl")
parser.add_argument("--dtype",type=str,default=None,choices=["fp32","bf16","fp16"],help="cast floating tensors, keep original dtype if not set")
args=parser.parse_args()

out_dir=args.out_dir or os.path.splitext(args.ckpt_path)[0]
os.makedirs(out_dir,exist_ok=True)

resume=torch.load(args.ckpt_path,"cpu",weights_only=False)
state_dict=common.strip_prefix(resume['model_state'])
# cos/sin are rotary buffers rebuilt at init, not real weights
state_dict={k:v for k,v in state_dict.items() if k not in {"cos","sin"}}

dtype={"fp32":torch.float32,"bf16":torch.bfloat16,"fp16":torch.float16}.get(args.dtype)
# safetensors refuses tensors sharing storage, so clone each one
tensors={}
for k,v in state_dict.items():
    if dtype is not None and v.is_floating_point():
        v=v.to(dtype)
    tensors[k]=v.detach().contiguous().clone()

config=resume['model_config']
save_file(tensors,os.path.join(out_dir,"model.safetensors"),metadata={"format":"pt","model_config":json.dumps(config)})

total=sum(v.numel() for v in tensors.values())
print(f"exported {len(tensors)} tensors ({total/1e6:.1f}M params) -> {out_dir}")

# ---- tokenizer: tiktoken pkl -> HF tokenizer.json (what vLLM loads) ----
with open(args.tokenizer_path,"rb") as f:
    enc=pickle.load(f)
ranks=enc._mergeable_ranks
special=enc._special_tokens

def bytes_to_unicode():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs=bs[:]
    n=0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n+=1
    return dict(zip(bs,map(chr,cs)))
byte_encoder=bytes_to_unicode()
def to_str(b:bytes):
    return "".join(byte_encoder[x] for x in b)

def bpe_split(token:bytes,max_rank:int):
    # re-run tiktoken's bpe using only merges ranked below the token, leaving the last pair
    parts=[bytes([b]) for b in token]
    while True:
        best=None
        for i in range(len(parts)-1):
            r=ranks.get(parts[i]+parts[i+1])
            if r is not None and r<max_rank and (best is None or r<best[0]):
                best=(r,i)
        if best is None:
            break
        i=best[1]
        parts=parts[:i]+[parts[i]+parts[i+1]]+parts[i+2:]
    return parts

vocab={to_str(b):r for b,r in ranks.items()}
merges=[]
for b,r in sorted(ranks.items(),key=lambda x:x[1]):
    if len(b)==1:
        continue
    parts=bpe_split(b,r)
    assert len(parts)==2,f"token {b} rank {r} can not be split into one merge"
    merges.append(f"{to_str(parts[0])} {to_str(parts[1])}")

tokenizer_json={
    "version":"1.0",
    "truncation":None,
    "padding":None,
    "added_tokens":[
        {"id":i,"content":s,"single_word":False,"lstrip":False,"rstrip":False,"normalized":False,"special":True}
        for s,i in sorted(special.items(),key=lambda x:x[1])
    ],
    "normalizer":None,
    "pre_tokenizer":{"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":enc._pat_str},"behavior":"Isolated","invert":False},
        {"type":"ByteLevel","add_prefix_space":False,"trim_offsets":True,"use_regex":False},
    ]},
    "post_processor":None,
    "decoder":{"type":"ByteLevel","add_prefix_space":True,"trim_offsets":True,"use_regex":True},
    "model":{"type":"BPE","dropout":None,"unk_token":None,"continuing_subword_prefix":None,"end_of_word_suffix":None,
             "fuse_unk":False,"byte_fallback":False,"ignore_merges":True,"vocab":vocab,"merges":merges},
}
with open(os.path.join(out_dir,"tokenizer.json"),"w",encoding="utf-8") as f:
    json.dump(tokenizer_json,f,ensure_ascii=False)

# mirrors Tokenizer.render_conversation for text-only chats
chat_template=(
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}"
    "<|doc|>{{ message['content'] or '' }}"
    "{%- if tools -%}<|tool|>{{ tools | tojson }}<|/tool|>{%- endif -%}"
    "<|/doc|>"
    "{%- elif message['role'] == 'user' -%}"
    "<|user|>{{ message['content'] }}<|/user|>"
    "{%- elif message['role'] == 'assistant' -%}"
    "<|assistant|>"
    "{%- if message.get('reasoning_content') -%}<|think|>{{ message['reasoning_content'] }}<|/think|>{%- endif -%}"
    "{{ message['content'] or '' }}"
    "{%- if message.get('tool_calls') -%}<|tool call|>{{ message['tool_calls'] if message['tool_calls'] is string else message['tool_calls'] | tojson }}<|/tool call|>{%- endif -%}"
    "<|/assistant|>"
    "{%- elif message['role'] == 'tool' -%}"
    "<|tool output|>{{ message['content'] }}<|/tool output|>"
    "{%- if not loop.last and messages[loop.index0 + 1]['role'] == 'user' -%}<|assistant|><|/assistant|>{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
)
eos="<|/assistant|>"
tokenizer_config={
    "tokenizer_class":"PreTrainedTokenizerFast",
    "model_max_length":config.get("seq_len",2048),
    "bos_token":None,
    "eos_token":eos,
    "pad_token":eos,
    "unk_token":None,
    "clean_up_tokenization_spaces":False,
    "add_bos_token":False,
    "add_eos_token":False,
    "additional_special_tokens":[s for s in special if s!=eos],
    "chat_template":chat_template,
}
with open(os.path.join(out_dir,"tokenizer_config.json"),"w",encoding="utf-8") as f:
    json.dump(tokenizer_config,f,ensure_ascii=False,indent=2)
with open(os.path.join(out_dir,"special_tokens_map.json"),"w",encoding="utf-8") as f:
    json.dump({"eos_token":eos,"pad_token":eos,"additional_special_tokens":tokenizer_config["additional_special_tokens"]},f,ensure_ascii=False,indent=2)
# ---- config: HF style fields so vLLM can build the model ----
# model_type "llama" only lets transformers' AutoConfig parse the file,
# the architecture itself must be registered to vLLM as a custom model
head_dim=config["model_dim"]//config["query_num"]
pattern=config["window_pattern"]
layer_types=["sliding_attention" if pattern[i%len(pattern)]=="S" else "full_attention" for i in range(config["n_layer"])]
hf_config={
    "architectures":["NanoGPTForCausalLM"],
    "model_type":"llama",
    "hidden_size":config["model_dim"],
    "intermediate_size":4*config["model_dim"],
    "num_hidden_layers":config["n_layer"],
    "num_attention_heads":config["query_num"],
    "num_key_value_heads":config["kv_num"],
    "head_dim":head_dim,
    "hidden_act":"relu",
    "mlp_bias":False,
    "attention_bias":False,
    # rms_norm has no weight and no eps, activations are bf16 so torch uses finfo(bf16).eps
    "rms_norm_eps":torch.finfo(torch.bfloat16).eps,
    "max_position_embeddings":config["seq_len"],
    "rope_theta":10000.0,
    "rope_scaling":None,
    "vocab_size":config["vocab_size"],
    # lm_head has vocab_size rows but logits are cut to the tokenizer size
    "output_vocab_size":len(vocab)+len(special),
    "tie_word_embeddings":False,
    "sliding_window":config["seq_len"]//4 if "S" in pattern else None,
    "layer_types":layer_types,
    "bos_token_id":None,
    "eos_token_id":special[eos],
    "pad_token_id":special[eos],
    "dtype":"bfloat16",
    "torch_dtype":"bfloat16",
    "nanogpt_config":config,
}
with open(os.path.join(out_dir,"config.json"),"w") as f:
    json.dump(hf_config,f,indent=2)
with open(os.path.join(out_dir,"generation_config.json"),"w") as f:
    json.dump({"eos_token_id":special[eos],"pad_token_id":special[eos]},f,indent=2)
print(f"exported tokenizer ({len(vocab)} bpe + {len(special)} special tokens, {len(merges)} merges) -> {out_dir}")
