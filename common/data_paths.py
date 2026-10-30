"""项目数据目录的统一约定；训练参数仍可覆盖这些默认路径。"""

from pathlib import Path

DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"
PREVIEW_ROOT = DATA_ROOT / "previews"
MANIFEST_ROOT = DATA_ROOT / "manifests"

RAW_TEXT_MINIMIND = RAW_ROOT / "text" / "minimind"
RAW_CUSTOM = RAW_ROOT / "custom"
RAW_SYNTHETIC = RAW_ROOT / "synthetic"
PRETRAIN_T2T = PROCESSED_ROOT / "pretrain" / "t2t"
# nanochat 的 SFT 数据，由 script/download_sft.py 下载，目录结构与 HF 仓库一致
RAW_SFT = RAW_ROOT / "sft"
SFT_SMOLTALK = RAW_SFT / "smol-smoltalk" / "data"
SFT_MMLU = RAW_SFT / "mmlu" / "all"
SFT_GSM8K = RAW_SFT / "gsm8k" / "main"
SFT_TOOLCALL = RAW_SFT / "hermes-function-calling-v1"
# 评测数据，由 script/download_eval.py 下载
# RL questions beyond GSM8K train (script/prepare_rl_data.py)
RAW_RL = RAW_ROOT / "rl"

# image understanding (script/download_vlm.py)
RAW_VLM = RAW_ROOT / "vlm"
VLM_MINIMIND_V = RAW_VLM / "minimind-v"          # ALLaVA-4V captions / instructions, 256x256 JPEG in parquet
VLM_DATA = PROCESSED_ROOT / "vlm"                # English subset, shuffled shards (script/prepare_vlm_data.py)
VLM_ENCODER = RAW_VLM / "siglip2-base-patch16-256"
VLM_EVAL = RAW_VLM / "eval"                      # MMStar, AI2D, ScienceQA-IMG, MMBench

RAW_EVAL = RAW_ROOT / "eval"
EVAL_BUNDLE = RAW_EVAL / "eval_bundle"  # CORE (DCLM)
EVAL_ARC = RAW_EVAL / "ai2_arc"
EVAL_HUMANEVAL = RAW_EVAL / "openai_humaneval"
SFT_IDENTITY = RAW_CUSTOM / "identity_conversations_en_d20.jsonl"
# datagen/ 用 DeepSeek 合成的数据
SFT_IDENTITY_QA = RAW_CUSTOM / "identity_qa_d20.jsonl"  # 短的身份问答（你是谁 / 你是 ChatGPT 吗）
SFT_TOOLCALL_SYNTH = RAW_CUSTOM / "toolcall_synth.jsonl"  # 参数纠错重试 / 缺参数先追问 / 连续调用 / 不该调用
SFT_T2A = PROCESSED_ROOT / "sft" / "t2a"
SFT_A2A = PROCESSED_ROOT / "sft" / "a2a"
SFT_I2T = PROCESSED_ROOT / "sft" / "i2t"
SFT_T2I = PROCESSED_ROOT / "sft" / "t2i"
ALIGNMENT = PROCESSED_ROOT / "alignment"
TTS_CACHE = INTERIM_ROOT / "audio" / "tts_cache"
IDENTITY_PREVIEWS = PREVIEW_ROOT / "identity_audio"
