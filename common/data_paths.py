"""Where the project keeps its data; command line arguments can still override these defaults."""

from pathlib import Path

DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"

RAW_CUSTOM = RAW_ROOT / "custom"
PRETRAIN_T2T = PROCESSED_ROOT / "pretrain" / "t2t"
# nanochat's SFT data, downloaded by script/download_sft.py, laid out like the Hugging Face repositories
RAW_SFT = RAW_ROOT / "sft"
SFT_SMOLTALK = RAW_SFT / "smol-smoltalk" / "data"
SFT_MMLU = RAW_SFT / "mmlu" / "all"
SFT_GSM8K = RAW_SFT / "gsm8k" / "main"
SFT_TOOLCALL = RAW_SFT / "hermes-function-calling-v1"
# evaluation data, downloaded by script/download_eval.py
# RL questions beyond GSM8K train (script/prepare_rl_data.py)
RAW_RL = RAW_ROOT / "rl"

# image understanding (script/download_vlm.py)
RAW_VLM = RAW_ROOT / "vlm"
VLM_MINIMIND_V = RAW_VLM / "minimind-v"          # ALLaVA-4V captions / instructions, 256x256 JPEG in parquet
VLM_DATA = PROCESSED_ROOT / "vlm"                # English subset, shuffled shards (script/prepare_vlm_data.py)
VLM_ENCODER = RAW_VLM / "siglip2-base-patch16-256"
VLM_EVAL = RAW_VLM / "eval"                      # MMStar, AI2D, ScienceQA-IMG, MMBench

# speech in / out of the chat page (script/download_audio.py)
RAW_AUDIO = RAW_ROOT / "audio"
AUDIO_SENSEVOICE = RAW_AUDIO / "SenseVoiceSmall" # writes down what the user says
AUDIO_KOKORO = RAW_AUDIO / "kokoro"              # reads the replies aloud

# image generation: the model that draws; the caption paths are only used to make the published data
RAW_T2I = RAW_ROOT / "t2i"
T2I_RAW_VQ = RAW_T2I / "vq256"                    # captions of 256x256 images (text-to-image-2M)
T2I_DATA = PROCESSED_ROOT / "t2i_en256"           # English captions only, shuffled shards
IMAGE_BACKEND = RAW_T2I / "z-image-turbo"         # the open-source model that draws in the conversation

RAW_EVAL = RAW_ROOT / "eval"
EVAL_BUNDLE = RAW_EVAL / "eval_bundle"  # CORE (DCLM)
EVAL_ARC = RAW_EVAL / "ai2_arc"
EVAL_HUMANEVAL = RAW_EVAL / "openai_humaneval"
SFT_IDENTITY = RAW_CUSTOM / "identity_conversations_en_d20.jsonl"
# data synthesized with DeepSeek, published on Hugging Face (script/download_release.py)
SFT_IDENTITY_QA = RAW_CUSTOM / "identity_qa_d20.jsonl"  # short identity questions (who are you / are you ChatGPT)
SFT_TOOLCALL_SYNTH = RAW_CUSTOM / "toolcall_synth.jsonl"  # retry after an argument error / ask for a missing argument / chained calls / no call needed
SFT_T2I_CHAT = RAW_CUSTOM / "t2i_chat.jsonl"  # conversational drawing: when to draw, writing the description, successive edits, when not to draw
