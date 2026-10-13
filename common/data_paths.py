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
SFT_T2T = PROCESSED_ROOT / "sft" / "t2t"
SFT_T2A = PROCESSED_ROOT / "sft" / "t2a"
SFT_A2A = PROCESSED_ROOT / "sft" / "a2a"
SFT_I2T = PROCESSED_ROOT / "sft" / "i2t"
SFT_T2I = PROCESSED_ROOT / "sft" / "t2i"
ALIGNMENT = PROCESSED_ROOT / "alignment"
TTS_CACHE = INTERIM_ROOT / "audio" / "tts_cache"
IDENTITY_PREVIEWS = PREVIEW_ROOT / "identity_audio"
