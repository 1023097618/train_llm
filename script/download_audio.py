"""Download the speech data and models into data/raw/audio.

- jingyaogong/minimind-o_dataset: English sft_t2a_mini (text -> speech, ~470 h) and sft_a2a_mini
  (speech -> speech, ~75 h in / ~57 h out); speech is stored as Mimi codes (8 codebooks, 12.5 Hz)
- kyutai/mimi: codec, decodes the generated codes to 24 kHz audio
- FunAudioLLM/SenseVoiceSmall: frozen speech encoder for spoken input

usage:
HF_TOKEN=hf_... python -m script.download_audio
"""
import argparse

from common.data_paths import AUDIO_MIMI, AUDIO_MINIMIND_O, AUDIO_SENSEVOICE
from common.download import large_file, snapshot

parser = argparse.ArgumentParser(description="download speech data and models")
parser.add_argument("--connections", type=int, default=16)
args = parser.parse_args()

snapshot("kyutai/mimi", "model", AUDIO_MIMI, ["*.json", "model.safetensors", "*.md"])
snapshot("FunAudioLLM/SenseVoiceSmall", "model", AUDIO_SENSEVOICE, ["*.json", "*.yaml", "*.mvn", "*.model", "*.md"])
large_file("FunAudioLLM/SenseVoiceSmall", "model", "model.pt", AUDIO_SENSEVOICE / "model.pt", args.connections)
for name in ("sft_t2a_mini.parquet", "sft_a2a_mini.parquet"):
    large_file("jingyaogong/minimind-o_dataset", "dataset", name, AUDIO_MINIMIND_O / name, args.connections)
