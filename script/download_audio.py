"""Download the speech models of the chat page into data/raw/audio: SenseVoice writes down what the user says, Kokoro reads
the replies aloud (inference/speech_io.py).

usage:
python -m script.download_audio
"""
import argparse

from common.data_paths import AUDIO_KOKORO, AUDIO_SENSEVOICE
from common.download import large_file, snapshot

parser = argparse.ArgumentParser(description="download the speech models")
parser.add_argument("--connections", type=int, default=16)
args = parser.parse_args()

snapshot("FunAudioLLM/SenseVoiceSmall", "model", AUDIO_SENSEVOICE, ["*.json", "*.yaml", "*.mvn", "*.model", "*.md"])
large_file("FunAudioLLM/SenseVoiceSmall", "model", "model.pt", AUDIO_SENSEVOICE / "model.pt", args.connections)
snapshot("hexgrad/Kokoro-82M", "model", AUDIO_KOKORO, ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"])
