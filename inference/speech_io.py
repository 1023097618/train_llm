"""Voice for the chat page: SenseVoice writes down what the user says, Kokoro (Apache 2.0) reads the reply aloud. Both on the CPU,
the GPU is left to the models that chat and draw. Kokoro lives outside the training environment:
python -m venv --system-site-packages ~/train_llm_pylibs/voice && ~/train_llm_pylibs/voice/bin/pip install kokoro "misaki[en]"
"""
import io
import re

import numpy as np
import soundfile as sf

from common.data_paths import AUDIO_SENSEVOICE, RAW_ROOT

KOKORO = RAW_ROOT / "audio" / "kokoro"


class SpeechIn:
    def __init__(self, model_dir=str(AUDIO_SENSEVOICE), device="cpu"):
        import funasr
        self.model = funasr.AutoModel(model=model_dir, disable_update=True, device=device)

    def transcribe(self, wav_bytes):
        """a WAV file -> text; the page sends 16 kHz mono"""
        audio, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        assert rate == 16000, f"expected 16 kHz, got {rate}"
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        text = self.model.generate(input=audio, language="en", use_itn=True, disable_pbar=True)[0]["text"]
        return re.sub(r"<\|[^|]*\|>", "", text).strip()  # SenseVoice's language, emotion and event tags


class SpeechOut:
    def __init__(self, voice="af_heart", device="cpu"):
        from kokoro import KModel, KPipeline
        model = KModel(config=str(KOKORO / "config.json"), model=str(KOKORO / "kokoro-v1_0.pth")).to(device).eval()
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
        self.voice = str(KOKORO / "voices" / f"{voice}.pt")

    def speak(self, text):
        """text -> a WAV file at 24 kHz"""
        parts = [r.audio.numpy() for r in self.pipeline(text, voice=self.voice) if r.audio is not None]
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(parts) if parts else np.zeros(2400, dtype=np.float32), 24000, format="WAV")
        return buffer.getvalue()
