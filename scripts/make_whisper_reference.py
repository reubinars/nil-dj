"""Write tests/fixtures/whisper-reference.npz from the original Whisper feature extractor.

Run with the dev-only parity group: uv run --group parity python scripts/make_whisper_reference.py
"""

import sys
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from reference_signals import noise_10s, tone_4s, voice_8s

from nil_dj.whisper import to_16k


def main() -> None:
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    signals = {"noise": to_16k(noise_10s()), "tone": to_16k(tone_4s()), "voice": voice_8s()}
    out = {}
    for name, x in signals.items():
        x = x.astype(np.float32)
        out[f"in_{name}"] = x
        out[f"feat_{name}"] = fe(x, sampling_rate=16000, return_tensors="np")["input_features"][
            0
        ].astype(np.float32)
    dest = ROOT / "tests" / "fixtures" / "whisper-reference.npz"
    np.savez_compressed(dest, **out)
    print(dest, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
