"""Write tests/fixtures/clap-reference.npz from the original PyTorch CLAP model.

Run with the dev-only parity group: uv run --group parity python scripts/make_reference.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from reference_signals import PROMPTS, SR, noise_10s, tone_4s

MODEL = "laion/clap-htsat-unfused"


def main() -> None:
    model = ClapModel.from_pretrained(MODEL).eval()
    fe = ClapFeatureExtractor.from_pretrained(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)

    out = {}
    with torch.no_grad():
        for name, x in (("noise", noise_10s()), ("tone", tone_4s())):
            feats = fe(x, sampling_rate=SR, return_tensors="pt")
            out[f"feat_{name}"] = feats["input_features"][0, 0].numpy().astype(np.float32)
            emb = model.get_audio_features(input_features=feats["input_features"])
            out[f"audio_{name}"] = emb[0].numpy().astype(np.float32)
        text = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt")
            text.append(model.get_text_features(**ids)[0].numpy().astype(np.float32))
        out["text"] = np.stack(text)

    dest = ROOT / "tests" / "fixtures" / "clap-reference.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(dest, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
