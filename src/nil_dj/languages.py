"""Language codes, the window -> verdict rule and prompt words.

Pure module (stdlib + numpy). Thresholds are applied at resolve time, so they can be re-tuned
without running Whisper again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

INSTRUMENTAL = "instrumental"
UNKNOWN = "unknown"
GROUP = "id"
GROUP_ALIASES = ("jw", "ms", "su")

CONF_MIN = 0.6
INSTRUMENTAL_NS = 0.7
WINDOW_CONF = 0.6
MIN_WEIGHT = 0.001
MAJORITY_MIN = 3
MAJORITY_SHARE = 0.7

# Whisper's language tokens (tokenizer order) without jw / ms / su, which fold into "id".
NAMES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "cs": "Czech", "ro": "Romanian",
    "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian", "th": "Thai",
    "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak", "te": "Telugu",
    "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani",
    "sl": "Slovenian", "kn": "Kannada", "et": "Estonian", "mk": "Macedonian", "br": "Breton",
    "eu": "Basque", "is": "Icelandic", "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian",
    "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili", "gl": "Galician",
    "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer", "sn": "Shona",
    "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian",
    "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic",
    "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese", "ht": "Haitian Creole",
    "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", "mt": "Maltese", "sa": "Sanskrit",
    "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy",
    "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir",
}


def normalise_code(code: str) -> str:
    c = code.strip().casefold()
    if c in GROUP_ALIASES:
        return GROUP
    if c == INSTRUMENTAL or c in NAMES:
        return c
    raise ValueError(f"unknown language '{code}'")


@dataclass(frozen=True)
class Window:
    p: dict[str, float]
    ns: float


@dataclass(frozen=True)
class Verdict:
    language: str
    alt: str | None
    conf: float
    vocal_score: float
    unsure: bool


def _grouped(p: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for code, prob in p.items():
        key = GROUP if code in GROUP_ALIASES else code
        out[key] = out.get(key, 0.0) + prob
    return out


def verdict(windows: list[Window]) -> Verdict:
    probs = [_grouped(w.p) for w in windows]
    ns = np.array([w.ns for w in windows], dtype=float)
    weights = np.maximum(MIN_WEIGHT, 1.0 - ns)
    codes = sorted({c for p in probs for c in p})
    total = float(weights.sum())
    agg = {
        c: float(sum(v * p.get(c, 0.0) for v, p in zip(weights, probs, strict=True)) / total)
        for c in codes
    }
    top = max(codes, key=lambda c: agg[c])
    conf = agg[top]
    mean_ns = float(ns.mean())
    vocal_score = 1.0 - mean_ns
    if conf < CONF_MIN and mean_ns > INSTRUMENTAL_NS:
        return Verdict(INSTRUMENTAL, None, conf, vocal_score, False)
    named = {
        max(p, key=lambda c: p[c])
        for p in probs
        if p and max(p.values()) >= WINDOW_CONF
    }
    others = [c for c in named if c != top]
    alt = max(others, key=lambda c: agg[c]) if len(named) >= 2 and others else None
    return Verdict(top, alt, conf, vocal_score, conf < CONF_MIN)


def encode_raw(windows: list[Window]) -> str:
    return json.dumps(
        {
            "windows": [
                {
                    "ns": round(w.ns, 4),
                    "p": {c: round(v, 4) for c, v in w.p.items() if v >= MIN_WEIGHT},
                }
                for w in windows
            ]
        },
        separators=(",", ":"),
    )


def encode_failed(reason: str) -> str:
    return json.dumps({"failed": reason}, separators=(",", ":"))


def decode_raw(text: str) -> list[Window] | None:
    data = json.loads(text)
    if "failed" in data:
        return None
    return [Window(dict(w["p"]), w["ns"]) for w in data["windows"]]
