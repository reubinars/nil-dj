"""Language codes, the window -> verdict rule and prompt words.

Pure module (stdlib + numpy). Thresholds are applied at resolve time, so they can be re-tuned
without running Whisper again.
"""

from __future__ import annotations

import json
import re
import tomllib
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nil_dj.index import Final, Index, LangRow, path_key

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


def _fold(name: str) -> str:
    """Case- and accent-insensitive form of a name, for matching overrides."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).casefold()


def decode_raw(text: str) -> list[Window] | None:
    data = json.loads(text)
    if "failed" in data:
        return None
    return [Window(dict(w["p"]), w["ns"]) for w in data["windows"]]


PROTECTED_GENRES = (
    "J-pop", "J-rock", "K-pop", "C-pop", "Mandopop", "Cantopop", "French house", "Latin",
    "Spanish guitar", "Italo disco",
)
INSTRUMENTAL_WORDS = (
    "instrumental", "instrumentals", "no vocals", "without vocals", "no singing",
)
_ALIASES: dict[str, str] = {
    "bahasa indonesia": "id", "bahasa": "id", "indonesia": "id", "malay": "id",
    "javanese": "id", "sundanese": "id", "mandarin": "zh", "cantonese": "zh",
    "español": "es", "français": "fr", "deutsch": "de", "русский": "ru",
    "português": "pt", "italiano": "it",
}
# Endonyms in scripts without word spacing match anywhere.
_CJK_ALIASES: dict[str, str] = {
    "日本語": "ja", "한국어": "ko", "中文": "zh", "汉语": "zh", "華語": "zh",
}
_GAP = re.compile(r"\s*(?:,\s*(?:and|or)|,|and|or|&)?\s*", re.IGNORECASE)


def _build_prompt_regex() -> tuple[re.Pattern[str], dict[str, str]]:
    kinds: dict[str, str] = {}  # casefolded phrase -> code ("" for a protected genre)
    for g in PROTECTED_GENRES:
        kinds[g.casefold()] = ""
    for word in INSTRUMENTAL_WORDS:
        kinds[word] = INSTRUMENTAL
    for code, name in NAMES.items():
        kinds.setdefault(name.casefold(), code)
    kinds.update(_ALIASES)
    pieces = []
    for phrase in sorted(kinds, key=lambda x: (-len(x), kinds[x] != "")):
        body = r"\s+".join(re.escape(part) for part in phrase.split())
        pieces.append(rf"(?<![\w-]){body}(?![\w-])")
    for phrase in _CJK_ALIASES:
        pieces.append(re.escape(phrase))
    kinds.update(_CJK_ALIASES)
    return re.compile("|".join(pieces), re.IGNORECASE), kinds


_PROMPT_RE, _PROMPT_KINDS = _build_prompt_regex()


def parse_prompt(text: str) -> tuple[list[str], str]:
    """Split language requests off a prompt: (codes in order, remaining text)."""
    langs: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _PROMPT_RE.finditer(text):
        code = _PROMPT_KINDS[re.sub(r"\s+", " ", m.group(0)).casefold()]
        if not code:
            continue  # protected genre: stays with CLAP
        if code not in langs:
            langs.append(code)
        spans.append((m.start(), m.end()))
    if not spans:
        return [], text
    # A connector is dropped only when it sits between two removed matches.
    merged = [spans[0]]
    for start, end in spans[1:]:
        if _GAP.fullmatch(text[merged[-1][1] : start]):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    out, pos = [], 0
    for start, end in merged:
        out.append(text[pos:start])
        out.append(" ")
        pos = end
    out.append(text[pos:])
    rest = re.sub(r"\s+", " ", "".join(out)).strip(" ,")
    return langs, rest or "music"


# --- overrides and resolve (spec section 6) ---

OVERRIDES_FILE = "languages.toml"
SOURCE_OVERRIDE = "override"
SOURCE_ARTIST = "artist"
SOURCE_WHISPER = "whisper"
_Value = tuple[str, str | None]


@dataclass(frozen=True)
class Overrides:
    tracks: dict[str, _Value]  # path_key -> (language, alt)
    albums: dict[tuple[str, str], _Value]  # (artist, album), folded
    artists: dict[str, _Value]  # folded


def _value(raw: object) -> _Value:
    if not isinstance(raw, str):
        raise TypeError("must be a language code")
    code, _, alt = raw.partition("+")
    return normalise_code(code), (normalise_code(alt) if alt else None)


def load_overrides(path: Path) -> tuple[Overrides, list[str]]:
    empty = Overrides({}, {}, {})
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return empty, []
    except OSError as e:
        return empty, [f"{OVERRIDES_FILE}: {e}"]
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError) as e:
        return empty, [f"{OVERRIDES_FILE}: {e}"]
    problems: list[str] = []
    tracks: dict[str, _Value] = {}
    albums: dict[tuple[str, str], _Value] = {}
    artists: dict[str, _Value] = {}

    def put(section: str, key: str, raw: object, store: dict, store_key) -> None:
        try:
            store[store_key] = _value(raw)
        except (TypeError, ValueError) as e:
            problems.append(f'{OVERRIDES_FILE}: [{section}] "{key}": {e}')

    for section, body in data.items():
        if section == "artist" and isinstance(body, dict):
            for k, v in body.items():
                put("artist", k, v, artists, _fold(k))
        elif section == "track" and isinstance(body, dict):
            for k, v in body.items():
                put("track", k, v, tracks, path_key(k))
        elif section == "album" and isinstance(body, dict):
            for artist, table in body.items():
                if not isinstance(table, dict):
                    problems.append(
                        f'{OVERRIDES_FILE}: [album] needs an artist table (got "{artist}")'
                    )
                    continue
                for album, v in table.items():
                    put(f'album."{artist}"', album, v, albums, (_fold(artist), _fold(album)))
        elif section in ("artist", "track", "album"):
            problems.append(f"{OVERRIDES_FILE}: [{section}] must be a table")
        else:
            problems.append(f"{OVERRIDES_FILE}: unknown section [{section}]")
    if problems:
        return empty, problems
    return Overrides(tracks, albums, artists), []


def _group(row: LangRow) -> str | None:
    name = row.album_artist or row.artist
    return _fold(name) if name else None


def _override_for(row: LangRow, ov: Overrides) -> _Value | None:
    if (hit := ov.tracks.get(path_key(row.path))) is not None:
        return hit
    group = _group(row)
    if group is not None and row.album is not None:
        hit = ov.albums.get((group, _fold(row.album)))
        if hit is not None:
            return hit
    for name in (row.album_artist, row.artist):
        if name and (hit := ov.artists.get(_fold(name))) is not None:
            return hit
    return None


def resolve(rows: list[LangRow], overrides: Overrides) -> dict[str, Final]:
    """Final language columns for every row that has raw. First match wins (spec section 6)."""
    finals: dict[str, Final] = {}
    verdicts: dict[str, Verdict | None] = {}
    votes: dict[str, Counter[str]] = {}
    for r in rows:
        if r.raw is None:
            continue
        try:
            windows = decode_raw(r.raw)
        except (ValueError, KeyError, TypeError, AttributeError):
            windows = None  # one malformed row is a failed row, not a failed library
        v = verdict(windows) if windows else None
        verdicts[r.path] = v
        ov = _override_for(r, overrides)
        if ov is not None:
            finals[r.path] = Final(
                ov[0], ov[1], None, v.vocal_score if v else None, SOURCE_OVERRIDE
            )
        elif v is not None and not v.unsure and v.language != INSTRUMENTAL and _group(r):
            votes.setdefault(_group(r), Counter())[v.language] += 1
    for r in rows:
        if r.raw is None or r.path in finals:
            continue
        v = verdicts[r.path]
        if v is None:
            finals[r.path] = Final(UNKNOWN, None, None, None, SOURCE_WHISPER)
            continue
        g = _group(r)
        tally = votes.get(g) if g is not None else None
        if tally and v.unsure and v.language != INSTRUMENTAL:
            lang, n = tally.most_common(1)[0]
            total = tally.total()
            if total >= MAJORITY_MIN and n / total >= MAJORITY_SHARE:
                finals[r.path] = Final(lang, None, v.conf, v.vocal_score, SOURCE_ARTIST)
                continue
        finals[r.path] = Final(v.language, v.alt, v.conf, v.vocal_score, SOURCE_WHISPER)
    return finals


def _mtime_ns(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except OSError:
        return "0"


def resolve_index(index: Index, overrides_path: Path) -> list[str]:
    mtime = _mtime_ns(overrides_path)
    overrides, problems = load_overrides(overrides_path)
    index.set_final(resolve(index.language_rows(), overrides))
    index.set_meta("overrides_mtime_ns", mtime)
    return problems


def overrides_changed(index: Index, overrides_path: Path) -> bool:
    return index.get_meta("overrides_mtime_ns") != _mtime_ns(overrides_path)
