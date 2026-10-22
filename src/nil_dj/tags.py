"""Tags and stream info with mutagen, read through each format's native keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mutagen
from mutagen.aiff import AIFF
from mutagen.dsdiff import DSDIFF
from mutagen.dsf import DSF
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.wave import WAVE


@dataclass(frozen=True)
class TrackInfo:
    title: str | None
    artist: str | None
    album_artist: str | None
    album: str | None
    genre: str | None
    year: int | None
    disc: int | None
    number: int | None
    duration_s: float
    codec: str
    sample_rate: int
    bits: int | None


class TagError(Exception):
    pass


_FIELDS = ("title", "artist", "album_artist", "album", "genre", "date", "disc", "number")
_VORBIS = dict(zip(_FIELDS, ("title", "artist", "albumartist", "album", "genre", "date",
                             "discnumber", "tracknumber"), strict=True))  # fmt: skip
_MP4 = dict(zip(_FIELDS, ("©nam", "©ART", "aART", "©alb", "©gen", "©day", "disk", "trkn"),
                strict=True))  # fmt: skip
_ID3 = dict(zip(_FIELDS, ("TIT2", "TPE1", "TPE2", "TALB", "TCON", "TDRC", "TPOS", "TRCK"),
                strict=True))  # fmt: skip


def _text(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _first(values) -> object | None:
    if not values:
        return None
    return values[0]


def _vorbis(tags) -> dict:
    return {k: _first(tags.get(key)) for k, key in _VORBIS.items()} if tags else {}


def _mp4(tags) -> dict:
    if not tags:
        return {}
    out = {k: _first(tags.get(key)) for k, key in _MP4.items()}
    for k in ("disc", "number"):  # (n, total) tuples
        if out[k] is not None:
            out[k] = out[k][0] or None
    return out


def _id3(tags) -> dict:
    if not tags:
        return {}
    out = {}
    for k, key in _ID3.items():
        frame = tags.get(key)
        out[k] = _first(frame.text) if frame is not None else None
    return out


def _leading_int(v) -> int | None:
    if v is None or isinstance(v, int):
        return v
    m = re.match(r"\s*(\d+)", str(v).split("/")[0])
    return int(m.group(1)) if m else None


def _year(v) -> int | None:
    m = re.search(r"\d{4}", str(v)) if v is not None else None
    return int(m.group(0)) if m else None


def read_info(path: Path) -> TrackInfo:
    try:
        f = mutagen.File(path)
    except Exception as e:  # mutagen raises many types for corrupt files
        raise TagError(f"can't read tags: {e}") from e
    if f is None or not getattr(f.info, "length", 0):
        raise TagError("not a readable audio file")

    info = f.info
    bits = getattr(info, "bits_per_sample", None) or None
    if isinstance(f, FLAC):
        codec, raw = "flac", _vorbis(f.tags)
    elif isinstance(f, MP4):
        codec = "alac" if getattr(info, "codec", "") == "alac" else "aac"
        raw = _mp4(f.tags)
        if codec == "aac":
            bits = None
    elif isinstance(f, WAVE):
        codec, raw = "wav", _id3(f.tags)
    elif isinstance(f, AIFF):
        codec, raw = "aiff", _id3(f.tags)
    elif isinstance(f, (DSF, DSDIFF)):
        codec, raw, bits = "dsd", _id3(f.tags), None
    else:
        raise TagError(f"unsupported format {type(f).__name__}")

    return TrackInfo(
        title=_text(raw.get("title")),
        artist=_text(raw.get("artist")),
        album_artist=_text(raw.get("album_artist")),
        album=_text(raw.get("album")),
        genre=_text(raw.get("genre")),
        year=_year(raw.get("date")),
        disc=_leading_int(raw.get("disc")),
        number=_leading_int(raw.get("number")),
        duration_s=float(info.length),
        codec=codec,
        sample_rate=int(info.sample_rate),
        bits=bits,
    )
