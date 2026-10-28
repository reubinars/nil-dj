"""Check the language column against the owner's hand labels. Prints counts only.

    uv run python eval/language.py --labels PATH\\language-labels.csv [--index PATH]

The CSV (UTF-8, maybe with a BOM) has the columns artist, title, whisper_guess, confidence,
no_speech, your_label. The expected label is `your_label`, else `whisper_guess`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

from nil_dj.config import load_config
from nil_dj.index import Index
from nil_dj.languages import INSTRUMENTAL, normalise_code

MIXED = "mixed"
NOT_DETECTED = "not detected"

# (codes that count as right, expects a second language): what a label asks of a row.
Label = tuple[frozenset[str], bool]


def parse_label(text: str | None) -> Label | None:
    """The rule for one label, or None to skip it (empty, `?`, or a code nil-dj doesn't know)."""
    t = (text or "").strip().casefold()
    if not t or t == "?":
        return None
    if t.startswith("instrumental"):  # "instrumental? (la)": a guess made on a sung language
        return frozenset({INSTRUMENTAL}), False
    if t == MIXED:
        return frozenset(), True
    if t.startswith(MIXED + " "):
        parts = t.split()[1:]
    else:
        parts = [t]
    try:
        codes = frozenset(normalise_code(p) for p in parts)
    except ValueError:
        return None
    return codes, False


def expected_label(whisper_guess: str | None, your_label: str | None) -> str:
    return (your_label or "").strip() or (whisper_guess or "")


def is_correct(label: Label, language: str | None, language_alt: str | None) -> bool:
    codes, needs_alt = label
    if needs_alt:
        return language_alt is not None
    return language in codes


def label_name(label: Label) -> str:
    codes, needs_alt = label
    return MIXED if needs_alt or len(codes) > 1 else next(iter(codes))


def read_labels(path: Path) -> list[tuple[str, str, str]]:
    """(artist, title, expected label) per row."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            (r.get("artist") or "", r.get("title") or "",
             expected_label(r.get("whisper_guess"), r.get("your_label")))
            for r in csv.DictReader(f)
        ]  # fmt: skip


def _key(artist: str, title: str) -> tuple[str, str]:
    return artist.strip().casefold(), title.strip().casefold()


def evaluate(index: Index, labels: Iterable[tuple[str, str, str]]) -> dict:
    rows, _ = index.load()
    by_key: dict[tuple[str, str], object] = {}
    for r in rows:
        by_key.setdefault(_key(r.info.artist or "", r.info.title or ""), r)
    confusion: Counter[tuple[str, str]] = Counter()
    correct = total = skipped = missing = 0
    for artist, title, text in labels:
        label = parse_label(text)
        if label is None:
            skipped += 1
            continue
        row = by_key.get(_key(artist, title))
        if row is None:
            missing += 1
            continue
        total += 1
        ok = is_correct(label, row.language, row.language_alt)
        correct += ok
        confusion[(label_name(label), row.language or NOT_DETECTED)] += 1
    return {"correct": correct, "total": total, "skipped": skipped, "missing": missing,
            "confusion": confusion}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--index", type=Path, default=None)
    args = p.parse_args(argv)
    path = args.index or load_config().index_path
    with closing(Index.open(path, None)) as index:
        r = evaluate(index, read_labels(args.labels))
    print(f"correct {r['correct']}/{r['total']}")
    print(f"skipped (empty, ?, unknown code): {r['skipped']}")
    print(f"labels with no row in the index: {r['missing']}")
    print("confusion (expected -> detected: count)")
    for (want, got), n in sorted(r["confusion"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {want} -> {got}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
