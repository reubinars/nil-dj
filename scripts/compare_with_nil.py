"""Compare the paths nil knows (library-cache.json) with the paths in nil-dj's index.

Expected: everything nil has is either indexed or listed in index_errors, and the index holds
nothing nil doesn't. Any other difference means `walk` disagrees with nil's scanner.

Usage: uv run python scripts/compare_with_nil.py
"""

import json
import sqlite3

from nil_dj.config import default_index_path, nil_config_dir

SHOW = 20


def main() -> int:
    with (nil_config_dir() / "library-cache.json").open(encoding="utf-8") as f:
        nil = {entry["path"] for entry in json.load(f)["files"]}
    db = sqlite3.connect(f"file:{default_index_path()}?mode=ro", uri=True)
    indexed = {p for (p,) in db.execute("SELECT path FROM tracks")}
    errors = {p for (p,) in db.execute("SELECT path FROM index_errors")}

    print(f"nil {len(nil)}, indexed {len(indexed)}, index_errors {len(errors)}")
    diffs = {
        "in nil not in index": nil - indexed,
        "  ...of which in index_errors": (nil - indexed) & errors,
        "  ...of which unexplained": nil - indexed - errors,
        "in index not in nil": indexed - nil,
        "in index_errors not in nil": errors - nil,
    }
    for label, paths in diffs.items():
        print(f"{label}: {len(paths)}")
        if "errors" not in label or "unexplained" in label:
            for p in sorted(paths)[:SHOW]:
                print(f"    {p}")
    bad = diffs["  ...of which unexplained"] or diffs["in index not in nil"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
