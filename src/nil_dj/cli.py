"""Command-line entry point: `nil-dj <command>`."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from filelock import FileLock, Timeout

from nil_dj import __version__
from nil_dj.config import Config, legacy_index_path, load_config
from nil_dj.decode import FfmpegMissing, require_ffmpeg
from nil_dj.dj import default_embedder
from nil_dj.embedder import PRECISIONS, Embedder
from nil_dj.index import Index, IndexFormatError
from nil_dj.indexer import (
    NO_TRACKS_FOUND,
    AlreadyRunning,
    EmptyLibrary,
    Progress,
    default_workers,
    run_index,
    walk,
)
from nil_dj.languages import CONF_MIN, resolve_index
from nil_dj.reindex import overrides_path
from nil_dj.search import CODECS, Filters, Hit, search_text, similar_to_paths
from nil_dj.whisper import default_detector

NO_INDEX = "No index yet. Run: nil-dj index"
PROGRESS_EVERY = 100  # lines, when stdout is not a terminal


def make_embedder(cfg: Config, args: argparse.Namespace) -> Embedder:
    """`dj.default_embedder` with the command line's --precision, --device-id and --cpu."""
    device_id = args.device_id if args.device_id is not None else cfg.gpu_device_id
    cfg = replace(cfg, precision=args.precision or cfg.precision, gpu_device_id=device_id)
    return default_embedder(cfg, use_gpu=not args.cpu)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _clock(seconds: float) -> str:
    s = max(0, round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _lock_path(index_path: Path) -> Path:
    return Path(f"{index_path}.lock")


# --- index ---------------------------------------------------------------------------------------


class _ProgressLine:
    def __init__(self) -> None:
        self.start = time.monotonic()
        self.tty = sys.stdout.isatty()

    def __call__(self, p: Progress) -> None:
        if p.error:
            self._clear()
            print(f"failed: {p.path}: {p.error}")
        if not self.tty and p.done % PROGRESS_EVERY and p.done != p.total:
            return
        elapsed = time.monotonic() - self.start
        rate = p.done / elapsed * 60 if elapsed > 0 else 0.0
        eta = (p.total - p.done) / rate * 60 if rate > 0 else 0.0
        line = f"{p.done}/{p.total}, {rate:.0f} tracks/min, ETA {_clock(eta)}"
        if self.tty:
            print(f"\r{line}\033[K", end="", flush=True)
        else:
            print(line, flush=True)

    def _clear(self) -> None:
        if self.tty:
            print("\r\033[K", end="")

    def finish(self) -> None:
        if self.tty:
            print()


def cmd_index(cfg: Config, args: argparse.Namespace) -> int:
    folders = [Path(f) for f in args.folder] if args.folder else cfg.folders
    if not folders:
        _err("No music folders: add one in nil, or pass --folder")
        return 2
    lock_path = _lock_path(cfg.index_path)
    # Before the old index goes and before the model loads: an unplugged drive must neither
    # erase the index nor start a 620 MB download only to be refused.
    if not walk(folders):
        _err(f"Nothing was changed: {NO_TRACKS_FOUND}")
        return 5
    if args.rebuild:
        try:
            with FileLock(lock_path, timeout=0):
                Index.remove(cfg.index_path)
        except Timeout:
            _err("Another nil-dj index is running")
            return 3

    # Before the model loads: a first run would otherwise download it and only then stop.
    try:
        require_ffmpeg()
    except FfmpegMissing as e:
        _err(str(e))
        return 4

    embedder = make_embedder(cfg, args)
    precision = args.precision or cfg.precision
    adapter = f" ({embedder.adapter_name})" if embedder.adapter_name else ""
    jobs = args.jobs or default_workers()
    print(
        f"Embedding on {embedder.provider}{adapter}, precision {precision}; "
        f"analysing with {jobs} workers",
        flush=True,
    )

    try:
        index = Index.open(cfg.index_path, embedder.model_id)
    except IndexFormatError as e:
        _err(str(e))
        return 1
    progress = _ProgressLine()
    detector = None if args.no_languages else default_detector(cfg, use_gpu=not args.cpu)
    try:
        index.set_meta("precision", precision)
        r = run_index(
            folders, index, embedder, lock_path, progress, workers=args.jobs, detector=detector
        )
        problems = _resolve_problems(index, cfg)
    except AlreadyRunning:
        _err("Another nil-dj index is running")
        return 3
    except EmptyLibrary:
        _err(f"Nothing was changed: {NO_TRACKS_FOUND}")
        return 5
    except FfmpegMissing as e:
        _err(str(e))
        return 4
    except KeyboardInterrupt:
        progress.finish()
        _err("Stopped; progress is saved. Run nil-dj index again to continue.")
        return 130
    finally:
        index.close()
    progress.finish()
    took = _clock(time.monotonic() - progress.start)
    print(
        f"added {r.added}, updated {r.updated}, removed {r.removed}, failed {r.failed}, "
        f"unchanged {r.unchanged} in {took}"
    )
    if r.detected + r.language_failed > 0 or r.language_problem:
        print(f"Languages: {r.detected} detected, {r.language_failed} failed")
    for problem in ([r.language_problem] if r.language_problem else []) + problems:
        _err(problem)
    return 0


def _resolve_problems(index: Index, cfg: Config) -> list[str]:
    """Resolve the final languages; problems come back as text, never raised."""
    try:
        return resolve_index(index, overrides_path(cfg))
    except Exception as e:  # noqa: BLE001 - must not turn a finished run into a crash
        return [f"Resolving languages failed: {str(e) or type(e).__name__}"]


# --- languages -----------------------------------------------------------------------------------


def _name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def cmd_languages(cfg: Config, args: argparse.Namespace) -> int:
    index = _open_existing(cfg, None)
    if index is None:
        return 1
    try:
        if args.resolve:
            for problem in _resolve_problems(index, cfg):
                _err(problem)
        for code, n in index.language_counts():
            print(f"{code}  {n:,}")
        unsure = index.unsure_tracks(CONF_MIN)
        if unsure:
            print(f"\nUnsure (confidence below {CONF_MIN}):")
            for path, artist, title, language, conf, source in unsure:
                label = f"{artist} — {title}" if artist and title else _name(path)
                print(f"{label}  {language} {conf:.2f} {source}")
        failed = index.failed_languages()
        if failed:
            print("\nDetection failed:")
            for path, reason in failed:
                print(f"{_name(path)} — {reason}")
    finally:
        index.close()
    return 0


# --- search / similar ----------------------------------------------------------------------------


def _filters(args: argparse.Namespace) -> Filters:
    return Filters(
        min_minutes=args.min_minutes,
        max_minutes=args.max_minutes,
        year_from=args.year_from,
        year_to=args.year_to,
        artists=args.artist,
        exclude_artists=args.exclude_artist,
        formats=args.format,
        hires_only=args.hires_only,
    )


def _print_hits(hits: list[Hit]) -> None:
    if not hits:
        print("No matching tracks.")
        return
    for h in hits:
        i = h.row.info
        artist = i.artist or i.album_artist or "?"
        title = i.title or Path(h.row.path).stem
        album = f"  [{i.album}]" if i.album else ""
        print(f"{h.score:.3f}  {artist} — {title}{album}  {_clock(i.duration_s)}  {i.codec}")


def _open_existing(cfg: Config, model_id: str | None) -> Index | None:
    if not cfg.index_path.exists():
        _err(NO_INDEX)
        return None
    try:
        return Index.open(cfg.index_path, model_id)
    except IndexFormatError as e:
        _err(str(e))
        return None


def cmd_search(cfg: Config, args: argparse.Namespace) -> int:
    if not cfg.index_path.exists():
        _err(NO_INDEX)
        return 1
    embedder = make_embedder(cfg, args)
    index = _open_existing(cfg, embedder.model_id)
    if index is None:
        return 1
    try:
        template = args.template if args.template is not None else cfg.prompt_template
        hits = search_text(index, embedder, args.text, args.limit, _filters(args), template)
    finally:
        index.close()
    _print_hits(hits)
    return 0


def cmd_similar(cfg: Config, args: argparse.Namespace) -> int:
    index = _open_existing(cfg, None)
    if index is None:
        return 1
    try:
        hits = similar_to_paths(index, [args.path], args.limit, _filters(args))
    except KeyError:
        _err(f"Not in the index: {args.path}")
        return 1
    finally:
        index.close()
    _print_hits(hits)
    return 0


# --- status --------------------------------------------------------------------------------------


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    index = _open_existing(cfg, None)
    if index is None:
        return 1
    try:
        errors = index.errors()
        print(f"index: {cfg.index_path}")
        print(f"tracks: {index.count()}")
        print(f"model {index.get_meta('model_id')}, precision {index.get_meta('precision') or '?'}")
        print(f"errors: {len(errors)}")
        for path, error in errors[:20]:
            print(f"  {path}: {error}")
        if len(errors) > 20:
            print(f"  … and {len(errors) - 20} more")
    finally:
        index.close()
    return 0


# --- parser --------------------------------------------------------------------------------------


def _positive_int(s: str) -> int:
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--min-minutes", type=float)
    p.add_argument("--max-minutes", type=float)
    p.add_argument("--year-from", type=int)
    p.add_argument("--year-to", type=int)
    p.add_argument("--artist", action="append", metavar="A")
    p.add_argument("--exclude-artist", action="append", metavar="A")
    p.add_argument("--format", action="append", choices=CODECS, metavar="F")
    p.add_argument("--hires-only", action="store_true")


def cmd_plugin(cfg: Config, args: argparse.Namespace) -> int:
    from nil_dj import plugin

    return plugin.main()


def cmd_serve(cfg: Config, args: argparse.Namespace) -> int:
    from nil_dj import server

    return server.serve_stdio(cfg) if args.stdio else server.serve_http(cfg)


def cmd_install(cfg: Config, args: argparse.Namespace) -> int:
    from nil_dj import install as installer

    try:
        installer.install(
            Path(args.nil),
            installer.repo_root(),
            run=installer.run_checked,
            legacy_index=legacy_index_path(),
        )
    except installer.InstallError as e:
        _err(str(e))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nil-dj", description="An AI DJ for the nil player.")
    parser.add_argument("--version", action="version", version=f"nil-dj {__version__}")
    parser.set_defaults(precision=None, device_id=None, cpu=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("index", help="build or refresh the index from your music folders")
    p.add_argument("--rebuild", action="store_true", help="start from an empty index")
    p.add_argument("--precision", choices=list(PRECISIONS))
    p.add_argument("--device-id", type=int, help="DirectML adapter index")
    p.add_argument("--cpu", action="store_true", help="don't use the GPU")
    p.add_argument("--folder", action="append", metavar="PATH", help="instead of nil's folders")
    p.add_argument("--no-languages", action="store_true", help="skip language detection")
    p.add_argument(
        "--jobs", type=_positive_int, metavar="N",
        help="tracks analysed in parallel (default: half the logical cores)",
    )  # fmt: skip
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="find tracks by how they sound")
    p.add_argument("text")
    _add_filter_args(p)
    p.add_argument("--template", help='prompt template, e.g. "This is a sound of {}"')
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("similar", help="tracks that sound like a track")
    p.add_argument("path")
    _add_filter_args(p)
    p.set_defaults(func=cmd_similar)

    p = sub.add_parser("languages", help="tracks per language, and the ones to check")
    p.add_argument(
        "--resolve", action="store_true", help="re-resolve first (after editing overrides)"
    )
    p.set_defaults(func=cmd_languages)

    p = sub.add_parser("status", help="what's in the index")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("plugin", help="run as a nil plugin (started by nil, not by hand)")
    p.set_defaults(func=cmd_plugin)

    p = sub.add_parser("serve", help="the MCP server, without nil (playback tools need nil)")
    p.add_argument(
        "--stdio", action="store_true",
        help="MCP over stdio; relays to nil's nil-dj plugin when it is running",
    )  # fmt: skip
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("install", help="install the plugin into a nil folder")
    p.add_argument("--nil", required=True, metavar="PATH", help="nil's folder (has nil.exe)")
    p.set_defaults(func=cmd_install)
    return parser


def _never_fail_on_encoding() -> None:
    # Redirected output on Windows uses the locale code page, which can't encode Japanese titles;
    # replace what it can't show rather than crash mid-run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    _never_fail_on_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(load_config(), args)
