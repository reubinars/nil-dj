"""Command-line entry point: `nil-dj <command>`."""

from __future__ import annotations

import argparse

from nil_dj import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nil-dj", description="An AI DJ for the nil player.")
    parser.add_argument("--version", action="version", version=f"nil-dj {__version__}")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0
