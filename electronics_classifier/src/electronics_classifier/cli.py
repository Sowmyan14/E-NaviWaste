"""Command line interface: ``run``, ``show-config``, ``list-components``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import ConfigError, config_hash, dump_yaml, load_config
from .registry import describe_components


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", action="append", default=[], metavar="NAME",
                        help="preset name from config/presets/ or a YAML path (repeatable)")
    parser.add_argument("--override", action="append", default=[], metavar="FILE",
                        help="override YAML file (repeatable)")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V",
                        help="dotted.key=value, value parsed as YAML (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="electronics_classifier.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show-config", help="print the resolved config without running anything")
    _add_config_args(show)

    sub.add_parser("list-components", help="list every component kind, name and its parameters")
    return parser


def _show_config(args: argparse.Namespace) -> int:
    resolved = load_config(args.preset, args.override, args.sets)
    sys.stdout.write(dump_yaml(resolved))
    sys.stdout.write(f"# config_hash: {config_hash(resolved)}\n")
    return 0


def _list_components() -> int:
    for kind, components in describe_components().items():
        print(kind)
        if not components:
            print("  (none registered)")
        for name, params in components.items():
            print(f"  {name}")
            for field_name, type_name, default, description in params:
                print(f"    {field_name}: {type_name} = {default}  {description}".rstrip())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "show-config":
            return _show_config(args)
        if args.command == "list-components":
            return _list_components()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
