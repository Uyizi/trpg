#!/usr/bin/env python3
"""Validate adventure references and print a spoiler-free identity summary."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_trpg.modules import ModuleValidationError, load_module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="module directory (default: reserve_cooling)")
    args = parser.parse_args()
    try:
        module = load_module(args.path)
    except ModuleValidationError as exc:
        print(f"Module validation failed: {exc}", file=sys.stderr)
        return 1
    world = module["world"]
    summary = {key: module[key] for key in ("module_id", "module_version", "module_hash")}
    summary.update(locations=len(world["locations"]), npcs=len(world["actors"]),
                   atomic_facts=len(world["facts"]), evidence=len(world["evidence"]),
                   actions=len(world["actions"]), triggers=len(world["triggers"]), routes=len(world["routes"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
