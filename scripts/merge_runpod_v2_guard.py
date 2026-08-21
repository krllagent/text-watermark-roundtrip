#!/usr/bin/env python3
"""Add the narrow RunPod v2 Pod-only service without replacing policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def merge(config_path: Path, profile_path: Path) -> bool:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Guard config root must be a mapping")
    services = config.setdefault("services", {})
    if not isinstance(services, dict):
        raise ValueError("Guard services must be a mapping")
    wanted = (profile.get("services") or {}).get("runpod_v2")
    if not isinstance(wanted, dict):
        raise ValueError("profile must contain services.runpod_v2")
    existing = services.get("runpod_v2")
    if existing is not None and existing != wanted:
        raise ValueError("refusing to replace a different services.runpod_v2 block")
    if existing == wanted:
        return False
    services["runpod_v2"] = wanted
    temporary = config_path.with_name("." + config_path.name + ".runpod-v2.tmp")
    try:
        temporary.write_text(
            yaml.dump(
                config,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o640)
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args()
    changed = merge(args.config, args.profile)
    print("added runpod_v2" if changed else "runpod_v2 already matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
