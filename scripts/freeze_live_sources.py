"""Write a sources overlay that replays saved live payloads instead of fetching them.

Used for method-only before/after comparisons: every HTTP source in the live config
is replaced by a ``fixture`` source pointing at a saved snapshot file, keeping the
same id, table, parser_version and parser_args, so the parser sees identical bytes.

Example::

    uv run python scripts/freeze_live_sources.py --base sources_senate_live.yaml \
        --snapshot-dir data/snapshots/2026-10-19 --out frozen_2026-10-19_senate.yaml

The overlay is written under ``configs/`` (``configs/frozen_*`` is gitignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HTTP_TYPES = {"http_json", "http_text", "http_csv"}


def build_overlay(base: str, snapshot_dir: Path, config_dir: Path) -> dict[str, object]:
    payload = yaml.safe_load((config_dir / base).read_text(encoding="utf-8")) or {}
    frozen: list[dict[str, object]] = []
    for item in payload.get("sources", []):
        if item.get("type") not in HTTP_TYPES:
            continue
        name = Path(str(item.get("path") or f"{item['id']}.dat")).name
        snapshot = snapshot_dir / name
        if not snapshot.exists():
            raise FileNotFoundError(f"missing snapshot for {item['id']}: {snapshot}")
        replaced = dict(item)
        replaced["type"] = "fixture"
        replaced["path"] = str(snapshot)
        replaced["url"] = f"{item['url']}#frozen={snapshot.name}"
        frozen.append(replaced)
    return {"extends": base, "sources": frozen}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="live sources config name in configs/")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, help="overlay file name written in configs/")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    args = parser.parse_args(argv)
    snapshot_dir = args.snapshot_dir
    if not snapshot_dir.is_absolute():
        snapshot_dir = ROOT / snapshot_dir
    overlay = build_overlay(args.base, snapshot_dir.resolve(), args.config_dir)
    out = args.config_dir / args.out
    out.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
