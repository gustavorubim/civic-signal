"""National-environment resolution: explicit override > measured > none (audit P0-6).

``fixtures/environment_override.json`` pins an analyst scenario. The audit decision (Oct 2026)
was to remove it, so the measured generic-ballot median in
``fixtures/measured_environment.json`` (``scripts/measure_environment.py``) is used. The
resolved value, its source and its date are recorded in every run's ``environment.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_environment(fixtures_dir: Path) -> dict[str, Any]:
    override = fixtures_dir / "environment_override.json"
    measured = fixtures_dir / "measured_environment.json"
    if override.exists():
        payload = json.loads(override.read_text(encoding="utf-8"))
        return {
            "kind": "override",
            "generic_ballot_margin_d": float(payload["generic_ballot_margin_d"]),
            "source": str(override.name),
            "as_of": payload.get("set_at"),
            "reason": payload.get("reason"),
            "measured_margin_d": _measured_value(measured),
        }
    if measured.exists():
        payload = json.loads(measured.read_text(encoding="utf-8"))
        return {
            "kind": "measured",
            "generic_ballot_margin_d": float(payload["generic_ballot_margin_d"]),
            "source": payload.get("source"),
            "as_of": payload.get("measured_at"),
            "method": payload.get("method"),
            "aggregators": payload.get("aggregators", {}),
            "input_sha256": payload.get("input_sha256"),
        }
    return {"kind": "none", "generic_ballot_margin_d": None, "source": None, "as_of": None}


def _measured_value(path: Path) -> float | None:
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["generic_ballot_margin_d"])
