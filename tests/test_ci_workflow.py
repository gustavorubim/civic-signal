"""CI runs the same gate as AGENTS.md (audit P2-2/3)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_runs_lint_format_and_coverage_gate() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    commands = " ".join(str(step.get("run", "")) for step in workflow["jobs"]["test"]["steps"])
    assert "ruff check" in commands
    assert "ruff format --check" in commands
    assert "--cov-fail-under=90" in commands
