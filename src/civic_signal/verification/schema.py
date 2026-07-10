"""Runtime JSON Schema validation for publication-bound artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from civic_signal.storage.io import read_json


def artifact_schema_errors(
    root: Path,
    payload: dict[str, Any],
    schema_name: str,
) -> list[str]:
    """Return stable, human-readable validation failures for an artifact payload."""
    schema_path = root / "schemas" / "artifact_contracts" / schema_name
    if not schema_path.exists():
        packaged_schema = (
            Path(__file__).resolve().parents[1] / "schemas" / "artifact_contracts" / schema_name
        )
        repository_schema = Path(__file__).resolve().parents[3] / "schemas" / "artifact_contracts"
        for candidate in (packaged_schema, repository_schema / schema_name):
            if candidate.exists():
                schema_path = candidate
                break
    if not schema_path.exists():
        return [f"schema file is missing: {schema_path}"]
    schema = read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in errors
    ]


def require_artifact_schema(
    root: Path,
    payload: dict[str, Any],
    schema_name: str,
) -> None:
    """Raise when a publication-bound payload violates its checked-in schema."""
    errors = artifact_schema_errors(root, payload, schema_name)
    if errors:
        raise ValueError(f"{schema_name} validation failed: {'; '.join(errors)}")
