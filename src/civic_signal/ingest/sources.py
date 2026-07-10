from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from civic_signal.config import ProjectContext


@dataclass(frozen=True)
class SourceDefinition:
    id: str
    table: str
    type: str
    path: Path | None
    parser_version: str
    license: str
    url: str
    auth_mode: str = "none"
    parser_args: dict[str, Any] = field(default_factory=dict)
    source_class: str = "unknown"
    access_policy: str = "unknown"
    terms_status: str = "unknown"
    terms_url: str = ""
    citation: str = ""
    priority: int = 0

    def parser_args_json(self) -> str:
        return json.dumps(self.parser_args, sort_keys=True)


class SourceRegistry:
    def __init__(self, sources: list[SourceDefinition]) -> None:
        self.sources = sources

    @classmethod
    def from_context(cls, context: ProjectContext) -> SourceRegistry:
        payload = cls._read_source_payload(context, context.sources_config, seen=set())
        raw_sources = payload.get("sources", [])
        sources = []
        for item in raw_sources:
            path = Path(item["path"]) if item.get("path") else None
            if path is not None and not path.is_absolute():
                path = context.root / path
            parser_args = item.get("parser_args") or {}
            if not isinstance(parser_args, dict):
                raise ValueError(f"parser_args for {item['id']} must be a mapping")
            sources.append(
                SourceDefinition(
                    id=str(item["id"]),
                    table=str(item["table"]),
                    type=str(item["type"]),
                    path=path,
                    parser_version=str(item["parser_version"]),
                    license=str(item["license"]),
                    url=str(item["url"]),
                    auth_mode=str(item.get("auth_mode", "none")),
                    parser_args=dict(parser_args),
                    source_class=str(
                        item.get("source_class")
                        or ("fixture" if str(item["type"]) == "fixture" else "production")
                    ),
                    access_policy=str(
                        item.get("access_policy")
                        or ("fixture_local" if str(item["type"]) == "fixture" else "unknown")
                    ),
                    terms_status=str(item.get("terms_status", "unknown")),
                    terms_url=str(item.get("terms_url", "")),
                    citation=str(item.get("citation", "")),
                    priority=int(
                        item.get(
                            "priority",
                            cls._default_priority(
                                str(
                                    item.get("source_class")
                                    or (
                                        "fixture"
                                        if str(item["type"]) == "fixture"
                                        else "production"
                                    )
                                )
                            ),
                        )
                    ),
                )
            )
        return cls(sources)

    @staticmethod
    def _default_priority(source_class: str) -> int:
        """Return a stable within-class priority; class rank remains the hard boundary."""
        return {
            "official_public": 400,
            "production_web": 300,
            "production": 250,
            "fixture": 50,
            "synthetic": 0,
        }.get(source_class.lower(), 100)

    @classmethod
    def _read_source_payload(
        cls, context: ProjectContext, name: str, seen: set[str]
    ) -> dict[str, Any]:
        if name in seen:
            chain = " -> ".join([*seen, name])
            raise ValueError(f"Cyclic source registry extends chain: {chain}")
        seen.add(name)
        payload = context.read_yaml(name)
        base_sources: list[dict[str, Any]] = []
        extends = payload.get("extends")
        if extends:
            if not isinstance(extends, str):
                raise ValueError(f"extends in {name} must be a config file name")
            base_sources = cls._read_source_payload(context, extends, seen).get("sources", [])
        overlay_sources = payload.get("sources", [])
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for item in [*base_sources, *overlay_sources]:
            source_id = str(item["id"])
            if source_id not in merged:
                order.append(source_id)
            merged[source_id] = dict(item)
        seen.remove(name)
        return {"sources": [merged[source_id] for source_id in order]}

    def by_table(self) -> dict[str, SourceDefinition]:
        return {source.table: source for source in self.sources}
