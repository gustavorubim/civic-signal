from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from election_outcomes.config import ProjectContext


@dataclass(frozen=True)
class SourceDefinition:
    id: str
    table: str
    type: str
    path: Path
    parser_version: str
    license: str
    url: str


class SourceRegistry:
    def __init__(self, sources: list[SourceDefinition]) -> None:
        self.sources = sources

    @classmethod
    def from_context(cls, context: ProjectContext) -> SourceRegistry:
        raw_sources = context.read_yaml("sources.yaml").get("sources", [])
        sources = []
        for item in raw_sources:
            path = Path(item["path"])
            if not path.is_absolute():
                path = context.root / path
            sources.append(
                SourceDefinition(
                    id=str(item["id"]),
                    table=str(item["table"]),
                    type=str(item["type"]),
                    path=path,
                    parser_version=str(item["parser_version"]),
                    license=str(item["license"]),
                    url=str(item["url"]),
                )
            )
        return cls(sources)

    def by_table(self) -> dict[str, SourceDefinition]:
        return {source.table: source for source in self.sources}
