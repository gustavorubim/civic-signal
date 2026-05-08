"""Ingestion layer."""

from election_outcomes.ingest.sources import SourceDefinition, SourceRegistry
from election_outcomes.ingest.sync import SyncResult, SyncRunner

__all__ = ["SourceDefinition", "SourceRegistry", "SyncResult", "SyncRunner"]
