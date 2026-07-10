"""Recorded-contract live-source canaries for free/public HTTP adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from civic_signal.storage.io import write_json


@dataclass(frozen=True)
class CanaryCase:
    name: str
    url: str
    expect_success: bool = True
    timeout_seconds: float = 5.0


class LiveSourceCanaryRunner:
    """Exercise schema/empty/timeout style canaries against recorded contracts."""

    def __init__(
        self,
        fetcher: Callable[[str, float], tuple[int, bytes]] | None = None,
    ) -> None:
        self.fetcher = fetcher or self._default_fetch

    @staticmethod
    def _default_fetch(url: str, timeout: float) -> tuple[int, bytes]:
        request = Request(url, headers={"User-Agent": "civic-signal-canary/1.0"})
        with urlopen(request, timeout=timeout) as response:  # explicit canary URLs only
            return int(getattr(response, "status", 200) or 200), response.read(2048)

    def run(
        self,
        cases: list[CanaryCase] | None = None,
        *,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        cases = cases or self.default_cases()
        history: list[dict[str, Any]] = []
        injected: list[dict[str, Any]] = []
        all_passed = True
        for case in cases:
            try:
                status_code, body = self.fetcher(case.url, case.timeout_seconds)
                empty = len(body.strip()) == 0
                success = 200 <= status_code < 300 and not empty
                if case.expect_success:
                    passed = success
                    status = "ok" if passed else "empty_or_http_error"
                else:
                    # Adversarial / negative contracts must not look like success.
                    passed = not success
                    status = "failed_as_expected" if passed else "unexpected_success"
                if not passed:
                    all_passed = False
                record = {
                    "name": case.name,
                    "url": case.url,
                    "status": status,
                    "http_status": status_code,
                    "bytes": len(body),
                    "expect_success": case.expect_success,
                    "passed": passed,
                }
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                # Network failures: success-expected cases fail; negative cases pass.
                passed = not case.expect_success
                if not passed:
                    all_passed = False
                record = {
                    "name": case.name,
                    "url": case.url,
                    "status": "error" if case.expect_success else "failed_as_expected",
                    "error": str(exc),
                    "expect_success": case.expect_success,
                    "passed": passed,
                }
            history.append(record)
            if not case.expect_success:
                injected.append(
                    {
                        "name": case.name,
                        "status": record["status"]
                        if record["status"] != "unexpected_success"
                        else "success",
                    }
                )

        payload = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "all_passed": all_passed,
            "history": history,
            "injected_failure_results": injected,
        }
        if output_path:
            write_json(payload, __import__("pathlib").Path(output_path))
        return payload

    @staticmethod
    def default_cases() -> list[CanaryCase]:
        return [
            CanaryCase(
                name="fred_unrate_csv",
                url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
                expect_success=True,
            ),
            # Intentionally bad endpoint: must not be reported as success.
            CanaryCase(
                name="empty_or_missing_feed",
                url="https://example.invalid/civic-signal-canary-missing.json",
                expect_success=False,
                timeout_seconds=2.0,
            ),
        ]
