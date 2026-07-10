"""Aggregate scientific CI checks (M7) into a recompute-style report."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.scientific.contract_parity import ContractParityChecker
from civic_signal.scientific.golden import validate_golden_bundle
from civic_signal.scientific.live_canaries import CanaryCase, LiveSourceCanaryRunner
from civic_signal.scientific.mutation import run_standard_mutation_probes
from civic_signal.scientific.parity import numerical_parity_report
from civic_signal.scientific.properties import run_offline_property_suite
from civic_signal.storage.io import write_json, write_text


class ScientificVerificationRunner:
    """Run property, parity, mutation, contract, canary, and optional NUTS smoke."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def verify(
        self,
        *,
        include_live_canaries: bool = False,
        include_nuts_smoke: bool = False,
        fixture_root: Path | None = None,
    ) -> dict[str, Any]:
        failures: list[str] = []
        missing_required_suites: list[str] = []

        properties = run_offline_property_suite()
        if not properties.get("passed"):
            failures.extend(f"property:{item}" for item in properties.get("failures") or [])
            if properties.get("missing_property_families"):
                missing_required_suites.append("offline property suite")
                failures.append("offline property suite is incomplete")

        parity = numerical_parity_report()
        if not parity.get("python_numba_match", False):
            failures.append("python/numba numerical mismatch")
        if not parity.get("serial_parallel_match", False):
            failures.append("serial/parallel numerical mismatch")

        resolved_fixtures = fixture_root
        if resolved_fixtures is None:
            candidate = self.context.root / "tests" / "golden_fixtures"
            if candidate.exists():
                resolved_fixtures = candidate
        golden = validate_golden_bundle(resolved_fixtures)
        if not golden.get("passed"):
            failures.extend(f"golden:{item}" for item in golden.get("failures") or [])

        expected_fp = (golden.get("cases") or {}).get("parity_fingerprint") or {}
        if expected_fp.get("python_fingerprint"):
            if parity.get("python_fingerprint") != expected_fp["python_fingerprint"]:
                failures.append("python_fingerprint drift vs golden fixture")
            if parity.get("numba_available") and expected_fp.get("numba_fingerprint"):
                if parity.get("numba_fingerprint") != expected_fp["numba_fingerprint"]:
                    failures.append("numba_fingerprint drift vs golden fixture")

        mutations = run_standard_mutation_probes(self.context.root)
        mutation_incomplete = bool(mutations.get("incomplete_mutation_families")) or not bool(
            mutations.get("actual_verifier_paths")
        )
        if mutation_incomplete or not mutations.get("all_mutations_detected"):
            failures.append("actual reward/publication mutation suite is incomplete or survived")
            missing_required_suites.append("actual reward/publication source mutation suite")

        contract = ContractParityChecker(self.context.root).run()
        if not contract.get("passed"):
            contract_issues = (
                contract.get("stale_claim_details") or contract.get("failure_reasons") or []
            )
            failures.extend(f"contract:{item}" for item in contract_issues)

        if include_live_canaries:
            canaries = LiveSourceCanaryRunner().run()
        else:
            # Deterministic offline canaries: success body + intentional failure.
            def _offline_fetch(url: str, timeout: float) -> tuple[int, bytes]:
                del timeout
                if "missing" in url or "invalid" in url:
                    raise OSError(f"offline canary forced failure for {url}")
                return 200, b"DATE,VALUE\n2026-01-01,4.0\n"

            canaries = LiveSourceCanaryRunner(fetcher=_offline_fetch).run(
                [
                    CanaryCase(name="offline_ok", url="https://example.test/ok.csv"),
                    CanaryCase(
                        name="offline_missing",
                        url="https://example.invalid/missing.json",
                        expect_success=False,
                    ),
                ]
            )
        if not canaries.get("all_passed"):
            failures.append("live/offline canaries failed")

        nuts_report: dict[str, Any] | None = None
        if include_nuts_smoke:
            from civic_signal.scientific.nuts_smoke import run_nuts_smoke

            nuts_report = run_nuts_smoke()
            if not nuts_report.get("ok"):
                failures.append("nuts smoke failed")

        checks_passed = not failures
        missing_optional_suites: list[str] = []
        if not include_nuts_smoke:
            missing_optional_suites.append("tiny real NUTS smoke was not executed")
        if not include_live_canaries:
            missing_optional_suites.append("real free-web live canaries were not executed")
        # Default offline gate: property/parity/mutation/golden/contract/canary.
        # Optional suites raise evidence quality but do not fail the CI entry point.
        evidence_state = (
            "pass"
            if checks_passed and not missing_optional_suites
            else ("fail" if failures else "checks_passed_optional_suites_pending")
        )
        out_dir = self.context.artifacts_dir / "scientific"
        report_path = out_dir / "scientific_report.json"
        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "passed": checks_passed,
            "checks_passed": checks_passed,
            "evidence_state": evidence_state,
            "failures": failures,
            "missing_required_suites": missing_required_suites,
            "missing_optional_suites": missing_optional_suites,
            "properties": properties,
            "parity": parity,
            "golden": golden,
            "mutations": mutations,
            "contract_parity": {
                "passed": contract.get("passed"),
                "stale_claims": contract.get("stale_claims"),
                "checked_documents": contract.get("checked_documents"),
            },
            "canaries": {
                "all_passed": canaries.get("all_passed"),
                "live": include_live_canaries,
                "history": canaries.get("history"),
            },
            "nuts_smoke": nuts_report,
            "report_path": str(report_path),
            "exit_nonzero": not checks_passed,
        }
        write_json(payload, report_path)
        write_text(
            f"scientific verification state={evidence_state} checks_passed={checks_passed} "
            f"failures={len(failures)} optional_pending={len(missing_optional_suites)}\n",
            out_dir / "scientific_report.txt",
        )
        return payload
