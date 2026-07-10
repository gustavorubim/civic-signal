"""Numba/Python and serial/parallel numerical parity reports."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from civic_signal.performance.kernels import (
    NUMBA_AVAILABLE,
    configure_numba_threads,
    simulate_binary_draw_arrays,
)


def _fingerprint(arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def numerical_parity_report(
    *,
    seed: int = 20260508,
    race_count: int = 4,
    draw_count: int = 64,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compare Python vs Numba kernels and serial vs parallel Numba when available."""
    rng = np.random.default_rng(seed)
    first_shares = rng.uniform(0.2, 0.8, size=race_count).astype(np.float64)
    turnout_bases = rng.integers(50_000, 200_000, size=race_count).astype(np.float64)
    national_errors = rng.normal(0.0, 0.02, size=draw_count).astype(np.float64)
    local_errors = rng.normal(0.0, 0.03, size=(race_count, draw_count)).astype(np.float64)
    party_signs = np.ones(race_count, dtype=np.float64)
    turnout_multipliers = np.ones(draw_count, dtype=np.float64)

    python = simulate_binary_draw_arrays(
        first_shares,
        turnout_bases,
        national_errors,
        local_errors,
        use_numba=False,
        party_signs=party_signs,
        turnout_multipliers=turnout_multipliers,
    )
    report: dict[str, Any] = {
        "numba_available": NUMBA_AVAILABLE,
        "python_fingerprint": _fingerprint(python),
        "python_numba_match": None,
        "serial_parallel_match": None,
        "numba_fingerprint": None,
        "parallel_fingerprint": None,
        "atol": atol,
    }
    if not NUMBA_AVAILABLE:
        report["python_numba_match"] = True
        report["serial_parallel_match"] = True
        report["note"] = "Numba unavailable; Python path is the only engine."
        return report

    previous = configure_numba_threads(1)
    serial = simulate_binary_draw_arrays(
        first_shares,
        turnout_bases,
        national_errors,
        local_errors,
        use_numba=True,
        party_signs=party_signs,
        turnout_multipliers=turnout_multipliers,
    )
    report["numba_fingerprint"] = _fingerprint(serial)
    report["python_numba_match"] = all(
        np.allclose(left, right, atol=atol, rtol=0.0)
        for left, right in zip(python, serial, strict=True)
    )

    configure_numba_threads(2)
    parallel = simulate_binary_draw_arrays(
        first_shares,
        turnout_bases,
        national_errors,
        local_errors,
        use_numba=True,
        party_signs=party_signs,
        turnout_multipliers=turnout_multipliers,
    )
    report["parallel_fingerprint"] = _fingerprint(parallel)
    report["serial_parallel_match"] = all(
        np.allclose(left, right, atol=atol, rtol=0.0)
        for left, right in zip(serial, parallel, strict=True)
    )
    if previous is not None:
        configure_numba_threads(previous)
    return report
