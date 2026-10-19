"""Numba kernel and pure-Python fallback agree (P1-1 performance path)."""

from __future__ import annotations

import numpy as np

from civic_signal.performance import kernels
from civic_signal.performance.kernels import simulate_binary_draw_arrays


def _inputs():
    rng = np.random.default_rng(0)
    return (
        np.array([0.48, 0.55, 0.9]),
        np.array([1000.0, 2000.0, 500.0]),
        rng.normal(0, 0.03, 50),
        rng.normal(0, 0.05, (3, 50)),
        np.array([1.0, -1.0, 0.0]),
        np.ones(50),
    )


def test_python_kernel_matches_fallback() -> None:
    first, turnout, national, local, signs, mult = _inputs()
    fallback = simulate_binary_draw_arrays(
        first, turnout, national, local, False, party_signs=signs, turnout_multipliers=mult
    )
    kernel = getattr(kernels.binary_draw_kernel, "py_func", kernels.binary_draw_kernel)
    direct = kernel(first, turnout, national, local, signs, mult)
    for left, right in zip(fallback, direct, strict=True):
        np.testing.assert_allclose(left, right)
    shares = direct[5].reshape(3, 50, 2)
    np.testing.assert_allclose(shares.sum(axis=2), 1.0)
    assert shares.min() >= 0.02 and shares.max() <= 0.98


def test_numba_path_matches_fallback() -> None:
    first, turnout, national, local, signs, mult = _inputs()
    fast = simulate_binary_draw_arrays(
        first, turnout, national, local, True, party_signs=signs, turnout_multipliers=mult
    )
    slow = simulate_binary_draw_arrays(
        first, turnout, national, local, False, party_signs=signs, turnout_multipliers=mult
    )
    for left, right in zip(fast, slow, strict=True):
        np.testing.assert_allclose(left, right)
    assert kernels.configure_numba_threads(None) in (None,) or isinstance(
        kernels.configure_numba_threads(None), int
    )
