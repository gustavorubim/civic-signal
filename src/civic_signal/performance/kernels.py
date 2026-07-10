from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - import path depends on optional runtime availability
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for unsupported platforms
    NUMBA_AVAILABLE = False
    get_num_threads = None
    set_num_threads = None
    prange = range

    def njit(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator


@njit(cache=True, parallel=True)
def binary_draw_kernel(  # pragma: no cover - executed inside compiled Numba code
    first_shares: np.ndarray,
    turnout_bases: np.ndarray,
    national_errors: np.ndarray,
    local_errors: np.ndarray,
    party_signs: np.ndarray,
    turnout_multipliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    race_count = first_shares.shape[0]
    draw_count = national_errors.shape[0]
    row_count = race_count * draw_count * 2
    draw_ids = np.empty(row_count, dtype=np.int64)
    correlated_error_draw_ids = np.empty(row_count, dtype=np.int64)
    race_indices = np.empty(row_count, dtype=np.int64)
    option_indices = np.empty(row_count, dtype=np.int64)
    turnouts = np.empty(row_count, dtype=np.int64)
    vote_shares = np.empty(row_count, dtype=np.float64)
    winners = np.empty(row_count, dtype=np.bool_)

    for race_index in prange(race_count):
        first_share = first_shares[race_index]
        turnout_base = turnout_bases[race_index]
        sign = party_signs[race_index]
        for draw_id in range(draw_count):
            share_zero = (
                first_share + sign * national_errors[draw_id] + local_errors[race_index, draw_id]
            )
            if share_zero < 0.02:
                share_zero = 0.02
            elif share_zero > 0.98:
                share_zero = 0.98
            share_one = 1.0 - share_zero
            turnout = int(np.rint(turnout_base * turnout_multipliers[draw_id]))
            winner_zero = share_zero >= share_one
            offset = (race_index * draw_count + draw_id) * 2

            draw_ids[offset] = draw_id
            correlated_error_draw_ids[offset] = draw_id
            race_indices[offset] = race_index
            option_indices[offset] = 0
            turnouts[offset] = turnout
            vote_shares[offset] = share_zero
            winners[offset] = winner_zero

            second = offset + 1
            draw_ids[second] = draw_id
            correlated_error_draw_ids[second] = draw_id
            race_indices[second] = race_index
            option_indices[second] = 1
            turnouts[second] = turnout
            vote_shares[second] = share_one
            winners[second] = not winner_zero

    return (
        draw_ids,
        correlated_error_draw_ids,
        race_indices,
        option_indices,
        turnouts,
        vote_shares,
        winners,
    )


def simulate_binary_draw_arrays(
    first_shares: np.ndarray,
    turnout_bases: np.ndarray,
    national_errors: np.ndarray,
    local_errors: np.ndarray,
    use_numba: bool,
    party_signs: np.ndarray | None = None,
    turnout_multipliers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if party_signs is None:
        party_signs = np.ones(first_shares.shape[0], dtype=np.float64)
    if turnout_multipliers is None:
        turnout_multipliers = np.ones(national_errors.shape[0], dtype=np.float64)
    party_signs = np.ascontiguousarray(party_signs, dtype=np.float64)
    turnout_multipliers = np.ascontiguousarray(turnout_multipliers, dtype=np.float64)
    if use_numba and NUMBA_AVAILABLE:
        return binary_draw_kernel(
            first_shares,
            turnout_bases,
            national_errors,
            local_errors,
            party_signs,
            turnout_multipliers,
        )
    return python_binary_draw_kernel(
        first_shares,
        turnout_bases,
        national_errors,
        local_errors,
        party_signs,
        turnout_multipliers,
    )


def python_binary_draw_kernel(
    first_shares: np.ndarray,
    turnout_bases: np.ndarray,
    national_errors: np.ndarray,
    local_errors: np.ndarray,
    party_signs: np.ndarray,
    turnout_multipliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    race_count = first_shares.shape[0]
    draw_count = national_errors.shape[0]
    row_count = race_count * draw_count * 2
    draw_ids = np.empty(row_count, dtype=np.int64)
    correlated_error_draw_ids = np.empty(row_count, dtype=np.int64)
    race_indices = np.empty(row_count, dtype=np.int64)
    option_indices = np.empty(row_count, dtype=np.int64)
    turnouts = np.empty(row_count, dtype=np.int64)
    vote_shares = np.empty(row_count, dtype=np.float64)
    winners = np.empty(row_count, dtype=np.bool_)

    for race_index in range(race_count):
        for draw_id in range(draw_count):
            share_zero = np.clip(
                first_shares[race_index]
                + party_signs[race_index] * national_errors[draw_id]
                + local_errors[race_index, draw_id],
                0.02,
                0.98,
            )
            share_one = 1.0 - share_zero
            turnout = round(turnout_bases[race_index] * turnout_multipliers[draw_id])
            winner_zero = bool(share_zero >= share_one)
            offset = (race_index * draw_count + draw_id) * 2
            draw_ids[offset : offset + 2] = draw_id
            correlated_error_draw_ids[offset : offset + 2] = draw_id
            race_indices[offset : offset + 2] = race_index
            option_indices[offset : offset + 2] = [0, 1]
            turnouts[offset : offset + 2] = turnout
            vote_shares[offset : offset + 2] = [share_zero, share_one]
            winners[offset : offset + 2] = [winner_zero, not winner_zero]
    return (
        draw_ids,
        correlated_error_draw_ids,
        race_indices,
        option_indices,
        turnouts,
        vote_shares,
        winners,
    )


def configure_numba_threads(thread_count: int | None) -> int | None:
    if not NUMBA_AVAILABLE or get_num_threads is None:
        return None
    if thread_count and thread_count > 0 and set_num_threads is not None:
        set_num_threads(thread_count)
    return int(get_num_threads())
