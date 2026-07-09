from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from civic_signal.features import FeatureBundle, filter_bundle_by_date, subset_bundle
from civic_signal.models.common import logit


def _empty_float_array() -> np.ndarray:
    return np.array([], dtype=np.float64)


def _empty_int_array() -> np.ndarray:
    return np.array([], dtype=np.int64)


@dataclass(frozen=True)
class StateSpaceData:
    poll_t: np.ndarray
    poll_s: np.ndarray
    poll_j: np.ndarray
    poll_o: np.ndarray
    poll_logit_y: np.ndarray
    poll_kappa: np.ndarray
    prior_logit: np.ndarray
    option_office: np.ndarray
    option_geography: np.ndarray
    option_race: np.ndarray
    race_option_keys: list[tuple[str, str]]
    pollster_ids: list[str]
    office_ids: list[str]
    geography_ids: list[str]
    race_ids: list[str]
    dims: tuple[int, int, int]
    metadata: dict[str, Any]
    # Margin-mode structures (P0.1/P0.2/P1.1): one party-signed two-party-margin
    # latent per two-way race, with a reverse random walk anchored on election
    # day. Options of margin races are reconstructed deterministically; options
    # not covered stay on the legacy per-option static path.
    margin_race_ids: list[str] = field(default_factory=list)
    margin_prior_logit: np.ndarray = field(default_factory=_empty_float_array)
    margin_hierarchy_loading: np.ndarray = field(default_factory=_empty_float_array)
    margin_race_office: np.ndarray = field(default_factory=_empty_int_array)
    margin_race_geography: np.ndarray = field(default_factory=_empty_int_array)
    margin_ref_option_index: np.ndarray = field(default_factory=_empty_int_array)
    margin_other_option_index: np.ndarray = field(default_factory=_empty_int_array)
    margin_poll_race: np.ndarray = field(default_factory=_empty_int_array)
    margin_poll_week: np.ndarray = field(default_factory=_empty_int_array)
    margin_poll_j: np.ndarray = field(default_factory=_empty_int_array)
    margin_poll_y: np.ndarray = field(default_factory=_empty_float_array)
    margin_poll_kappa: np.ndarray = field(default_factory=_empty_float_array)
    margin_max_weeks: int = 0
    residual_option_indices: np.ndarray = field(default_factory=_empty_int_array)
    legacy_poll_indices: np.ndarray = field(default_factory=_empty_int_array)


@dataclass(frozen=True)
class HyperPriors:
    sigma_state: float = 0.35
    tau_pollster: float = 0.04
    sigma_office: float = 0.08
    sigma_geography: float = 0.06
    sigma_race: float = 0.08
    # Prior scale for the weekly reverse-random-walk innovation sd (logit units).
    sigma_walk: float = 0.016


def _party_sign(party: object) -> float:
    value = str(party or "").upper()
    if value in {"DEM", "YES"}:
        return 1.0
    if value in {"REP", "NO"}:
        return -1.0
    return 0.0


def build_state_space_data(
    bundle: FeatureBundle,
    *,
    as_of: str,
    office_type: str | None = "president",
    cycle: int | None = None,
    prior_logit_by_key: dict[tuple[str, str], float] | None = None,
    poll_half_life_days: float = 21.0,
    process_drift_sd_per_sqrt_day: float = 0.0,
    pollster_house_effects: dict[tuple[str, str | None], float] | None = None,
    election_day_by_race: dict[str, date] | None = None,
    pollster_quality_weights: dict[str, float] | None = None,
) -> StateSpaceData:
    """Build poll-level tensors consumed by the hierarchical NumPyro backend.

    When `election_day_by_race` is provided, two-way races are additionally
    encoded as margin observations with weekly time buckets so the model can fit
    a reverse random walk anchored on election day. Races without an election
    date or with a non-two-way option structure remain on the legacy per-option
    static encoding.
    """

    cutoff = date.fromisoformat(as_of)
    catalog = bundle.race_catalog
    if office_type is not None and "office_type" in catalog.columns:
        catalog = catalog.filter(pl.col("office_type") == office_type)
    if cycle is not None and "cycle" in catalog.columns:
        catalog = catalog.filter(pl.col("cycle") == cycle)
    active = filter_bundle_by_date(subset_bundle(bundle, catalog), as_of)
    if active.polls.is_empty():
        return _empty_state_space_data({"as_of": as_of, "office_type": office_type, "cycle": cycle})

    poll_date_expr = pl.coalesce(["end_date", "start_date"]).alias("_poll_date")
    polls = (
        active.polls.with_columns(poll_date_expr)
        .filter(pl.col("_poll_date").is_not_null() & (pl.col("_poll_date") <= cutoff))
        .filter(pl.col("pct").is_not_null())
        .sort(["race_id", "option_id", "_poll_date", "pollster", "poll_id"])
    )
    if polls.is_empty():
        return _empty_state_space_data({"as_of": as_of, "office_type": office_type, "cycle": cycle})

    option_prior = _option_prior(active.options)
    option_party = _option_party(active.options)
    race_metadata = _race_metadata(active.race_catalog)
    keys = sorted(
        {
            (str(row["race_id"]), str(row["option_id"]))
            for row in polls.select(["race_id", "option_id"]).iter_rows(named=True)
        }
    )
    race_ids = sorted({race_id for race_id, _option_id in keys})
    office_ids = sorted(
        {race_metadata.get(race_id, {}).get("office", "unknown") for race_id in race_ids}
    )
    geography_ids = sorted(
        {race_metadata.get(race_id, {}).get("geography_group", "unknown") for race_id in race_ids}
    )
    pollsters = sorted(str(value) for value in polls["pollster"].fill_null("unknown").unique())
    key_index = {key: index for index, key in enumerate(keys)}
    pollster_index = {pollster: index for index, pollster in enumerate(pollsters)}
    race_index = {race_id: index for index, race_id in enumerate(race_ids)}
    office_index = {office_id: index for index, office_id in enumerate(office_ids)}
    geography_index = {geography_id: index for index, geography_id in enumerate(geography_ids)}
    min_date = polls["_poll_date"].min()
    if not hasattr(min_date, "toordinal"):
        min_date = date.fromisoformat(str(min_date))

    poll_t: list[int] = []
    poll_s: list[int] = []
    poll_j: list[int] = []
    poll_o: list[int] = []
    y: list[float] = []
    kappa: list[float] = []
    observation_weights: list[float] = []
    house_effect_values: list[float] = []
    poll_rows: list[dict[str, Any]] = []
    house_effect_lookup = pollster_house_effects or {}
    for row in polls.iter_rows(named=True):
        key = (str(row["race_id"]), str(row["option_id"]))
        pollster = str(row.get("pollster") or "unknown")
        option_id = str(row["option_id"])
        poll_date = row["_poll_date"]
        if not hasattr(poll_date, "toordinal"):
            poll_date = date.fromisoformat(str(poll_date))
        observed_share = min(0.999, max(0.001, float(row["pct"]) / 100.0))
        house_effect = house_effect_lookup.get(
            (pollster, option_id),
            house_effect_lookup.get((pollster, None), 0.0),
        )
        share = min(0.999, max(0.001, observed_share - house_effect))
        sample_size = max(float(row.get("sample_size") or 600.0), 50.0)
        # Track-record weighting hook: per-pollster multipliers (e.g. from a
        # historical-accuracy table) sharpen or discount effective sample size.
        quality_weight = _poll_quality_weight(row) * max(
            float((pollster_quality_weights or {}).get(pollster, 1.0)), 0.1
        )
        age_days = max((cutoff - poll_date).days, 0)
        recency_weight = _recency_weight(age_days, poll_half_life_days)
        observation_weight = max(quality_weight * recency_weight, 1e-3)
        effective_sample_size = sample_size * observation_weight
        share_sd = math.sqrt(max(share * (1.0 - share) / effective_sample_size, 1e-6))
        obs_sd_logit = share_sd / max(share * (1.0 - share), 1e-6)
        process_sd_logit = max(float(process_drift_sd_per_sqrt_day), 0.0) * math.sqrt(age_days)
        poll_t.append(int((poll_date - min_date).days))
        poll_s.append(key_index[key])
        poll_j.append(pollster_index[pollster])
        poll_o.append(0)
        y.append(logit(share))
        kappa.append(max(math.sqrt(obs_sd_logit**2 + process_sd_logit**2), 0.02))
        observation_weights.append(observation_weight)
        house_effect_values.append(float(house_effect))
        poll_rows.append(
            {
                "race_id": str(row["race_id"]),
                "option_id": option_id,
                "pollster": pollster,
                "poll_date": poll_date,
                "share": share,
                "sample_size": sample_size,
                "quality_weight": quality_weight,
                "pair_key": (
                    str(row["race_id"]),
                    pollster,
                    str(row.get("start_date") or ""),
                    str(row.get("end_date") or ""),
                    float(row.get("sample_size") or 0.0),
                ),
            }
        )

    prior_lookup = prior_logit_by_key or {}
    prior_logit = np.array(
        [
            float(prior_lookup[key]) if key in prior_lookup else logit(option_prior.get(key, 0.5))
            for key in keys
        ],
        dtype=np.float64,
    )
    option_office = np.array(
        [
            office_index[race_metadata.get(race_id, {}).get("office", "unknown")]
            for race_id, _option_id in keys
        ],
        dtype=np.int64,
    )
    option_geography = np.array(
        [
            geography_index[race_metadata.get(race_id, {}).get("geography_group", "unknown")]
            for race_id, _option_id in keys
        ],
        dtype=np.int64,
    )
    option_race = np.array(
        [race_index[race_id] for race_id, _option_id in keys],
        dtype=np.int64,
    )

    margin = _build_margin_structures(
        keys=keys,
        key_index=key_index,
        prior_logit=prior_logit,
        option_party=option_party,
        race_metadata=race_metadata,
        office_index=office_index,
        geography_index=geography_index,
        pollster_index=pollster_index,
        poll_rows=poll_rows,
        election_day_by_race=election_day_by_race or {},
        cutoff=cutoff,
        nonsampling_logit_floor=0.04,
    )

    return StateSpaceData(
        poll_t=np.array(poll_t, dtype=np.int64),
        poll_s=np.array(poll_s, dtype=np.int64),
        poll_j=np.array(poll_j, dtype=np.int64),
        poll_o=np.array(poll_o, dtype=np.int64),
        poll_logit_y=np.array(y, dtype=np.float64),
        poll_kappa=np.array(kappa, dtype=np.float64),
        prior_logit=prior_logit,
        option_office=option_office,
        option_geography=option_geography,
        option_race=option_race,
        race_option_keys=keys,
        pollster_ids=pollsters,
        office_ids=office_ids,
        geography_ids=geography_ids,
        race_ids=race_ids,
        dims=(len(keys), int(max(poll_t, default=0) + 1), len(pollsters)),
        metadata={
            "as_of": as_of,
            "office_type": office_type,
            "cycle": cycle,
            "poll_count": len(y),
            "poll_half_life_days": float(poll_half_life_days),
            "process_drift_sd_per_sqrt_day": float(process_drift_sd_per_sqrt_day),
            "temporal_process_variance": "poll_age_logit_variance",
            "observation_weight_min": float(min(observation_weights, default=0.0)),
            "observation_weight_max": float(max(observation_weights, default=0.0)),
            "observation_weight_mean": float(np.mean(observation_weights))
            if observation_weights
            else 0.0,
            "pollster_house_effect_adjustment_mean_abs": float(np.mean(np.abs(house_effect_values)))
            if house_effect_values
            else 0.0,
            "race_option_count": len(keys),
            "pollster_count": len(pollsters),
            "office_count": len(office_ids),
            "geography_count": len(geography_ids),
            "race_count": len(race_ids),
            "margin_race_count": len(margin["margin_race_ids"]),
            "margin_poll_count": int(margin["margin_poll_y"].size),
            "margin_max_weeks": int(margin["margin_max_weeks"]),
            "temporal_model": (
                "reverse_random_walk_weekly"
                if margin["margin_poll_y"].size
                else "static_recency_weighted"
            ),
            "hierarchy": {
                "office_ids": office_ids,
                "geography_ids": geography_ids,
                "race_ids": race_ids,
            },
        },
        **margin,
    )


def _build_margin_structures(
    *,
    keys: list[tuple[str, str]],
    key_index: dict[tuple[str, str], int],
    prior_logit: np.ndarray,
    option_party: dict[tuple[str, str], str],
    race_metadata: dict[str, dict[str, str]],
    office_index: dict[str, int],
    geography_index: dict[str, int],
    pollster_index: dict[str, int],
    poll_rows: list[dict[str, Any]],
    election_day_by_race: dict[str, date],
    cutoff: date,
    nonsampling_logit_floor: float,
) -> dict[str, Any]:
    empty = {
        "margin_race_ids": [],
        "margin_prior_logit": _empty_float_array(),
        "margin_hierarchy_loading": _empty_float_array(),
        "margin_race_office": _empty_int_array(),
        "margin_race_geography": _empty_int_array(),
        "margin_ref_option_index": _empty_int_array(),
        "margin_other_option_index": _empty_int_array(),
        "margin_poll_race": _empty_int_array(),
        "margin_poll_week": _empty_int_array(),
        "margin_poll_j": _empty_int_array(),
        "margin_poll_y": _empty_float_array(),
        "margin_poll_kappa": _empty_float_array(),
        "margin_max_weeks": 0,
        "residual_option_indices": np.arange(len(keys), dtype=np.int64),
        "legacy_poll_indices": np.arange(len(poll_rows), dtype=np.int64),
    }
    if not election_day_by_race:
        return empty

    options_by_race: dict[str, list[tuple[str, str]]] = {}
    for race_id, option_id in keys:
        options_by_race.setdefault(race_id, []).append((race_id, option_id))

    margin_race_ids: list[str] = []
    margin_prior: list[float] = []
    margin_loading: list[float] = []
    margin_office: list[int] = []
    margin_geography: list[int] = []
    ref_option_index: list[int] = []
    other_option_index: list[int] = []
    race_to_margin: dict[str, int] = {}
    ref_key_by_race: dict[str, tuple[str, str]] = {}
    other_key_by_race: dict[str, tuple[str, str]] = {}

    def _inv_logit(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    for race_id, race_keys in sorted(options_by_race.items()):
        if len(race_keys) != 2 or race_id not in election_day_by_race:
            continue
        signs = [_party_sign(option_party.get(key)) for key in race_keys]
        if signs[0] > 0 or (signs[0] != 0 and signs[1] == 0):
            ref, other = race_keys[0], race_keys[1]
            ref_sign, other_sign = signs[0], signs[1]
        elif signs[1] != 0:
            ref, other = race_keys[1], race_keys[0]
            ref_sign, other_sign = signs[1], signs[0]
        else:
            ref, other = race_keys[0], race_keys[1]
            ref_sign, other_sign = 0.0, 0.0
        loading = (ref_sign - other_sign) / 2.0
        prior_ref = _inv_logit(float(prior_logit[key_index[ref]]))
        prior_other = _inv_logit(float(prior_logit[key_index[other]]))
        two_party = min(max(prior_ref / max(prior_ref + prior_other, 1e-9), 1e-4), 1.0 - 1e-4)
        meta = race_metadata.get(race_id, {})
        race_to_margin[race_id] = len(margin_race_ids)
        ref_key_by_race[race_id] = ref
        other_key_by_race[race_id] = other
        margin_race_ids.append(race_id)
        margin_prior.append(logit(two_party))
        margin_loading.append(loading)
        margin_office.append(office_index.get(meta.get("office", "unknown"), 0))
        margin_geography.append(geography_index.get(meta.get("geography_group", "unknown"), 0))
        ref_option_index.append(key_index[ref])
        other_option_index.append(key_index[other])

    if not margin_race_ids:
        return empty

    paired: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row_index, row in enumerate(poll_rows):
        race_id = row["race_id"]
        if race_id not in race_to_margin:
            continue
        if (race_id, row["option_id"]) == ref_key_by_race[race_id]:
            role = "ref"
        elif (race_id, row["option_id"]) == other_key_by_race[race_id]:
            role = "other"
        else:
            continue
        paired.setdefault(row["pair_key"], {})[role] = {**row, "row_index": row_index}

    margin_poll_race: list[int] = []
    margin_poll_week: list[int] = []
    margin_poll_j: list[int] = []
    margin_poll_y: list[float] = []
    margin_poll_kappa: list[float] = []
    consumed_rows: set[int] = set()
    for pair in paired.values():
        if "ref" not in pair or "other" not in pair:
            continue
        ref_row, other_row = pair["ref"], pair["other"]
        race_id = ref_row["race_id"]
        total = ref_row["share"] + other_row["share"]
        if total <= 1e-9:
            continue
        two_party = min(max(ref_row["share"] / total, 1e-4), 1.0 - 1e-4)
        election_day = election_day_by_race[race_id]
        week = max(int((election_day - ref_row["poll_date"]).days // 7), 0)
        effective_sample_size = max(
            ref_row["sample_size"] * max(ref_row["quality_weight"], 1e-3), 50.0
        )
        share_sd = math.sqrt(max(two_party * (1.0 - two_party) / effective_sample_size, 1e-8))
        obs_sd_logit = share_sd / max(two_party * (1.0 - two_party), 1e-6)
        margin_poll_race.append(race_to_margin[race_id])
        margin_poll_week.append(week)
        margin_poll_j.append(pollster_index.get(ref_row["pollster"], 0))
        margin_poll_y.append(logit(two_party))
        margin_poll_kappa.append(math.sqrt(obs_sd_logit**2 + nonsampling_logit_floor**2))
        consumed_rows.add(ref_row["row_index"])
        consumed_rows.add(other_row["row_index"])

    if not margin_poll_y:
        return empty

    covered_option_indices = set(ref_option_index) | set(other_option_index)
    residual_option_indices = np.array(
        [index for index in range(len(keys)) if index not in covered_option_indices],
        dtype=np.int64,
    )
    legacy_poll_indices = np.array(
        [index for index in range(len(poll_rows)) if index not in consumed_rows],
        dtype=np.int64,
    )
    return {
        "margin_race_ids": margin_race_ids,
        "margin_prior_logit": np.array(margin_prior, dtype=np.float64),
        "margin_hierarchy_loading": np.array(margin_loading, dtype=np.float64),
        "margin_race_office": np.array(margin_office, dtype=np.int64),
        "margin_race_geography": np.array(margin_geography, dtype=np.int64),
        "margin_ref_option_index": np.array(ref_option_index, dtype=np.int64),
        "margin_other_option_index": np.array(other_option_index, dtype=np.int64),
        "margin_poll_race": np.array(margin_poll_race, dtype=np.int64),
        "margin_poll_week": np.array(margin_poll_week, dtype=np.int64),
        "margin_poll_j": np.array(margin_poll_j, dtype=np.int64),
        "margin_poll_y": np.array(margin_poll_y, dtype=np.float64),
        "margin_poll_kappa": np.array(margin_poll_kappa, dtype=np.float64),
        "margin_max_weeks": int(max(margin_poll_week, default=0)),
        "residual_option_indices": residual_option_indices,
        "legacy_poll_indices": legacy_poll_indices,
    }


def state_space_model(
    data: StateSpaceData,
    hyperpriors: HyperPriors | None = None,
    *,
    parameterization: str = "noncentered",
) -> None:  # pragma: no cover
    """NumPyro hierarchical polling model used by the opt-in NUTS backend.

    Margin mode (two-way races with election dates): one party-signed
    two-party-margin latent per race with office/geography effects entering the
    margin, plus a weekly reverse random walk anchored on election day. The
    legacy static per-option encoding remains for everything else.
    """

    if data.margin_poll_y.size and parameterization == "noncentered":
        _margin_state_space_model(data, hyperpriors or HyperPriors())
        return
    _legacy_state_space_model(data, hyperpriors, parameterization=parameterization)


def _margin_state_space_model(
    data: StateSpaceData,
    priors: HyperPriors,
) -> None:  # pragma: no cover
    try:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
    except ImportError as exc:
        raise RuntimeError("NumPyro/JAX are required; run `uv sync`.") from exc

    option_count = len(data.race_option_keys)
    pollster_count = max(len(data.pollster_ids), 1)
    office_count = max(len(data.office_ids), 1)
    geography_count = max(len(data.geography_ids), 1)
    margin_count = len(data.margin_race_ids)
    max_weeks = int(data.margin_max_weeks)

    sigma_office = numpyro.sample("sigma_office", dist.HalfNormal(priors.sigma_office))
    sigma_geography = numpyro.sample("sigma_geography", dist.HalfNormal(priors.sigma_geography))
    sigma_race = numpyro.sample("sigma_race", dist.HalfNormal(priors.sigma_race))
    tau_pollster = numpyro.sample("tau_pollster", dist.HalfNormal(priors.tau_pollster))
    sigma_walk = numpyro.sample("sigma_walk", dist.HalfNormal(priors.sigma_walk))

    office_z = numpyro.sample("office_z", dist.Normal(0.0, 1.0).expand([office_count]))
    geography_z = numpyro.sample("geography_z", dist.Normal(0.0, 1.0).expand([geography_count]))
    office_effect = numpyro.deterministic(
        "office_effect", _centered_effect(sigma_office * office_z)
    )
    geography_effect = numpyro.deterministic(
        "geography_effect", _centered_effect(sigma_geography * geography_z)
    )
    race_z = numpyro.sample("race_z", dist.Normal(0.0, 1.0).expand([margin_count]))
    numpyro.deterministic("race_effect", _centered_effect(sigma_race * race_z))

    loading = jnp.asarray(data.margin_hierarchy_loading)
    margin_t0 = numpyro.deterministic(
        "race_margin_logit",
        jnp.asarray(data.margin_prior_logit)
        + loading
        * (
            office_effect[jnp.asarray(data.margin_race_office)]
            + geography_effect[jnp.asarray(data.margin_race_geography)]
        )
        + sigma_race * race_z,
    )

    raw_pollster = numpyro.sample("pollster_raw", dist.Normal(0.0, 1.0).expand([pollster_count]))
    pollster_effect = numpyro.deterministic(
        "pollster_effect",
        tau_pollster * (raw_pollster - jnp.mean(raw_pollster)),
    )

    poll_race = jnp.asarray(data.margin_poll_race)
    if max_weeks > 0:
        walk_z = numpyro.sample("walk_z", dist.Normal(0.0, 1.0).expand([margin_count, max_weeks]))
        walk = jnp.cumsum(sigma_walk * walk_z, axis=1)
        poll_week = jnp.asarray(data.margin_poll_week)
        theta = jnp.where(
            poll_week > 0,
            margin_t0[poll_race] + walk[poll_race, jnp.clip(poll_week - 1, 0, max_weeks - 1)],
            margin_t0[poll_race],
        )
    else:
        theta = margin_t0[poll_race]

    margin_mu = theta + loading[poll_race] * pollster_effect[jnp.asarray(data.margin_poll_j)]
    numpyro.sample(
        "margin_poll_y",
        dist.Normal(margin_mu, jnp.asarray(data.margin_poll_kappa)),
        obs=jnp.asarray(data.margin_poll_y),
    )

    state_logit = jnp.zeros(option_count)
    state_logit = state_logit.at[jnp.asarray(data.margin_ref_option_index)].set(margin_t0)
    state_logit = state_logit.at[jnp.asarray(data.margin_other_option_index)].set(-margin_t0)

    residual_indices = np.asarray(data.residual_option_indices, dtype=np.int64)
    if residual_indices.size:
        sigma_state = numpyro.sample("sigma_state", dist.HalfNormal(priors.sigma_state))
        residual_z = numpyro.sample(
            "state_z", dist.Normal(0.0, 1.0).expand([int(residual_indices.size)])
        )
        residual_logit = jnp.asarray(data.prior_logit[residual_indices]) + sigma_state * residual_z
        state_logit = state_logit.at[jnp.asarray(residual_indices)].set(residual_logit)
        legacy = np.asarray(data.legacy_poll_indices, dtype=np.int64)
        if legacy.size:
            residual_set = {int(v) for v in residual_indices}
            legacy_obs = [index for index in legacy if int(data.poll_s[index]) in residual_set]
            if legacy_obs:
                obs_idx = np.array(legacy_obs, dtype=np.int64)
                mu = state_logit[jnp.asarray(data.poll_s[obs_idx])] + pollster_effect[
                    jnp.asarray(data.poll_j[obs_idx])
                ]
                numpyro.sample(
                    "poll_logit_y",
                    dist.Normal(mu, jnp.asarray(data.poll_kappa[obs_idx])),
                    obs=jnp.asarray(data.poll_logit_y[obs_idx]),
                )

    numpyro.deterministic("state_logit", state_logit)


def _legacy_state_space_model(
    data: StateSpaceData,
    hyperpriors: HyperPriors | None = None,
    *,
    parameterization: str = "noncentered",
) -> None:  # pragma: no cover
    try:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
    except ImportError as exc:
        raise RuntimeError("NumPyro/JAX are required; run `uv sync`.") from exc

    priors = hyperpriors or HyperPriors()
    state_count, _time_count, pollster_count = data.dims
    office_count = len(data.office_ids)
    geography_count = len(data.geography_ids)
    race_count = len(data.race_ids)
    sigma_state = numpyro.sample("sigma_state", dist.HalfNormal(priors.sigma_state))
    sigma_office = numpyro.sample("sigma_office", dist.HalfNormal(priors.sigma_office))
    sigma_geography = numpyro.sample("sigma_geography", dist.HalfNormal(priors.sigma_geography))
    sigma_race = numpyro.sample("sigma_race", dist.HalfNormal(priors.sigma_race))
    tau_pollster = numpyro.sample("tau_pollster", dist.HalfNormal(priors.tau_pollster))
    prior = jnp.asarray(data.prior_logit)
    option_office = jnp.asarray(data.option_office)
    option_geography = jnp.asarray(data.option_geography)
    option_race = jnp.asarray(data.option_race)
    if parameterization == "noncentered":
        office_z = numpyro.sample("office_z", dist.Normal(0.0, 1.0).expand([max(office_count, 1)]))
        geography_z = numpyro.sample(
            "geography_z", dist.Normal(0.0, 1.0).expand([max(geography_count, 1)])
        )
        race_z = numpyro.sample("race_z", dist.Normal(0.0, 1.0).expand([max(race_count, 1)]))
        state_z = numpyro.sample("state_z", dist.Normal(0.0, 1.0).expand([state_count]))
        office_effect = numpyro.deterministic(
            "office_effect", _centered_effect(sigma_office * office_z)
        )
        geography_effect = numpyro.deterministic(
            "geography_effect", _centered_effect(sigma_geography * geography_z)
        )
        race_effect = numpyro.deterministic("race_effect", _centered_effect(sigma_race * race_z))
        option_effect = sigma_state * state_z
        state_logit = numpyro.deterministic(
            "state_logit",
            prior
            + office_effect[option_office]
            + geography_effect[option_geography]
            + race_effect[option_race]
            + option_effect,
        )
    elif parameterization == "centered":
        office_effect = numpyro.deterministic(
            "office_effect",
            _centered_effect(
                numpyro.sample(
                    "office_effect_raw",
                    dist.Normal(0.0, sigma_office).expand([max(office_count, 1)]),
                )
            ),
        )
        geography_effect = numpyro.deterministic(
            "geography_effect",
            _centered_effect(
                numpyro.sample(
                    "geography_effect_raw",
                    dist.Normal(0.0, sigma_geography).expand([max(geography_count, 1)]),
                )
            ),
        )
        race_effect = numpyro.deterministic(
            "race_effect",
            _centered_effect(
                numpyro.sample(
                    "race_effect_raw",
                    dist.Normal(0.0, sigma_race).expand([max(race_count, 1)]),
                )
            ),
        )
        state_logit = numpyro.sample(
            "state_logit",
            dist.Normal(
                prior
                + office_effect[option_office]
                + geography_effect[option_geography]
                + race_effect[option_race],
                sigma_state,
            ),
        )
    else:
        raise ValueError("parameterization must be 'centered' or 'noncentered'")
    raw_pollster = numpyro.sample(
        "pollster_raw", dist.Normal(0.0, 1.0).expand([max(pollster_count, 1)])
    )
    pollster_effect = numpyro.deterministic(
        "pollster_effect",
        tau_pollster * (raw_pollster - jnp.mean(raw_pollster)),
    )
    mu = state_logit[jnp.asarray(data.poll_s)] + pollster_effect[jnp.asarray(data.poll_j)]
    numpyro.sample(
        "poll_logit_y",
        dist.Normal(mu, jnp.asarray(data.poll_kappa)),
        obs=jnp.asarray(data.poll_logit_y),
    )


def _centered_effect(values: Any) -> Any:  # pragma: no cover
    return values - values.mean()


def _option_prior(options: pl.DataFrame) -> dict[tuple[str, str], float]:
    if options.is_empty():
        return {}
    priors: dict[tuple[str, str], float] = {}
    for row in options.iter_rows(named=True):
        value = row.get("previous_vote_share")
        priors[(str(row["race_id"]), str(row["option_id"]))] = (
            float(value) if value is not None else 0.5
        )
    return priors


def _option_party(options: pl.DataFrame) -> dict[tuple[str, str], str]:
    if options.is_empty() or "party" not in options.columns:
        return {}
    return {
        (str(row["race_id"]), str(row["option_id"])): str(row.get("party") or "")
        for row in options.iter_rows(named=True)
    }


def _race_metadata(races: pl.DataFrame) -> dict[str, dict[str, str]]:
    if races.is_empty():
        return {}
    metadata: dict[str, dict[str, str]] = {}
    required = {"race_id", "office_type", "geography"}
    columns = [
        column for column in ["race_id", "office_type", "geography"] if column in races.columns
    ]
    if not required.issubset(set(columns)):
        return {}
    for row in races.select(columns).iter_rows(named=True):
        race_id = str(row["race_id"])
        geography = str(row.get("geography") or "unknown")
        metadata[race_id] = {
            "office": str(row.get("office_type") or "unknown").lower(),
            "geography_group": _geography_group(geography),
        }
    return metadata


def _geography_group(geography: str) -> str:
    if not geography:
        return "unknown"
    if "-" in geography:
        return geography.split("-", 1)[0]
    return geography


def _poll_quality_weight(row: dict[str, Any]) -> float:
    population = str(row.get("population") or row.get("population_full") or "").lower()
    methodology = str(row.get("methodology") or "").lower()
    population_weight = {"lv": 1.1, "likely": 1.1, "rv": 1.0, "registered": 1.0, "a": 0.85}
    methodology_weight = {
        "live_phone": 1.1,
        "mixed": 1.05,
        "online": 0.95,
        "ivr": 0.9,
        "text": 0.9,
    }
    pop_weight = next(
        (weight for key, weight in population_weight.items() if key in population),
        1.0,
    )
    method_weight = next(
        (weight for key, weight in methodology_weight.items() if key in methodology),
        1.0,
    )
    return max(pop_weight * method_weight, 0.1)


def _recency_weight(age_days: int, half_life_days: float) -> float:
    half_life = max(float(half_life_days), 1.0)
    return 0.5 ** (max(age_days, 0) / half_life)


def _empty_state_space_data(metadata: dict[str, Any]) -> StateSpaceData:
    return StateSpaceData(
        poll_t=np.array([], dtype=np.int64),
        poll_s=np.array([], dtype=np.int64),
        poll_j=np.array([], dtype=np.int64),
        poll_o=np.array([], dtype=np.int64),
        poll_logit_y=np.array([], dtype=np.float64),
        poll_kappa=np.array([], dtype=np.float64),
        prior_logit=np.array([], dtype=np.float64),
        option_office=np.array([], dtype=np.int64),
        option_geography=np.array([], dtype=np.int64),
        option_race=np.array([], dtype=np.int64),
        race_option_keys=[],
        pollster_ids=[],
        office_ids=[],
        geography_ids=[],
        race_ids=[],
        dims=(0, 0, 0),
        metadata=metadata
        | {
            "poll_count": 0,
            "race_option_count": 0,
            "pollster_count": 0,
            "office_count": 0,
            "geography_count": 0,
            "race_count": 0,
            "margin_race_count": 0,
            "margin_poll_count": 0,
            "margin_max_weeks": 0,
            "temporal_model": "static_recency_weighted",
            "hierarchy": {"office_ids": [], "geography_ids": [], "race_ids": []},
        },
    )
