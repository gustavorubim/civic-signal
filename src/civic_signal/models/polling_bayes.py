from __future__ import annotations

import hashlib
import math
import re
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl

from civic_signal.features import FeatureBundle
from civic_signal.inference.failover import (
    FailoverPolicy,
    LoadedPreviousPosterior,
    PreviousPosteriorCompatibilityError,
    dispatch_failover,
    load_previous_posterior_artifact,
)
from civic_signal.models.common import inv_logit, logit, normal_cdf, normalize_rows
from civic_signal.models.polling_kalman import (
    HouseEffectEstimate,
    KalmanPollingModel,
    PollObservation,
)


class BayesianPollingModel(KalmanPollingModel):
    """Opt-in Bayesian polling component with a conjugate logit-normal update.

    This is the operational Phase 1 bridge: it preserves the existing component
    schema and provenance surface while producing posterior draws and diagnostics.
    Full NumPyro NUTS can replace this fitter behind the same public methods.
    """

    component = "polling"
    POSTERIOR_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "draw_id": pl.Int64,
        "chain_id": pl.Int64,
        "race_id": pl.String,
        "option_id": pl.String,
        "geography": pl.String,
        "trajectory_date": pl.Date,
        "latent_logit": pl.Float64,
        "latent_share": pl.Float64,
        "systematic_error": pl.Float64,
        "pollster_effect": pl.Float64,
        "diagnostic_only": pl.Boolean,
    }

    def __init__(self, config: dict[str, object] | None = None, as_of: str | None = None) -> None:
        super().__init__(config=config, as_of=as_of)
        config = config or {}
        bayesian_config = dict(config.get("bayesian", {}))
        state_space = dict(bayesian_config.get("state_space", {}))
        self.backend = str(
            config.get("_bayesian_backend") or bayesian_config.get("backend", "analytic")
        )
        self.posterior_draw_count = int(
            bayesian_config.get("posterior_draw_count", config.get("simulation_count", 1000))
        )
        self.posterior_draw_count = max(min(self.posterior_draw_count, 5000), 100)
        self.initial_state_logit_sd = float(state_space.get("initial_state_logit_sd", 0.5))
        self.election_day_extra_sd = float(state_space.get("election_day_extra_sd", 0.025))
        self.forecast_drift_sd_per_sqrt_day = float(
            state_space.get("forecast_drift_sd_per_sqrt_day", 0.006)
        )
        self.nonsampling_logit_floor = float(
            dict(bayesian_config.get("observation", {})).get("nonsampling_logit_floor", 0.02)
        )
        observation_config = dict(bayesian_config.get("observation", {}))
        self.mode_bias_share = {
            str(key).lower(): float(value)
            for key, value in dict(observation_config.get("mode_bias_share", {})).items()
        }
        self.sponsor_bias_share = {
            str(key).lower(): float(value)
            for key, value in dict(observation_config.get("sponsor_bias_share", {})).items()
        }
        self.mode_nonsampling_sd = {
            str(key).lower(): float(value)
            for key, value in dict(observation_config.get("mode_nonsampling_sd", {})).items()
        }
        self.sponsor_nonsampling_sd = {
            str(key).lower(): float(value)
            for key, value in dict(observation_config.get("sponsor_nonsampling_sd", {})).items()
        }
        self.undecided_nonsampling_multiplier = float(
            observation_config.get("undecided_nonsampling_multiplier", 0.20)
        )
        self.parameterization = str(state_space.get("parameterization", "noncentered"))
        self.failover_policy = FailoverPolicy.from_config(config)
        nuts_config = dict(bayesian_config.get("nuts", {}))
        failover_config = dict(nuts_config.get("failover", config.get("failover", {})))
        self._previous_posterior_config = dict(failover_config.get("previous_posterior", {}))
        self._config = config
        self._fallback_audit_override: dict[str, Any] | None = None
        self._fundamentals_prior = self._fundamentals_prior_lookup(
            config.get("_fundamentals_prior_rows", [])
        )
        self._cached_posterior_draws: pl.DataFrame = self._empty_posterior_draws()
        self._cached_diagnostics: dict[str, Any] = self._empty_diagnostics()

    def posterior_draws(self, bundle: FeatureBundle) -> pl.DataFrame:
        self._ensure_fit(bundle)
        return self._cached_posterior_draws

    def diagnostics(self, bundle: FeatureBundle | None = None) -> dict[str, Any]:
        if bundle is not None:
            self._ensure_fit(bundle)
        return dict(self._cached_diagnostics)

    def _prepare_likelihood_polls(
        self,
        bundle: FeatureBundle,
        polls: pl.DataFrame,
        *,
        split_overlap_precision: bool,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Canonicalize one row per question/option and build honest estimands."""
        option_rows = list(bundle.options.iter_rows(named=True))
        options_by_race: dict[str, set[str]] = {}
        party_by_key: dict[tuple[str, str], str] = {}
        for row in option_rows:
            race_id = str(row["race_id"])
            option_id = str(row["option_id"])
            options_by_race.setdefault(race_id, set()).add(option_id)
            party_by_key[(race_id, option_id)] = str(row.get("party") or "").upper()
        unsupported = {race_id for race_id, options in options_by_race.items() if len(options) > 2}
        grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
        duplicate_rows = 0
        for row in sorted(
            polls.iter_rows(named=True),
            key=lambda item: (
                str(item.get("race_id") or ""),
                self._question_identity(item),
                str(item.get("option_id") or ""),
                str(item.get("source_hash") or ""),
            ),
        ):
            race_id = str(row.get("race_id") or "")
            option_id = str(row.get("option_id") or "")
            if race_id in unsupported or option_id not in options_by_race.get(race_id, set()):
                continue
            question_id = self._question_identity(row)
            group = grouped.setdefault((race_id, question_id), {})
            duplicate_rows += int(option_id in group)
            group[option_id] = dict(row)

        prepared: list[dict[str, object]] = []
        paired_questions = 0
        incomplete_binary_questions = 0
        incomplete_binary_races: set[str] = set()
        unallocated_values: list[float] = []
        for (race_id, question_id), option_map in sorted(grouped.items()):
            expected_options = options_by_race.get(race_id, set())
            complete_binary = len(expected_options) == 2 and expected_options.issubset(option_map)
            if len(expected_options) == 2 and not complete_binary:
                incomplete_binary_questions += 1
                incomplete_binary_races.add(race_id)
                continue
            total = sum(max(float(row.get("pct") or 0.0), 0.0) for row in option_map.values())
            unallocated = max(100.0 - total, 0.0) / 100.0 if complete_binary else 0.0
            paired_questions += int(complete_binary)
            unallocated_values.append(unallocated)
            overlap = max(len(option_map), 1)
            for option_id, raw_row in sorted(option_map.items()):
                row = dict(raw_row)
                raw_pct = max(float(row.get("pct") or 0.0), 0.0)
                normalized = raw_pct / total * 100.0 if complete_binary and total > 0 else raw_pct
                party = party_by_key.get((race_id, option_id), "")
                sign = 1.0 if party in {"DEM", "YES"} else -1.0 if party in {"REP", "NO"} else 0.0
                methodology = str(row.get("methodology") or "unknown").lower()
                sponsor = str(row.get("sponsor_class") or "unknown").lower()
                bias = self._effect_lookup(self.mode_bias_share, methodology) + self._effect_lookup(
                    self.sponsor_bias_share, sponsor
                )
                corrected = min(max(normalized - sign * bias * 100.0, 0.1), 99.9)
                mode_sd = self._effect_lookup(self.mode_nonsampling_sd, methodology)
                sponsor_sd = self._effect_lookup(self.sponsor_nonsampling_sd, sponsor)
                nonsampling_sd = math.sqrt(
                    self.min_nonsampling_error**2
                    + mode_sd**2
                    + sponsor_sd**2
                    + (unallocated * self.undecided_nonsampling_multiplier) ** 2
                )
                original_sample = float(row.get("sample_size") or self.default_sample_size)
                row.update(
                    {
                        "pct": corrected,
                        "_likelihood_question_id": question_id,
                        "_likelihood_overlap_count": overlap,
                        "_likelihood_unallocated_share": unallocated,
                        "_likelihood_nonsampling_sd": nonsampling_sd,
                        "_likelihood_original_sample_size": original_sample,
                        "sample_size": original_sample / overlap
                        if split_overlap_precision
                        else original_sample,
                    }
                )
                prepared.append(row)
        audit = {
            "question_count": len(grouped),
            "paired_binary_question_count": paired_questions,
            "incomplete_binary_question_count": incomplete_binary_questions,
            "incomplete_binary_question_races": sorted(incomplete_binary_races),
            "duplicate_question_option_rows_removed": duplicate_rows,
            "unsupported_multi_option_races": sorted(unsupported),
            "unsupported_multi_option_race_count": len(unsupported),
            "blocked_race_ids": sorted(unsupported | incomplete_binary_races),
            "multi_option_estimand_status": "withheld_no_coherent_k_category_likelihood",
            "undecided_other_treatment": "proportional_two_party_renormalization",
            "mean_unallocated_share": float(np.mean(unallocated_values))
            if unallocated_values
            else 0.0,
            "mode_effects_explicit": True,
            "sponsor_effects_explicit": True,
            "shared_question_overlap_handled": True,
        }
        return (pl.DataFrame(prepared) if prepared else polls.clear()), audit

    @staticmethod
    def _question_identity(row: dict[str, object]) -> str:
        survey_id = str(row.get("survey_id") or "")
        question_id = str(row.get("question_id") or "")
        if question_id:
            return f"{survey_id}:{question_id}" if survey_id else question_id
        poll_id = str(row.get("poll_id") or "")
        if poll_id:
            stem = re.sub(r"-(?:DEM|REP|IND|OTHER|D|R|I|O)$", "", poll_id, flags=re.IGNORECASE)
            return f"{survey_id}:{stem}" if survey_id else stem
        return survey_id

    @staticmethod
    def _effect_lookup(mapping: dict[str, float], value: str) -> float:
        if value in mapping:
            return mapping[value]
        return next((effect for key, effect in mapping.items() if key in value), 0.0)

    def _observation(
        self,
        row: dict[str, object],
        house_effects: dict[tuple[str, str | None], HouseEffectEstimate],
    ) -> PollObservation | None:
        observation = super()._observation(row, house_effects)
        if observation is None:
            return None
        floor = float(row.get("_likelihood_nonsampling_sd") or self.min_nonsampling_error)
        sampling = (
            observation.observed_share
            * (1.0 - observation.observed_share)
            / max(observation.effective_sample_size, 1.0)
        )
        return replace(observation, observation_variance=sampling + floor**2)

    @staticmethod
    def _binary_option_pairs(options: pl.DataFrame) -> dict[str, tuple[str, str]]:
        pairs: dict[str, tuple[str, str]] = {}
        for race_key, group in options.group_by("race_id", maintain_order=True):
            race_id = str(race_key[0] if isinstance(race_key, tuple) else race_key)
            rows = list(group.select(["option_id", "party"]).iter_rows(named=True))
            if len(rows) != 2:
                continue
            rows.sort(
                key=lambda row: (
                    0 if str(row.get("party") or "").upper() in {"DEM", "YES"} else 1,
                    str(row["option_id"]),
                )
            )
            pairs[race_id] = (str(rows[0]["option_id"]), str(rows[1]["option_id"]))
        return pairs

    def _paired_binary_observations(
        self,
        polls: pl.DataFrame,
        race_id: str,
        reference_option: str,
        other_option: str,
        house_effects: dict[tuple[str, str | None], HouseEffectEstimate],
    ) -> list[PollObservation]:
        observations: list[PollObservation] = []
        scoped = polls.filter(pl.col("race_id") == race_id)
        for _question, group in scoped.group_by("_likelihood_question_id", maintain_order=True):
            ref = group.filter(pl.col("option_id") == reference_option)
            other = group.filter(pl.col("option_id") == other_option)
            if ref.is_empty() or other.is_empty():
                continue
            row = ref.row(0, named=True)
            row["sample_size"] = row.get("_likelihood_original_sample_size")
            observation = self._observation(row, house_effects)
            if observation is not None:
                observations.append(observation)
        return observations

    @staticmethod
    def _mirror_trajectory_rows(
        rows: list[dict[str, object]], other_option: str
    ) -> list[dict[str, object]]:
        mirrored: list[dict[str, object]] = []
        for row in rows:
            copied = dict(row)
            copied["option_id"] = other_option
            for column in (
                "latent_vote_share",
                "initial_vote_share_prior",
                "marginal_win_probability",
                "mean_observed_share",
                "mean_adjusted_share",
            ):
                value = copied.get(column)
                if value is not None:
                    copied[column] = 1.0 - float(value)
            copied["mean_house_effect"] = -float(copied.get("mean_house_effect") or 0.0)
            mirrored.append(copied)
        return mirrored

    def _fit(
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        self._fallback_audit_override = None
        if self.backend == "nuts":
            try:
                return self._fit_nuts_backend(bundle, as_of)
            except (RuntimeError, TimeoutError, ValueError, ImportError) as exc:
                dispatched = dispatch_failover(
                    self.failover_policy,
                    primary_engine="numpyro-nuts",
                    reason=str(exc),
                    handlers={
                        self.failover_policy.ANALYTIC: lambda: self._fit_analytic(bundle, as_of),
                        self.failover_policy.KALMAN: lambda: self._fit_kalman_fallback(
                            bundle, as_of
                        ),
                    },
                    previous_posterior=self._load_previous_posterior_fallback(bundle, as_of),
                )
                fallback_label = str(dispatched.audit["fallback_used"])
                self._fallback_audit_override = {
                    "fallback_used": fallback_label,
                    "failover_audit": dispatched.audit,
                }
                self._cached_diagnostics.update(self._fallback_audit_override)
                return dispatched.result
        return self._fit_analytic(bundle, as_of)

    def _fit_kalman_fallback(
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        model = KalmanPollingModel(
            config=dict(self._config),
            as_of=as_of.isoformat() if as_of else None,
        )
        estimates = model.run(bundle)
        trajectory = model.trajectory(bundle)
        self._cached_posterior_draws = self._empty_posterior_draws()
        self._cached_diagnostics = {
            **self._empty_diagnostics(),
            "engine": "legacy-kalman-fallback",
            "draw_count": 0,
            "race_option_count": estimates.height,
            "poll_count": bundle.polls.height,
            "poll_likelihood": {},
            "previous_posterior_artifact": None,
        }
        return estimates, trajectory, model.cached_house_effects

    def _load_previous_posterior_fallback(
        self, bundle: FeatureBundle, as_of: date | None
    ) -> (
        LoadedPreviousPosterior[
            tuple[
                pl.DataFrame,
                pl.DataFrame,
                dict[tuple[str, str | None], HouseEffectEstimate],
            ]
        ]
        | None
    ):
        if self.failover_policy.PREVIOUS_POSTERIOR not in self.failover_policy.fallback_order:
            return None
        path = self._previous_posterior_config.get("path")
        if not path:
            return None
        validation_metadata: dict[str, Any] = {}

        def loader(artifact_path: Path):
            posterior = pl.read_parquet(artifact_path)
            normalized = self._validate_previous_posterior(
                posterior,
                bundle=bundle,
                as_of=as_of,
                metadata=validation_metadata,
            )
            self._cached_posterior_draws = normalized
            estimates = self._estimate_frame(
                self._estimate_rows_from_posterior(
                    normalized,
                    "Compatible previous Bayesian posterior reused after primary "
                    "inference failure.",
                )
            )
            self._cached_diagnostics = {
                **self._empty_diagnostics(),
                "engine": "previous-posterior-reuse",
                "draw_count": normalized["draw_id"].n_unique(),
                "race_option_count": estimates.height,
                "poll_count": 0,
                "poll_likelihood": {},
                "previous_posterior_artifact": dict(validation_metadata),
            }
            return estimates, self._empty_trajectory(), {}

        return load_previous_posterior_artifact(
            str(path),
            loader=loader,
            validator=lambda _result: (True, "compatible", validation_metadata),
        )

    def _validate_previous_posterior(
        self,
        posterior: pl.DataFrame,
        *,
        bundle: FeatureBundle,
        as_of: date | None,
        metadata: dict[str, Any],
    ) -> pl.DataFrame:
        if posterior.is_empty():
            raise ValueError("previous posterior artifact is empty")
        required = set(self.POSTERIOR_SCHEMA) | {"model_config_hash", "source_manifest_hash"}
        missing = sorted(required - set(posterior.columns))
        if missing:
            raise ValueError("previous posterior artifact missing columns: " + ", ".join(missing))
        duplicate_count = (
            posterior.height
            - posterior.unique(
                subset=["draw_id", "race_id", "option_id"], maintain_order=True
            ).height
        )
        if duplicate_count:
            raise ValueError(f"previous posterior artifact has {duplicate_count} duplicate keys")
        invalid_probability = posterior.filter(
            pl.col("latent_share").is_null()
            | ~pl.col("latent_share").is_finite()
            | (pl.col("latent_share") <= 0.0)
            | (pl.col("latent_share") >= 1.0)
        ).height
        if invalid_probability:
            raise ValueError(
                f"previous posterior artifact has {invalid_probability} invalid latent shares"
            )
        expected_hash = str(self._previous_posterior_config.get("model_config_hash") or "")
        hashes = posterior["model_config_hash"].drop_nulls().unique().cast(pl.String).to_list()
        if not expected_hash:
            raise PreviousPosteriorCompatibilityError(
                "previous_posterior.model_config_hash is required for compatibility"
            )
        if hashes != [expected_hash]:
            raise PreviousPosteriorCompatibilityError(
                f"model_config_hash mismatch: artifact={hashes}, expected={expected_hash}"
            )
        expected_source_hash = str(
            self._previous_posterior_config.get("source_manifest_hash") or ""
        )
        source_hashes = (
            posterior["source_manifest_hash"].drop_nulls().unique().cast(pl.String).to_list()
        )
        if not expected_source_hash:
            raise PreviousPosteriorCompatibilityError(
                "previous_posterior.source_manifest_hash is required for compatibility"
            )
        if source_hashes != [expected_source_hash]:
            raise PreviousPosteriorCompatibilityError(
                "source_manifest_hash mismatch: "
                f"artifact={source_hashes}, expected={expected_source_hash}"
            )
        artifact_as_of_raw = self._previous_posterior_config.get("as_of")
        if not artifact_as_of_raw or as_of is None:
            raise PreviousPosteriorCompatibilityError(
                "previous_posterior.as_of and current as_of are required for compatibility"
            )
        artifact_as_of = date.fromisoformat(str(artifact_as_of_raw))
        age_days = (as_of - artifact_as_of).days
        max_age_days = int(self._previous_posterior_config.get("max_age_days", 7))
        if age_days < 0 or age_days > max_age_days:
            raise PreviousPosteriorCompatibilityError(
                f"artifact age {age_days} days is outside [0, {max_age_days}]"
            )
        artifact_keys = set(posterior.select(["race_id", "option_id"]).unique().iter_rows())
        expected_keys = set(bundle.options.select(["race_id", "option_id"]).unique().iter_rows())
        if artifact_keys != expected_keys:
            missing_keys = len(expected_keys - artifact_keys)
            extra_keys = len(artifact_keys - expected_keys)
            raise PreviousPosteriorCompatibilityError(
                f"race-option lineage mismatch: missing={missing_keys}, extra={extra_keys}"
            )
        normalized = posterior.select(
            [pl.col(column).cast(dtype) for column, dtype in self.POSTERIOR_SCHEMA.items()]
        ).sort(["race_id", "option_id", "draw_id"])
        metadata.update(
            {
                "model_config_hash": expected_hash,
                "source_manifest_hash": expected_source_hash,
                "artifact_as_of": artifact_as_of.isoformat(),
                "current_as_of": as_of.isoformat(),
                "age_days": age_days,
                "max_age_days": max_age_days,
                "draw_count": normalized["draw_id"].n_unique(),
                "race_option_count": len(artifact_keys),
            }
        )
        return normalized

    def _fit_analytic(
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        self._cached_posterior_draws = self._empty_posterior_draws()
        self._cached_diagnostics = self._empty_diagnostics()
        if as_of is None:
            return normalize_rows([]), self._empty_trajectory(), {}

        polls = self._eligible_polls(bundle.polls, as_of)
        polls, likelihood_audit = self._prepare_likelihood_polls(
            bundle, polls, split_overlap_precision=True
        )
        if polls.is_empty():
            self._cached_diagnostics = {
                **self._empty_diagnostics(),
                "poll_likelihood": likelihood_audit,
            }
            return normalize_rows([]), self._empty_trajectory(), {}

        option_priors = self._option_priors(bundle.options)
        geography_by_race = self._geography_by_race(bundle.race_catalog)
        office_by_race = self._office_by_race(bundle.race_catalog)
        election_day_by_race = self._election_day_by_race(bundle.race_catalog)
        house_effects = self._estimate_house_effects(polls, option_priors)
        trajectory_rows: list[dict[str, object]] = []
        draw_rows: list[dict[str, object]] = []
        fitted_keys: set[tuple[str, str]] = set()

        seed = self._draw_seed(bundle, as_of)
        rng = np.random.default_rng(seed)
        posterior_sds: list[float] = []
        poll_counts: list[int] = []
        handled_binary_races: set[str] = set()
        for race_id, (reference_option, other_option) in self._binary_option_pairs(
            bundle.options
        ).items():
            observations = self._paired_binary_observations(
                polls,
                race_id,
                reference_option,
                other_option,
                house_effects,
            )
            if not observations:
                continue
            handled_binary_races.add(race_id)
            fitted_keys.update({(race_id, reference_option), (race_id, other_option)})
            ref_prior = option_priors.get((race_id, reference_option), 0.5)
            other_prior = option_priors.get((race_id, other_option), 0.5)
            prior = ref_prior / max(ref_prior + other_prior, 1e-9)
            prior_spec = self._fundamentals_prior.get((race_id, reference_option))
            prior_sd_logit = (
                float(prior_spec["sd_logit"])
                if prior_spec is not None
                else self.initial_state_logit_sd
            )
            mean_logit, sd_logit = self._posterior_logit(
                prior, observations, prior_sd_logit=prior_sd_logit
            )
            forecast_sd_logit = self._forecast_logit_sd(sd_logit, inv_logit(mean_logit))
            horizon_sd = self._forecast_horizon_logit_sd(race_id, as_of, election_day_by_race)
            forecast_sd_logit = math.sqrt(forecast_sd_logit**2 + horizon_sd**2)
            posterior_sds.extend([forecast_sd_logit, forecast_sd_logit])
            poll_counts.append(len(observations))
            latent_logits = rng.normal(mean_logit, forecast_sd_logit, self.posterior_draw_count)
            latent_shares = np.array([inv_logit(float(value)) for value in latent_logits])
            reference_trajectory = self._trajectory_rows_for_option(
                race_id=race_id,
                option_id=reference_option,
                observations=observations,
                as_of=as_of,
                initial_mean=prior,
                initial_sd_logit=prior_sd_logit,
            )
            trajectory_rows.extend(reference_trajectory)
            trajectory_rows.extend(self._mirror_trajectory_rows(reference_trajectory, other_option))
            geography = geography_by_race.get(race_id, "")
            mean_house_effect = self._mean_or_zero(
                [observation.house_effect for observation in observations]
            )
            for draw_id, (latent_logit, latent_share) in enumerate(
                zip(latent_logits, latent_shares, strict=True)
            ):
                for option_id, option_logit, option_share, effect in (
                    (reference_option, latent_logit, latent_share, mean_house_effect),
                    (other_option, -latent_logit, 1.0 - latent_share, -mean_house_effect),
                ):
                    draw_rows.append(
                        {
                            "draw_id": draw_id,
                            "chain_id": 0,
                            "race_id": race_id,
                            "option_id": option_id,
                            "geography": geography,
                            "trajectory_date": as_of,
                            "latent_logit": float(option_logit),
                            "latent_share": float(option_share),
                            "systematic_error": float(
                                option_logit
                                - (mean_logit if option_id == reference_option else -mean_logit)
                            ),
                            "pollster_effect": effect,
                            "diagnostic_only": False,
                        }
                    )
        likelihood_audit["analytic_binary_joint_race_count"] = len(handled_binary_races)
        likelihood_audit["analytic_binary_option_rows_independent"] = False
        sort_columns = [
            column
            for column in ["race_id", "option_id", "_poll_end_date", "pollster", "poll_id"]
            if column in polls.columns
        ]
        sorted_polls = polls.sort(sort_columns) if sort_columns else polls
        for key, group in sorted_polls.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = str(key[0]), str(key[1])
            if race_id in handled_binary_races:
                continue
            observations = [
                self._observation(row, house_effects) for row in group.iter_rows(named=True)
            ]
            observations = [observation for observation in observations if observation is not None]
            if not observations:
                continue
            fitted_keys.add((race_id, option_id))
            prior_spec = self._fundamentals_prior.get((race_id, option_id))
            if prior_spec is not None:
                prior = inv_logit(prior_spec["mean_logit"])
                prior_sd_logit = prior_spec["sd_logit"]
            else:
                prior = option_priors.get((race_id, option_id), 0.5)
                prior_sd_logit = self.initial_state_logit_sd
            mean_logit, sd_logit = self._posterior_logit(
                prior, observations, prior_sd_logit=prior_sd_logit
            )
            poll_counts.append(len(observations))
            forecast_sd_logit = self._forecast_logit_sd(sd_logit, inv_logit(mean_logit))
            horizon_sd = self._forecast_horizon_logit_sd(race_id, as_of, election_day_by_race)
            forecast_sd_logit = math.sqrt(forecast_sd_logit**2 + horizon_sd**2)
            posterior_sds.append(forecast_sd_logit)
            latent_logits = rng.normal(mean_logit, forecast_sd_logit, self.posterior_draw_count)
            latent_shares = np.array([inv_logit(float(value)) for value in latent_logits])
            trajectory_rows.extend(
                self._trajectory_rows_for_option(
                    race_id=race_id,
                    option_id=option_id,
                    observations=observations,
                    as_of=as_of,
                    initial_mean=prior,
                    initial_sd_logit=prior_sd_logit,
                )
            )
            geography = geography_by_race.get(race_id, "")
            mean_house_effect = self._mean_or_zero(
                [observation.house_effect for observation in observations]
            )
            draw_rows.extend(
                {
                    "draw_id": draw_id,
                    "chain_id": 0,
                    "race_id": race_id,
                    "option_id": option_id,
                    "geography": geography,
                    "trajectory_date": as_of,
                    "latent_logit": float(latent_logit),
                    "latent_share": float(latent_share),
                    "systematic_error": float(latent_logit - mean_logit),
                    "pollster_effect": mean_house_effect,
                    "diagnostic_only": False,
                }
                for draw_id, (latent_logit, latent_share) in enumerate(
                    zip(latent_logits, latent_shares, strict=True)
                )
            )

        unpolled_shifts, propagation_metadata = self._unpolled_hierarchical_shifts(
            bundle=bundle,
            fitted_keys=fitted_keys,
            draw_rows=draw_rows,
            office_by_race=office_by_race,
            geography_by_race=geography_by_race,
        )
        prior_only_count = self._append_prior_only_draws(
            bundle=bundle,
            rng=rng,
            as_of=as_of,
            fitted_keys=fitted_keys,
            office_by_race=office_by_race,
            geography_by_race=geography_by_race,
            draw_rows=draw_rows,
            posterior_sds=posterior_sds,
            election_day_by_race=election_day_by_race,
            hierarchical_shifts=unpolled_shifts,
        )
        self._cached_posterior_draws = self._posterior_frame(draw_rows, bundle.options)
        unsupported_races = set(likelihood_audit["unsupported_multi_option_races"])
        if unsupported_races:
            self._cached_posterior_draws = self._cached_posterior_draws.filter(
                ~pl.col("race_id").is_in(unsupported_races)
            )
        estimate_rows = self._estimate_rows_from_posterior(
            self._cached_posterior_draws,
            (
                "Bayesian logit-normal polling posterior with empirical-Bayes pollster "
                "house-effect initialization, race-constrained posterior draws, and "
                "election-day horizon inflation."
            ),
        )
        self._cached_diagnostics = {
            "engine": "bayes-analytic-logit-normal",
            "parameterization": self.parameterization,
            "draw_count": self.posterior_draw_count,
            "race_option_count": len(estimate_rows),
            "polling_observed_race_option_count": len(fitted_keys),
            "prior_only_race_option_count": prior_only_count,
            "unpolled_hierarchical_propagation": propagation_metadata,
            "poll_count": int(sum(poll_counts)),
            "poll_likelihood": likelihood_audit,
            "fundamentals_prior_rows": len(self._fundamentals_prior),
            "fundamentals_prior_used": bool(self._fundamentals_prior),
            "posterior_logit_sd_mean": float(np.mean(posterior_sds)) if posterior_sds else None,
            "forecast_horizon_inflation": self._forecast_horizon_metadata(
                election_day_by_race, as_of, fitted_keys
            ),
            "r_hat_max": None,
            "ess_min": None,
            "divergences": 0,
            "fallback_used": None,
            "failover_policy": self.failover_policy.to_dict(),
            "failover_audit": {
                "status": "not_exercised_analytic_bridge",
                "primary_engine": "bayes-analytic-logit-normal",
                "fallback_used": None,
                "publication_blocked": False,
            },
        }
        if self._fallback_audit_override:
            self._cached_diagnostics.update(self._fallback_audit_override)
        return (
            self._estimate_frame(estimate_rows),
            self._trajectory_frame(trajectory_rows),
            house_effects,
        )

    def _fit_nuts_backend(  # pragma: no cover - optional NumPyro/JAX backend
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        if as_of is None:
            return normalize_rows([]), self._empty_trajectory(), {}
        from civic_signal.inference.nuts import NutsConfig, fit_nuts
        from civic_signal.inference.state_space import build_state_space_data

        eligible_polls = self._eligible_polls(bundle.polls, as_of)
        eligible_polls, likelihood_audit = self._prepare_likelihood_polls(
            bundle, eligible_polls, split_overlap_precision=False
        )
        if eligible_polls.is_empty():
            self._cached_diagnostics = {
                **self._empty_diagnostics(),
                "poll_likelihood": likelihood_audit,
            }
            return normalize_rows([]), self._empty_trajectory(), {}
        house_effects = self._estimate_house_effects(
            eligible_polls,
            self._option_priors(bundle.options),
        )
        election_day_by_race = self._election_day_by_race(bundle.race_catalog)
        # House effects are estimated inside the model (pollster_effect); the
        # empirical-Bayes estimates are kept for reporting artifacts only, so
        # the same information is never subtracted twice from the observations.
        likelihood_bundle = replace(bundle, polls=eligible_polls)
        data = build_state_space_data(
            likelihood_bundle,
            as_of=as_of.isoformat(),
            office_type=None,
            prior_logit_by_key={
                key: float(value["mean_logit"]) for key, value in self._fundamentals_prior.items()
            },
            poll_half_life_days=float(
                dict(dict(self._config.get("bayesian", {})).get("state_space", {})).get(
                    "poll_half_life_days", self.half_life_days
                )
            ),
            process_drift_sd_per_sqrt_day=self.forecast_drift_sd_per_sqrt_day,
            pollster_house_effects={},
            election_day_by_race=election_day_by_race,
            pollster_quality_weights={
                str(key): float(value)
                for key, value in dict(
                    dict(self._config.get("polling", {})).get("pollster_quality_weights", {})
                ).items()
            },
        )
        if data.poll_logit_y.size == 0:
            return normalize_rows([]), self._empty_trajectory(), {}
        nuts_config = dict(dict(self._config.get("bayesian", {})).get("nuts", {}))
        cfg = NutsConfig(
            num_warmup=int(nuts_config.get("num_warmup", 200)),
            num_samples=int(nuts_config.get("num_samples", self.posterior_draw_count)),
            num_chains=int(nuts_config.get("num_chains", 1)),
            chain_method=str(nuts_config.get("chain_method", "vectorized")),
            target_accept_prob=float(nuts_config.get("target_accept_prob", 0.99)),
            parameterization=self.parameterization,
            wall_clock_timeout_seconds=(
                float(nuts_config["wall_clock_timeout_seconds"])
                if nuts_config.get("wall_clock_timeout_seconds") is not None
                else None
            ),
        )
        result = fit_nuts(
            data,
            hyperpriors=self._nuts_hyperpriors(),
            config=cfg,
            seed=self._draw_seed(bundle, as_of),
        )
        state_logit = np.asarray(result.samples["state_logit"], dtype=np.float64)
        if state_logit.ndim == 1:
            state_logit = state_logit.reshape(1, -1)
        pollster_effect = np.asarray(
            result.samples.get("pollster_effect", np.zeros((state_logit.shape[0], 1))),
            dtype=np.float64,
        )
        if pollster_effect.ndim == 1:
            pollster_effect = pollster_effect.reshape(-1, 1)
        sample_count = int(state_logit.shape[0])
        if sample_count <= 0:
            raise ValueError("NUTS returned no posterior state samples")

        geography_by_race = self._geography_by_race(bundle.race_catalog)
        office_by_race = self._office_by_race(bundle.race_catalog)
        # Races on the reverse random walk already carry poll-to-election-day
        # drift inside the posterior; adding the analytic horizon term would
        # double-count that uncertainty.
        walk_covered_races = set(data.margin_race_ids) if data.margin_poll_y.size else set()
        fitted_keys = set(data.race_option_keys)
        poll_counts = np.bincount(data.poll_s, minlength=len(data.race_option_keys))
        draw_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []
        posterior_sds: list[float] = []
        draw_rng = np.random.default_rng(self._draw_seed(bundle, as_of) + 2)
        selected_indices = self._selected_posterior_indices(sample_count, draw_rng)
        for option_index, (race_id, option_id) in enumerate(data.race_option_keys):
            logits = state_logit[:, option_index]
            shares = np.array([inv_logit(float(value)) for value in logits])
            mean_logit = float(logits.mean())
            vote_share = float(shares.mean())
            current_sd = float(logits.std())
            forecast_sd_logit = self._forecast_logit_sd(current_sd, vote_share)
            if race_id not in walk_covered_races:
                horizon_sd = self._forecast_horizon_logit_sd(race_id, as_of, election_day_by_race)
                forecast_sd_logit = math.sqrt(forecast_sd_logit**2 + horizon_sd**2)
            posterior_sds.append(forecast_sd_logit)
            marginal_win_probability = float(normal_cdf(mean_logit / max(forecast_sd_logit, 1e-8)))
            draw_logits = np.asarray(logits[selected_indices], dtype=np.float64)
            extra_variance = max(0.0, forecast_sd_logit**2 - current_sd**2)
            if extra_variance > 0:
                draw_logits += draw_rng.normal(
                    0, math.sqrt(extra_variance), size=self.posterior_draw_count
                )
            draw_shares = np.array([inv_logit(float(value)) for value in draw_logits])
            uncertainty = max(
                float(draw_shares.std()),
                self.min_nonsampling_error,
            )
            trajectory_rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "trajectory_date": as_of,
                    "as_of": as_of,
                    "latent_vote_share": vote_share,
                    "latent_variance": uncertainty**2,
                    "latent_sigma": uncertainty,
                    "initial_vote_share_prior": inv_logit(float(data.prior_logit[option_index])),
                    "marginal_win_probability": marginal_win_probability,
                    "poll_count": int(poll_counts[option_index]),
                    "effective_sample_size": float(result.diagnostics.get("ess_min") or 0.0),
                    "mean_observed_share": None,
                    "mean_adjusted_share": None,
                    "mean_observation_variance": None,
                    "mean_house_effect": 0.0,
                    "process_variance": 0.0,
                    "nonsampling_variance": uncertainty**2,
                    "admitted": True,
                    "explanation": "NumPyro NUTS posterior summary at forecast as-of date.",
                }
            )
            geography = geography_by_race.get(race_id, "")
            for draw_id, (latent_logit, latent_share) in enumerate(
                zip(draw_logits, draw_shares, strict=True)
            ):
                draw_rows.append(
                    {
                        "draw_id": draw_id,
                        "chain_id": 0,
                        "race_id": race_id,
                        "option_id": option_id,
                        "geography": geography,
                        "trajectory_date": as_of,
                        "latent_logit": float(latent_logit),
                        "latent_share": float(latent_share),
                        "systematic_error": float(latent_logit - mean_logit),
                        "pollster_effect": float(np.mean(pollster_effect[:, 0])),
                        "diagnostic_only": False,
                    }
                )

        rng = np.random.default_rng(self._draw_seed(bundle, as_of) + 1)
        unpolled_shifts, propagation_metadata = self._unpolled_hierarchical_shifts(
            bundle=bundle,
            fitted_keys=fitted_keys,
            draw_rows=draw_rows,
            office_by_race=office_by_race,
            geography_by_race=geography_by_race,
        )
        prior_only_count = self._append_prior_only_draws(
            bundle=bundle,
            rng=rng,
            as_of=as_of,
            fitted_keys=fitted_keys,
            office_by_race=office_by_race,
            geography_by_race=geography_by_race,
            draw_rows=draw_rows,
            posterior_sds=posterior_sds,
            election_day_by_race=election_day_by_race,
            hierarchical_shifts=unpolled_shifts,
        )
        self._cached_posterior_draws = self._posterior_frame(draw_rows, bundle.options)
        unsupported_races = set(likelihood_audit["unsupported_multi_option_races"])
        if unsupported_races:
            self._cached_posterior_draws = self._cached_posterior_draws.filter(
                ~pl.col("race_id").is_in(unsupported_races)
            )
        estimate_rows = self._estimate_rows_from_posterior(
            self._cached_posterior_draws,
            (
                "Joint Bayesian polling posterior fitted with NumPyro NUTS, converted "
                "to race-constrained election-day posterior draws."
            ),
        )
        independent_poll_count = int(data.margin_poll_y.size + data.legacy_poll_indices.size)
        likelihood_audit.update(
            {
                "nuts_binary_margin_likelihood_count": int(data.margin_poll_y.size),
                "nuts_legacy_likelihood_count": int(data.legacy_poll_indices.size),
                "nuts_binary_option_rows_independent": False,
            }
        )
        self._cached_diagnostics = {
            **result.diagnostics,
            "engine": "numpyro-nuts",
            "parameterization": self.parameterization,
            "temporal_model": data.metadata.get("temporal_model"),
            "margin_race_count": data.metadata.get("margin_race_count", 0),
            "margin_poll_count": data.metadata.get("margin_poll_count", 0),
            "draw_count": self.posterior_draw_count,
            "nuts_sample_count": sample_count,
            "race_option_count": len(data.race_option_keys) + prior_only_count,
            "polling_observed_race_option_count": len(data.race_option_keys),
            "prior_only_race_option_count": prior_only_count,
            "unpolled_hierarchical_propagation": propagation_metadata,
            "poll_count": independent_poll_count,
            "poll_likelihood": likelihood_audit,
            "fundamentals_prior_rows": len(self._fundamentals_prior),
            "fundamentals_prior_used": bool(self._fundamentals_prior),
            "posterior_logit_sd_mean": float(np.mean(posterior_sds)) if posterior_sds else None,
            "posterior_sample_resampling": "with_replacement"
            if sample_count < self.posterior_draw_count
            else "without_replacement",
            "forecast_horizon_inflation": self._forecast_horizon_metadata(
                election_day_by_race, as_of, fitted_keys
            ),
            "hierarchical_effects": {
                "office_count": len(data.office_ids),
                "geography_count": len(data.geography_ids),
                "race_count": len(data.race_ids),
                "office_ids": list(data.office_ids),
                "geography_ids": list(data.geography_ids),
            },
            "fallback_used": None,
            "failover_policy": self.failover_policy.to_dict(),
        }
        return (
            self._estimate_frame(estimate_rows),
            self._trajectory_frame(trajectory_rows),
            house_effects,
        )

    def _nuts_hyperpriors(self):
        from civic_signal.inference.state_space import HyperPriors

        bayesian = dict(self._config.get("bayesian", {}))
        state_space = dict(bayesian.get("state_space", {}))
        # Weekly walk-innovation prior scale: backtest-learned drift when a
        # promoted estimate exists, else the configured per-sqrt-day drift.
        drift_per_sqrt_day = float(
            self._config.get(
                "_learned_horizon_drift_sd_per_sqrt_day", self.forecast_drift_sd_per_sqrt_day
            )
            or self.forecast_drift_sd_per_sqrt_day
        )
        return HyperPriors(
            sigma_state=self.initial_state_logit_sd,
            tau_pollster=float(
                dict(bayesian.get("observation", {})).get("pollster_effect_sd", 0.04)
            ),
            sigma_office=float(
                dict(bayesian.get("cross_office", {})).get("office_offset_prior_sd", 0.02)
            ),
            sigma_geography=float(state_space.get("geography_effect_sd", 0.06)),
            sigma_race=float(state_space.get("race_effect_sd", 0.08)),
            sigma_walk=max(drift_per_sqrt_day * math.sqrt(7.0), 1e-4),
        )

    def _append_prior_only_draws(
        self,
        *,
        bundle: FeatureBundle,
        rng: np.random.Generator,
        as_of: date,
        fitted_keys: set[tuple[str, str]],
        office_by_race: dict[str, str],
        geography_by_race: dict[str, str],
        draw_rows: list[dict[str, object]],
        posterior_sds: list[float],
        election_day_by_race: dict[str, date],
        hierarchical_shifts: dict[tuple[str, str], float] | None = None,
    ) -> int:
        candidate_offices = {"president", "senate", "house", "governor"}
        prior_only_count = 0
        for row in bundle.options.sort(["race_id", "option_id"]).iter_rows(named=True):
            race_id = str(row["race_id"])
            option_id = str(row["option_id"])
            if (race_id, option_id) in fitted_keys:
                continue
            if office_by_race.get(race_id) not in candidate_offices:
                continue
            prior_spec = self._fundamentals_prior.get((race_id, option_id))
            if prior_spec is None:
                continue
            mean_logit = float(prior_spec["mean_logit"]) + float(
                (hierarchical_shifts or {}).get((race_id, option_id), 0.0)
            )
            sd_logit = float(prior_spec["sd_logit"])
            horizon_sd = self._forecast_horizon_logit_sd(race_id, as_of, election_day_by_race)
            forecast_sd_logit = math.sqrt(sd_logit**2 + horizon_sd**2)
            posterior_sds.append(forecast_sd_logit)
            latent_logits = rng.normal(mean_logit, forecast_sd_logit, self.posterior_draw_count)
            latent_shares = np.array([inv_logit(float(value)) for value in latent_logits])
            geography = geography_by_race.get(race_id, "")
            draw_rows.extend(
                {
                    "draw_id": draw_id,
                    "chain_id": 0,
                    "race_id": race_id,
                    "option_id": option_id,
                    "geography": geography,
                    "trajectory_date": as_of,
                    "latent_logit": float(latent_logit),
                    "latent_share": float(latent_share),
                    "systematic_error": float(latent_logit - mean_logit),
                    "pollster_effect": 0.0,
                    "diagnostic_only": True,
                }
                for draw_id, (latent_logit, latent_share) in enumerate(
                    zip(latent_logits, latent_shares, strict=True)
                )
            )
            prior_only_count += 1
        return prior_only_count

    def _unpolled_hierarchical_shifts(
        self,
        *,
        bundle: FeatureBundle,
        fitted_keys: set[tuple[str, str]],
        draw_rows: list[dict[str, object]],
        office_by_race: dict[str, str],
        geography_by_race: dict[str, str],
    ) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
        """Partially pool observed signed swings into fundamentals-only races.

        The pooling unit is one party-signed residual per race, which prevents
        complementary D/R options from cancelling or counting twice. Global,
        office, and geography means are shrunk toward zero before being combined.
        """
        parties = {
            (str(row["race_id"]), str(row["option_id"])): str(row.get("party") or "")
            for row in bundle.options.select(
                [
                    column
                    for column in ("race_id", "option_id", "party")
                    if column in bundle.options.columns
                ]
            ).iter_rows(named=True)
        }
        fitted = pl.DataFrame(draw_rows)
        if fitted.is_empty():
            return {}, {
                "status": "insufficient_no_polled_reference",
                "method": "signed_global_office_geography_partial_pooling",
                "observed_race_count": 0,
                "propagated_race_option_count": 0,
            }
        means = {
            (str(row["race_id"]), str(row["option_id"])): float(row["posterior_mean_logit"])
            for row in fitted.group_by(["race_id", "option_id"])
            .agg(pl.col("latent_logit").mean().alias("posterior_mean_logit"))
            .iter_rows(named=True)
        }
        signed_by_race: dict[str, list[float]] = {}
        for key in fitted_keys:
            prior = self._fundamentals_prior.get(key)
            posterior_mean = means.get(key)
            sign = self._party_sign(parties.get(key))
            if prior is None or posterior_mean is None or sign == 0.0:
                continue
            signed_by_race.setdefault(key[0], []).append(
                sign * (posterior_mean - float(prior["mean_logit"]))
            )
        race_swings = {
            race_id: float(np.mean(values)) for race_id, values in signed_by_race.items() if values
        }
        if not race_swings:
            return {}, {
                "status": "insufficient_no_signed_reference",
                "method": "signed_global_office_geography_partial_pooling",
                "observed_race_count": 0,
                "propagated_race_option_count": 0,
            }

        global_values = list(race_swings.values())
        office_values: dict[str, list[float]] = {}
        geography_values: dict[str, list[float]] = {}
        for race_id, swing in race_swings.items():
            office_values.setdefault(office_by_race.get(race_id, ""), []).append(swing)
            geography_values.setdefault(geography_by_race.get(race_id, ""), []).append(swing)
        prior_races = float(
            dict(dict(self._config.get("bayesian", {})).get("state_space", {})).get(
                "unpolled_pooling_prior_races", 3.0
            )
        )

        def pooled(values: list[float] | None) -> tuple[float, float]:
            if not values:
                return 0.0, 0.0
            return float(sum(values)), len(values) + max(prior_races, 0.0)

        shifts: dict[tuple[str, str], float] = {}
        for row in bundle.options.iter_rows(named=True):
            key = (str(row["race_id"]), str(row["option_id"]))
            if key in fitted_keys or key not in self._fundamentals_prior:
                continue
            sign = self._party_sign(row.get("party"))
            if sign == 0.0:
                continue
            components = [
                pooled(global_values),
                pooled(office_values.get(office_by_race.get(key[0], ""))),
                pooled(geography_values.get(geography_by_race.get(key[0], ""))),
            ]
            total_denominator = sum(denominator for _value, denominator in components)
            if total_denominator <= 0.0:
                continue
            signed_shift = sum(value for value, _denominator in components) / total_denominator
            shifts[key] = sign * min(0.75, max(-0.75, signed_shift))
        absolute = [abs(value) for value in shifts.values()]
        return shifts, {
            "status": "applied" if shifts else "no_eligible_unpolled_options",
            "method": "signed_global_office_geography_partial_pooling",
            "one_signed_residual_per_race": True,
            "observed_race_count": len(race_swings),
            "propagated_race_option_count": len(shifts),
            "office_pool_count": len(office_values),
            "geography_pool_count": len(geography_values),
            "prior_races": prior_races,
            "mean_absolute_logit_shift": float(np.mean(absolute)) if absolute else 0.0,
            "max_absolute_logit_shift": max(absolute, default=0.0),
        }

    @staticmethod
    def _party_sign(party: object) -> float:
        value = str(party or "").upper()
        if value in {"DEM", "YES"}:
            return 1.0
        if value in {"REP", "NO"}:
            return -1.0
        return 0.0

    def _estimate_rows_from_posterior(
        self,
        posterior: pl.DataFrame,
        explanation: str,
    ) -> list[dict[str, object]]:
        if posterior.is_empty():
            return []
        frame = posterior.with_columns(
            (
                pl.col("latent_share") == pl.col("latent_share").max().over(["draw_id", "race_id"])
            ).alias("_winner")
        )
        estimates = frame.group_by(["race_id", "option_id"]).agg(
            pl.col("_winner").mean().alias("marginal_win_probability"),
            pl.col("latent_share").mean().alias("vote_share"),
            pl.col("latent_share").std().alias("uncertainty"),
            pl.col("diagnostic_only").all().alias("prior_only"),
        )
        return [
            {
                "race_id": row["race_id"],
                "option_id": row["option_id"],
                "component": self.component,
                "marginal_win_probability": float(row["marginal_win_probability"]),
                "vote_share": float(row["vote_share"]),
                "uncertainty": max(float(row["uncertainty"] or 0.0), self.min_nonsampling_error),
                "admitted": True,
                "explanation": (
                    "Fundamentals-prior-only Bayesian election-day posterior for sparse race."
                    if bool(row["prior_only"])
                    else explanation
                ),
            }
            for row in estimates.iter_rows(named=True)
        ]

    def _estimate_frame(self, rows: list[dict[str, object]]) -> pl.DataFrame:
        frame = normalize_rows(rows)
        if frame.is_empty() or "marginal_win_probability" not in frame.columns:
            return frame
        return (
            frame.with_columns(
                pl.len().over("race_id").alias("_race_option_count"),
                pl.col("marginal_win_probability")
                .sum()
                .over("race_id")
                .alias("_race_probability_sum"),
            )
            .with_columns(
                pl.when(
                    (pl.col("_race_option_count") > 1) & (pl.col("_race_probability_sum") > 0.0)
                )
                .then(pl.col("marginal_win_probability") / pl.col("_race_probability_sum"))
                .otherwise(pl.col("marginal_win_probability"))
                .alias("marginal_win_probability")
            )
            .drop(["_race_option_count", "_race_probability_sum"])
        )

    def _posterior_logit(
        self,
        prior_share: float,
        observations: list[PollObservation],
        prior_sd_logit: float | None = None,
    ) -> tuple[float, float]:
        prior_mean = logit(prior_share)
        prior_sd = self.initial_state_logit_sd if prior_sd_logit is None else prior_sd_logit
        prior_variance = max(prior_sd**2, 1e-8)
        precision = 1.0 / prior_variance
        weighted = prior_mean * precision
        for observation in observations:
            share = min(0.999999, max(0.000001, observation.adjusted_share))
            obs_logit = logit(share)
            obs_sd_share = math.sqrt(max(observation.observation_variance, 1e-10))
            obs_sd_logit = max(
                obs_sd_share / max(share * (1.0 - share), 1e-6),
                self.nonsampling_logit_floor,
            )
            obs_precision = 1.0 / max(obs_sd_logit**2, 1e-10)
            precision += obs_precision
            weighted += obs_logit * obs_precision
        posterior_mean = weighted / precision
        posterior_sd = math.sqrt(1.0 / precision)
        return posterior_mean, posterior_sd

    def _forecast_logit_sd(self, posterior_sd_logit: float, mean_share: float) -> float:
        share = min(0.999999, max(0.000001, mean_share))
        floor_logit_sd = self.min_nonsampling_error / max(share * (1.0 - share), 1e-6)
        return max(
            math.sqrt(max(posterior_sd_logit, 0.0) ** 2 + self.election_day_extra_sd**2),
            floor_logit_sd,
            self.nonsampling_logit_floor,
        )

    @staticmethod
    def _share_sd_from_logit_sd(mean_share: float, logit_sd: float) -> float:
        share = min(0.999999, max(0.000001, mean_share))
        return share * (1.0 - share) * max(logit_sd, 0.0)

    def _forecast_win_probability(self, mean_logit: float, sd_logit: float) -> float:
        mean_share = inv_logit(mean_logit)
        forecast_sd_logit = self._forecast_logit_sd(sd_logit, mean_share)
        return float(normal_cdf(mean_logit / max(forecast_sd_logit, 1e-8)))

    def _trajectory_rows_for_option(
        self,
        race_id: str,
        option_id: str,
        observations: list[PollObservation],
        as_of: date,
        initial_mean: float,
        initial_sd_logit: float,
    ) -> list[dict[str, object]]:
        observations_by_date: dict[date, list[PollObservation]] = {}
        for observation in sorted(observations, key=lambda item: (item.end_date, item.poll_id)):
            observations_by_date.setdefault(observation.end_date, []).append(observation)
        if not observations_by_date:
            return []
        rows: list[dict[str, object]] = []
        start_date = min(observations_by_date)
        trajectory_date = start_date
        observed_so_far: list[PollObservation] = []
        while trajectory_date <= as_of:
            observed_so_far.extend(observations_by_date.get(trajectory_date, []))
            mean_logit, sd_logit = self._posterior_logit(
                initial_mean, observed_so_far, prior_sd_logit=initial_sd_logit
            )
            share = inv_logit(mean_logit)
            share_sd = max(share * (1.0 - share) * sd_logit, self.min_nonsampling_error)
            todays_observations = observations_by_date.get(trajectory_date, [])
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "trajectory_date": trajectory_date,
                    "as_of": as_of,
                    "latent_vote_share": share,
                    "latent_variance": share_sd**2,
                    "latent_sigma": share_sd,
                    "initial_vote_share_prior": initial_mean,
                    "marginal_win_probability": self._forecast_win_probability(
                        mean_logit, sd_logit
                    ),
                    "poll_count": len(todays_observations),
                    "effective_sample_size": self._mean_or_zero(
                        [observation.effective_sample_size for observation in todays_observations]
                    ),
                    "mean_observed_share": self._mean_or_none(
                        [observation.observed_share for observation in todays_observations]
                    ),
                    "mean_adjusted_share": self._mean_or_none(
                        [observation.adjusted_share for observation in todays_observations]
                    ),
                    "mean_observation_variance": self._mean_or_none(
                        [observation.observation_variance for observation in todays_observations]
                    ),
                    "mean_house_effect": self._mean_or_zero(
                        [observation.house_effect for observation in todays_observations]
                    ),
                    "process_variance": 0.0,
                    "nonsampling_variance": self.min_nonsampling_error**2,
                    "admitted": True,
                    "explanation": (
                        "Bayesian logit-normal posterior trajectory after same-day updates."
                    ),
                }
            )
            trajectory_date += timedelta(days=1)
        return rows

    def _draw_seed(self, bundle: FeatureBundle, as_of: date) -> int:
        payload = f"{self._bundle_fingerprint(bundle)}:{as_of}:{self.posterior_draw_count}:bayes"
        return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2**32)

    def _selected_posterior_indices(
        self, sample_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        if sample_count >= self.posterior_draw_count:
            return rng.choice(sample_count, size=self.posterior_draw_count, replace=False).astype(
                np.int64
            )
        return rng.choice(sample_count, size=self.posterior_draw_count, replace=True).astype(
            np.int64
        )

    @staticmethod
    def _geography_by_race(race_catalog: pl.DataFrame) -> dict[str, str]:
        if race_catalog.is_empty() or not {"race_id", "geography"}.issubset(race_catalog.columns):
            return {}
        return {
            str(row["race_id"]): str(row.get("geography") or "")
            for row in race_catalog.select(["race_id", "geography"]).iter_rows(named=True)
        }

    @staticmethod
    def _office_by_race(race_catalog: pl.DataFrame) -> dict[str, str]:
        if race_catalog.is_empty() or not {"race_id", "office_type"}.issubset(race_catalog.columns):
            return {}
        return {
            str(row["race_id"]): str(row.get("office_type") or "")
            for row in race_catalog.select(["race_id", "office_type"]).iter_rows(named=True)
        }

    @classmethod
    def _posterior_frame(cls, rows: list[dict[str, object]], options: pl.DataFrame) -> pl.DataFrame:
        if not rows:
            return cls._empty_posterior_draws()
        frame = pl.DataFrame(rows, schema=cls.POSTERIOR_SCHEMA)
        if options.is_empty() or not {"race_id", "option_id"}.issubset(options.columns):
            return frame.sort(["race_id", "option_id", "draw_id"])
        option_counts = options.group_by("race_id").agg(
            pl.col("option_id").n_unique().alias("_option_count")
        )
        return (
            frame.join(option_counts, on="race_id", how="left")
            .with_columns(
                pl.col("latent_share").sum().over(["draw_id", "race_id"]).alias("_sum_share")
            )
            .with_columns(
                pl.when(pl.col("_option_count").fill_null(1) > 1)
                .then((pl.col("latent_share") / pl.col("_sum_share")).clip(1e-6, 1.0 - 1e-6))
                .otherwise(pl.col("latent_share"))
                .alias("latent_share")
            )
            .with_columns(
                (pl.col("latent_share") / (1.0 - pl.col("latent_share")))
                .log()
                .alias("latent_logit")
            )
            .with_columns(
                (
                    pl.col("latent_logit")
                    - pl.col("latent_logit").mean().over(["race_id", "option_id"])
                ).alias("systematic_error")
            )
            .drop(["_option_count", "_sum_share"])
            .select(list(cls.POSTERIOR_SCHEMA))
            .sort(["race_id", "option_id", "draw_id"])
        )

    @staticmethod
    def _election_day_by_race(race_catalog: pl.DataFrame) -> dict[str, date]:
        if race_catalog.is_empty() or not {"race_id", "election_date"}.issubset(
            race_catalog.columns
        ):
            return {}
        values = {}
        for row in race_catalog.select(["race_id", "election_date"]).iter_rows(named=True):
            election_day = row.get("election_date")
            if election_day is None:
                continue
            if not hasattr(election_day, "toordinal"):
                election_day = date.fromisoformat(str(election_day))
            values[str(row["race_id"])] = election_day
        return values

    def _forecast_horizon_logit_sd(
        self, race_id: str, as_of: date, election_day_by_race: dict[str, date]
    ) -> float:
        election_day = election_day_by_race.get(race_id)
        if election_day is None:
            return 0.0
        horizon_days = max((election_day - as_of).days, 0)
        return float(self.forecast_drift_sd_per_sqrt_day * math.sqrt(horizon_days))

    def _forecast_horizon_metadata(
        self,
        election_day_by_race: dict[str, date],
        as_of: date,
        fitted_keys: set[tuple[str, str]],
    ) -> dict[str, object]:
        race_ids = sorted({race_id for race_id, _option_id in fitted_keys})
        days = [
            max((election_day_by_race[race_id] - as_of).days, 0)
            for race_id in race_ids
            if race_id in election_day_by_race
        ]
        sds = [self.forecast_drift_sd_per_sqrt_day * math.sqrt(value) for value in days]
        return {
            "method": "random_walk_logit_inflation",
            "drift_sd_per_sqrt_day": self.forecast_drift_sd_per_sqrt_day,
            "race_count": len(race_ids),
            "max_horizon_days": max(days) if days else 0,
            "mean_horizon_days": float(np.mean(days)) if days else 0.0,
            "mean_horizon_sd_logit": float(np.mean(sds)) if sds else 0.0,
        }

    @classmethod
    def _empty_posterior_draws(cls) -> pl.DataFrame:
        return pl.DataFrame(schema=cls.POSTERIOR_SCHEMA)

    @staticmethod
    def _fundamentals_prior_lookup(rows: object) -> dict[tuple[str, str], dict[str, float]]:
        if not isinstance(rows, list):
            return {}
        lookup: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            race_id = row.get("race_id")
            option_id = row.get("option_id")
            mean_logit = row.get("mean_logit")
            sd_logit = row.get("sd_logit")
            if race_id is None or option_id is None or mean_logit is None or sd_logit is None:
                continue
            lookup[(str(race_id), str(option_id))] = {
                "mean_logit": float(mean_logit),
                "sd_logit": float(sd_logit),
            }
        return lookup

    def _empty_diagnostics(self) -> dict[str, Any]:
        return {
            "engine": "bayes-analytic-logit-normal",
            "draw_count": 0,
            "race_option_count": 0,
            "poll_count": 0,
            "fundamentals_prior_rows": 0,
            "fundamentals_prior_used": False,
            "r_hat_max": None,
            "ess_min": None,
            "divergences": 0,
            "fallback_used": None,
            "failover_policy": self.failover_policy.to_dict(),
            "failover_audit": {
                "status": "not_exercised_analytic_bridge",
                "primary_engine": "bayes-analytic-logit-normal",
                "fallback_used": None,
                "publication_blocked": False,
            },
        }
