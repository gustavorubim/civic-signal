from __future__ import annotations

import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.models.common import clamp, logistic, normalize_rows


class FundamentalsModel:
    component = "fundamentals"

    def run(self, bundle: FeatureBundle) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        fundamentals = {row["race_id"]: row for row in bundle.fundamentals.iter_rows(named=True)}
        for race_id, group in bundle.options.group_by("race_id", maintain_order=True):
            race_key = race_id[0] if isinstance(race_id, tuple) else race_id
            fundamental = fundamentals.get(str(race_key))
            if fundamental is None:
                continue
            shares = self._raw_shares(group, fundamental)
            total = sum(shares.values()) or 1.0
            for option_id, share in shares.items():
                normalized_share = clamp(share / total)
                probability = logistic((normalized_share - 0.5) / 0.055)
                rows.append(
                    {
                        "race_id": str(race_key),
                        "option_id": option_id,
                        "component": self.component,
                        "win_probability": probability,
                        "vote_share": normalized_share,
                        "uncertainty": 0.08,
                        "admitted": True,
                        "explanation": (
                            "Historical lean, incumbency, finance, and turnout fundamentals."
                        ),
                    }
                )
        return normalize_rows(rows)

    def _raw_shares(
        self, options: pl.DataFrame, fundamental: dict[str, object]
    ) -> dict[str, float]:
        lean = float(fundamental.get("partisan_lean") or 0.0) / 100.0
        economy = float(fundamental.get("economic_index") or 0.0) / 50.0
        demographic = float(fundamental.get("demographic_turnout_index") or 0.0) / 80.0
        shares: dict[str, float] = {}
        for row in options.iter_rows(named=True):
            base = float(row.get("previous_vote_share") or 0.5)
            party = str(row.get("party") or "")
            incumbent = 0.01 if bool(row.get("incumbent")) else 0.0
            finance = min(float(row.get("fundraising_usd") or 0.0) / 1_000_000_000, 0.03)
            party_shift = lean + economy + demographic if party in {"DEM", "YES"} else -lean
            shares[str(row["option_id"])] = clamp(
                base + party_shift + incumbent + finance, 0.05, 0.95
            )
        return shares
