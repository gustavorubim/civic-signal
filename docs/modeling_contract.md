# Modeling Contract

Forecastable rows must include source lineage and model-config lineage. Tier C races
must be tracked in `race_catalog.parquet` but must not receive trusted winner
probabilities in `race_forecasts.parquet`.

Component estimates use the columns:

- `race_id`
- `option_id`
- `component`
- `win_probability`
- `vote_share`
- `uncertainty`
- `admitted`
- `explanation`

Final forecast artifacts are generated from correlated simulation draws, not from
point estimates alone.

