# Modeling Contract

Forecastable rows must include source lineage and model-config lineage. Tier C races
must be tracked in `race_catalog.parquet` but must not receive trusted winner
probabilities in `race_forecasts.parquet`.

Component estimates use the columns:

- `race_id`
- `option_id`
- `component`
- `marginal_win_probability`
- `vote_share`
- `uncertainty`
- `admitted`
- `explanation`

Final forecast artifacts are generated from correlated simulation draws, not from
point estimates alone.

`race_catalog.seats` is interpreted relative to `control_body`: House seats for House
races, Senate seats for Senate races, and Electoral College votes for presidential
state races. Presidential control thresholds therefore compare against summed electoral
votes, not a count of state contests.
