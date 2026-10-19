# Data Sources

The default source registry lives in `configs/sources.yaml`. A first live registry lives
in `configs/sources_live.yaml`. Each source records:

- stable source id
- logical curated table
- retrieval type
- URL or local path
- parser version
- license or terms note

The fixture registry is intentionally shaped like public sources so live adapters can
write the same raw manifest and curated tables. The default fixture registry includes a
compact presidential state-cycle panel for 2000-2024. It is derived from public
presidential returns in a MIT Election Data and Science Lab-style schema, using a public
Hugging Face mirror of `1976-2024-president-extended.csv` for deterministic offline
testing. The production adapter should prefer official MIT Election Lab / Harvard
Dataverse or FEC state-result releases when direct access is available. The 2024 fixture
rows include full 50-state-plus-DC Electoral College weights.

Current source modes:

- `fixture`: copy a local CSV into the raw lake and hash it.
- `http_csv`: download a public CSV endpoint into the raw lake and hash it.
- `http_text`: download a public text endpoint into the raw lake and hash it.
- `http_json`: download a public JSON endpoint into the raw lake and hash it.

HTTP sources use bounded retries. When a refresh fails but the same source id, URL, and
parser version already have a raw snapshot, sync records `status = stale_reused` and
continues from the cached content hash. This keeps live verification from blocking on a
transient upstream stall while preserving the manifest evidence that the current run
used a stale raw snapshot.

The live polling adapters in `configs/sources_live.yaml` download public
FiveThirtyEight/Datasette CSV streams for presidential verification and other legacy
feeds. `fivethirtyeight_president_polls` normalizes Wisconsin 2020
Democratic/Republican rows into the existing `polls` contract for `US-PRES-WI-2020`.
The archived Datasette mirror currently returns zero rows for `cycle=2026`, so it is
not a viable 2026 midterm poll source.

For 2026 Senate and House live forecasts, `configs/sources_senate_live.yaml` and
`configs/sources_house_live.yaml` use VoteHub's keyless JSON API
(`https://api.votehub.com/polls`, CC BY 4.0 — attribute [VoteHub](https://votehub.com/polls/api/)).
The `votehub-race-polls-v1` parser expands nested `answers[{choice,pct}]`, maps subjects
such as `2026 Georgia` / `2026 CA-22` onto `US-SEN-{ST}-2026` and hyphenated
`US-HOUSE-{ST}-{NN}-2026` race ids, drops primary-labeled subjects, and maps candidate
names to DEM/REP via `fixtures/candidates_2026_parties.csv` (Wikipedia + VoteHub primary
subject labels). Polls with unmapped candidates are skipped rather than guessed.
Synthetic panel polls for cycle 2026 are excluded from those live registries so curated
2026 poll rows come from VoteHub only.

`fred_unrate_2026_fundamentals` downloads FRED's keyless UNRATE CSV and emits
model-bearing national macro fundamentals rows for the compact 2026
Senate/Governor/House verification races. The unemployment value is transformed into a
small signed `economic_index`; non-economic race priors such as partisan lean and
historical turnout remain configured in `sources_live.yaml` and are documented there as
adapter assumptions.

The same live registry includes Wikipedia raw-page race-presence metadata for
`US-SEN-GA-2026`, `US-GOV-GA-2026`, and `US-HOUSE-CA45-2026`. These entries use
`http_text` and emit neutral `public_signals` rows with `z_score = 0.0`. They are useful
for exercising keyless text ingestion, parser provenance, and the Phase 8 scope audit,
but they are metadata-only. They do not influence forecasts while public signals remain
untrusted, and they do not satisfy the production-default live-source gate without
model-bearing poll, fundamentals, or market rows.

The compact presidential panel is exposed through five parser versions that all point to
the same raw file and emit separate curated contracts:

- `president-state-panel-races-v1`
- `president-state-panel-options-v1`
- `president-state-panel-results-v1`
- `president-state-panel-fundamentals-v1`
- `president-state-panel-polls-v1`

`configs/sources_historical_panels.yaml` is a separate production-dimension synthetic
registry for congressional validation. It loads:

- `fixtures/senate_state_panel.csv`: 33-34 Senate races per cycle for 2014-2026.
- `fixtures/house_district_panel.csv`: 435 House districts per cycle for 2014-2026.
- the compact fixture registry for Governor/local support tables and non-model tables.

This registry is intentionally not the default source config. Use it when the operator
wants Phase 4/5/7 historical calibration and sampler diagnostics at full Senate/House
dimensions without increasing routine test and smoke-run cost.

Remaining source families to implement include MIT Election Lab, FEC, additional
poll feeds beyond VoteHub, Census/FRED/BEA/BLS, Kalshi, Polymarket, GDELT, Wikimedia,
and optional Civic-style race catalog enrichment.
