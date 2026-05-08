# API Requirements

The current repo runs without external credentials because it uses deterministic
fixtures. Live adapters should read optional credentials from `.env` or shell
environment variables and record whether each source ran authenticated or public-only
in `source_manifest.parquet`.

## Not Needed For Current Runs

- Plot generation: no keys. Plots are generated from local Parquet/JSON artifacts.
- Fixture ingestion: no keys.
- Backtests over committed golden fixtures: no keys.

## Recommended Live Credentials

- `GOOGLE_CIVIC_API_KEY`: required for Google Civic Information API requests.
- `CENSUS_API_KEY`: recommended for Census API volume; Census documents a key need for
  mobile/web apps or more than 500 daily queries.
- `GDELT_API_KEY`: needed for GDELT Cloud API endpoints that require bearer auth.

## Usually Public Or Keyless For Read-Only Use

- Polymarket market data: public market/event endpoints are generally keyless; trading,
  portfolio, and authenticated WebSocket flows require credentials and are out of scope.
- Kalshi market data: public market data can be read without trading credentials; trading
  and account endpoints are out of scope.
- Wikimedia pageviews: public analytics reads are keyless for this use case.
- FEC and official election-office downloads: design adapters to support public downloads
  first and optional keys/rate-limit settings where available.

Before implementing each live adapter, re-check that source's current terms, rate limits,
and authentication requirements. The sync layer must record source URL, retrieval time,
content hash, parser version, and any auth mode in the manifest.
