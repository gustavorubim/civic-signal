# Data Sources

The source registry lives in `configs/sources.yaml`. Each source records:

- stable source id
- logical curated table
- retrieval type
- URL or local path
- parser version
- license or terms note

The fixture registry is intentionally shaped like public sources so live adapters for
FiveThirtyEight, MIT Election Lab, FEC, VoteHub, civicAPI, Census, Kalshi, Polymarket,
GDELT, and Wikimedia can write the same raw manifest and curated tables.

