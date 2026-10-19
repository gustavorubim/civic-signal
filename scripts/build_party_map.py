"""Rebuild ``fixtures/candidates_2026_parties.csv`` with a nominee ``status`` column.

Inputs:
* the previous party map (names seen in VoteHub choices, some from primaries), and
* ``fixtures/ballot_2026_candidates.csv`` (general-election candidates parsed from the
  Wikipedia 2026 Senate/House summary tables by ``scripts/parse_wikipedia_2026_ballots.py``).

Rules (no inference beyond the sources):
* a name listed on the general-election ballot for its seat takes the party from that
  listing; status ``nominee`` for DEM/REP/IND and ``minor`` for other parties;
* a name in the previous map that is *not* on the seat's general-election list keeps its
  previous party label and gets status ``not_on_ballot`` (primary loser, withdrawn or
  hypothetical; the parser treats all three the same);
* every general-election candidate for every seat is included so full-ballot polls map.
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civic_signal.normalize.votehub import normalize_name  # noqa: E402

FIELDS = ["office", "jurisdiction", "name", "party", "status", "source"]
WIKI_SOURCE = "wikipedia-2026-summary-table"
# VoteHub spelling -> name as listed on the seat's general-election ballot (same person).
SEAT_ALIASES: dict[tuple[str, str, str], str] = {
    ("senate", "Alaska", "Dan Sullivan"): "Dan S. Sullivan",  # incumbent; not Dan J. Sullivan
    ("senate", "Alaska", "Dan Sullvian"): "Dan S. Sullivan",
    ("senate", "New Hampshire", "John Sununu"): "John E. Sununu",
    ("senate", "Ohio", "Bill Redpath"): "William Redpath",
    ("house", "NJ-07", "Tom Kean Jr."): "Thomas Kean Jr.",
    ("house", "ME-02", "Matt Dunlap"): "Matthew Dunlap",
    ("house", "MI-10", "Michael Bouchard"): "Mike Bouchard Jr.",
    ("house", "NH-01", "Stephany Shaheen"): "Stefany Shaheen",
    ("house", "NM-01", "Ndidiamaka Okpareke"): "Didi Okpareke",
    ("house", "UT-03", "Kent S. Udell"): "Kent Udell",
}
# VoteHub-only spellings that are not in the prior map but need an alias row.
EXTRA_ALIAS_NAMES = [key for key in SEAT_ALIASES]
GENERIC_NAMES = {"dem", "rep", "democrat", "republican", "democratic", "gop"}
PRIOR_SOURCE = "prior-map"


def _key(office: str, jurisdiction: str, name: str) -> tuple[str, str, str]:
    folded = unicodedata.normalize("NFKD", normalize_name(name)).encode("ascii", "ignore")
    return office, jurisdiction.strip(), folded.decode()


def build(prior_rows: list[dict[str, str]], ballot_rows: list[dict[str, str]]) -> list[dict]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    ballot_seats = {(r["office"], r["jurisdiction"]) for r in ballot_rows}
    for row in ballot_rows:
        party = row["party"].upper()
        out[_key(row["office"], row["jurisdiction"], row["name"])] = {
            "office": row["office"],
            "jurisdiction": row["jurisdiction"],
            "name": row["name"],
            "party": party,
            "status": "nominee" if party in {"DEM", "REP", "IND"} else "minor",
            "source": WIKI_SOURCE,
        }
    seen = {(r["office"], r.get("jurisdiction", ""), r["name"]) for r in prior_rows}
    extra = [
        {"office": o, "jurisdiction": j, "name": n, "party": ""}
        for (o, j, n) in EXTRA_ALIAS_NAMES
        if (o, j, n) not in seen
    ]
    for row in [*prior_rows, *extra]:
        office, juris, name = row["office"], row.get("jurisdiction", ""), row["name"]
        if normalize_name(name) in GENERIC_NAMES:
            continue  # generic party labels are handled in the parser
        target = SEAT_ALIASES.get((office, juris, name))
        if target is not None:
            canonical = out.get(_key(office, juris, target))
            if canonical is not None:
                out[(office, juris, f"alias:{name}")] = dict(canonical, name=name)
                continue
        if not row["party"]:
            continue
        key = _key(office, juris, name)
        if key in out:
            if normalize_name(name) != normalize_name(out[key]["name"]):
                # keep the VoteHub spelling too, as an alias row with the same facts
                alias = dict(out[key], name=name)
                out[(office, juris, f"alias:{name}")] = alias
            continue
        status = "not_on_ballot" if (office, juris) in ballot_seats else "unverified"
        out[key] = {
            "office": office,
            "jurisdiction": juris,
            "name": name,
            "party": row["party"].upper(),
            "status": row.get("status") or status,
            "source": row.get("source") or PRIOR_SOURCE,
        }
    return sorted(out.values(), key=lambda r: (r["office"], r["jurisdiction"], r["name"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=ROOT / "fixtures/candidates_2026_parties.csv")
    parser.add_argument("--ballot", type=Path, default=ROOT / "fixtures/ballot_2026_candidates.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures/candidates_2026_parties.csv")
    args = parser.parse_args(argv)
    with args.prior.open(newline="", encoding="utf-8") as handle:
        prior = list(csv.DictReader(handle))
    with args.ballot.open(newline="", encoding="utf-8") as handle:
        ballot = list(csv.DictReader(handle))
    rows = build(prior, ballot)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
