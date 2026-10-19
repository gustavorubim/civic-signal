"""Rebuild the 2026 rows of ``fixtures/senate_state_panel.csv`` from cited data (audit P0-4, P0-5).

Only ``cycle == 2026`` rows are replaced; the 2014-2024 rows are the synthetic backtest panel
from ``scripts/generate_senate_panel.py`` and are kept as-is, flagged ``synthetic=True``.

Sources (see ``fixtures/senate_2026_panel.SOURCES.txt``):
* seats, specials, incumbents, open-seat status and Cook PVI:
  ``fixtures/ballot_2026_seats.csv`` (parsed from the Wikipedia 2026 Senate elections
  article by ``scripts/parse_wikipedia_2026_ballots.py``; CC BY-SA 4.0);
* general-election candidates: ``fixtures/ballot_2026_candidates.csv`` (same Wikipedia parse);
* national_swing: the measured generic-ballot median in ``fixtures/measured_environment.json``
  (``scripts/measure_environment.py``; resolved by ``civic_signal.features.environment``);
* 2024 presidential result by state: MIT Election Data and Science Lab,
  "U.S. President 1976-2024", doi:10.7910/DVN/42MVDX (CC0);
* previous election for each seat: MIT Election Data and Science Lab,
  "U.S. Senate 1976-2024", doi:10.7910/DVN/PEJ5QU (CC0).

Fields with no cited 2026 source (fundraising, economic index, national swing, turnout
indices, registered voters, the synthetic panel poll columns) are carried over from the
previous synthetic 2026 row and listed in ``synthetic_fields`` so quality labels can flag
them. FL/OH (new specials) carry those fields from the national median of the old rows.
Nothing is invented: a value with no source is either carried over and flagged, or blank.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civic_signal.features.environment import resolve_environment  # noqa: E402

# national_swing is in points of two-party SHARE (= margin change / 2), the units the
# fundamentals model applies it in. It is the change from the environment of the seat's
# previous election (U.S. House national popular-vote margin that year, from the Wikipedia
# "<year> United States House of Representatives elections" infoboxes: 2020 D+3.1,
# 2022 R+2.7, 2024 R+2.6) to the measured 2026 generic-ballot margin, tempered 1 pt for
# Senate candidate effects (same convention as scripts/generate_senate_panel.py).
SENATE_TEMPER = 1.0
HOUSE_POPULAR_VOTE_MARGIN_D = {"2020": 3.1, "2021": 3.1, "2022": -2.7, "2024": -2.6}

STATE_NAMES = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "CO": "Colorado", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "ME": "Maine",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MT": "Montana",
    "NC": "North Carolina", "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "VA": "Virginia", "WV": "West Virginia", "WY": "Wyoming",
}  # fmt: skip
# Most recent election for the seat on the 2026 ballot, when it is not the 2020 Class II
# general: (year, stage, special). GA 2020 went to the Jan 2021 runoff; NE (Sasse seat) had a
# 2024 special; OK (Inhofe seat) a 2022 special; FL/OH are Class III seats last elected 2022.
PREVIOUS_ELECTION: dict[str, tuple[str, str, bool]] = {
    "GA": ("2021", "runoff", False),
    "NE": ("2024", "gen", True),
    "OK": ("2022", "gen", True),
    "FL": ("2022", "gen", False),
    "OH": ("2022", "gen", False),
}
DEFAULT_PREVIOUS = ("2020", "gen", False)
# MIT file rows with a blank party (party_simplified OTHER); parties from the Wikipedia article
# "2020 United States Senate election in Wyoming" (Lummis R, Ben-David D).
PARTY_FIXES = {
    ("2020", "WY", "CYNTHIA M. LUMMIS"): "REPUBLICAN",
    ("2020", "WY", "MERAV BEN DAVID"): "DEMOCRAT",
}
NON_CANDIDATES = {"", "OVER VOTES", "UNDER VOTES", "BLANK VOTES", "SCATTERING", "OTHER"}
CLASS_BY_STATE = {"FL": "III", "OH": "III"}
# Seats with several same-party general-election candidates: the modeled principal
# candidate, with the reason. Alaska is top-four (incumbent Dan S. Sullivan, not the second
# "Dan J. Sullivan" on the ballot); Idaho's Democratic nominee withdrew and the Wikipedia
# infobox lists independent Todd Achilles as the principal challenger (Natalie Fleming is a
# second independent).
PRINCIPAL_CANDIDATE: dict[tuple[str, str], str] = {
    ("Alaska", "REP"): "Dan S. Sullivan",
    ("Idaho", "IND"): "Todd Achilles",
}
CARRIED_FIELDS = [
    "dem_fundraising_usd",
    "rep_fundraising_usd",
    "economic_index",
    "demographic_turnout_index",
    "historical_turnout_rate",
    "registered_voters",
    "pollster",
    "poll_sample_size",
    "poll_population",
    "poll_sponsor_class",
    "poll_methodology",
    "dem_poll_pct",
    "rep_poll_pct",
    "polling_dem_share_base",
]
NEW_COLUMNS = [
    "dem_party",
    "pres_2024_dem_two_party",
    "pres_2024_margin",
    "previous_election",
    "open_seat",
    "special",
    "synthetic",
    "synthetic_fields",
]


def _read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def president_2024(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    """state_po -> (DEM two-party share, DEM-minus-REP margin in points of total vote)."""
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["year"] != "2024" or row.get("writein", "False") == "True":
            continue
        party = row["party_simplified"]
        if party not in {"DEMOCRAT", "REPUBLICAN"}:
            continue
        state = row["state_po"]
        entry = out.setdefault(state, {"DEMOCRAT": 0.0, "REPUBLICAN": 0.0, "total": 0.0})
        entry[party] += float(row["candidatevotes"])
        entry["total"] = float(row["totalvotes"])
    result = {}
    for state, entry in out.items():
        dem, rep, total = entry["DEMOCRAT"], entry["REPUBLICAN"], entry["total"]
        result[state] = (round(dem / (dem + rep), 4), round(100 * (dem - rep) / total, 2))
    return result


def previous_seat_result(rows: list[dict[str, str]], state: str) -> tuple[str, float, float] | None:
    """(label, DEM two-party share, REP two-party share) for the seat's last election.

    The "DEM" side is the top Democrat; in NE 2024 and SD 2020 there was a Democrat on the
    ballot, so the anti-REP independents running in 2026 inherit that (documented) proxy.
    """
    year, stage, special = PREVIOUS_ELECTION.get(state, DEFAULT_PREVIOUS)
    votes: dict[str, float] = {}
    challengers: dict[str, float] = {}
    for row in rows:
        if (
            row["year"] == year
            and row["state_po"] == state
            and row["stage"].lower() == stage
            and (row["special"] == "True") == special
            and row.get("writein", "False") != "True"
        ):
            candidate = row["candidate"].strip()
            party = PARTY_FIXES.get((year, state, candidate), row["party_simplified"])
            count = float(row["candidatevotes"])
            if party in {"DEMOCRAT", "REPUBLICAN"}:
                votes[party] = max(votes.get(party, 0.0), count)
            elif candidate.upper() not in NON_CANDIDATES:
                challengers[candidate] = count
    if "REPUBLICAN" not in votes:
        return None
    label = f"{year} {stage}{' special' if special else ''}"
    if "DEMOCRAT" not in votes:
        # No Democrat on the ballot (e.g. AR 2020): the top non-REP candidate is the proxy.
        if not challengers:
            return None
        top = max(challengers, key=challengers.__getitem__)
        votes["DEMOCRAT"] = challengers[top]
        label += f" (no DEM; top challenger {top.title()})"
    dem, rep = votes["DEMOCRAT"], votes["REPUBLICAN"]
    return label, round(dem / (dem + rep), 4), round(rep / (dem + rep), 4)


def _same_person(left: str, right: str) -> bool:
    """First + last token match ("Dan Sullivan" == "Dan S. Sullivan")."""

    def key(name: str) -> tuple[str, str]:
        tokens = [t for t in name.replace(",", " ").split() if t.rstrip(".").lower() not in
                  {"jr", "sr", "ii", "iii", "iv"}]  # fmt: skip
        return (tokens[0].lower(), tokens[-1].lower()) if tokens else ("", "")

    return key(left) == key(right)


def nominees(ballot: list[dict[str, str]]) -> dict[str, dict[str, list[str]]]:
    """jurisdiction -> party -> general-election candidates (DEM/REP/IND only)."""
    out: dict[str, dict[str, list[str]]] = {}
    for row in ballot:
        if row["office"] != "senate" or row["party"] not in {"DEM", "REP", "IND"}:
            continue
        key = (row["jurisdiction"], row["party"])
        if key in PRINCIPAL_CANDIDATE and row["name"] != PRINCIPAL_CANDIDATE[key]:
            continue
        out.setdefault(row["jurisdiction"], {}).setdefault(row["party"], []).append(row["name"])
    return out


def build_rows(
    seats: list[dict[str, str]],
    ballot: list[dict[str, str]],
    president: list[dict[str, str]],
    senate: list[dict[str, str]],
    old_rows: list[dict[str, str]],
    environment_margin_d: float | None = None,
) -> list[dict[str, str]]:
    pres = president_2024(president)
    noms = nominees(ballot)
    old_2026 = {row["state"]: row for row in old_rows if row["cycle"] == "2026"}
    medians = {}
    for field in CARRIED_FIELDS:
        values = [row[field] for row in old_2026.values() if row.get(field) not in (None, "")]
        try:
            medians[field] = str(round(statistics.median(float(v) for v in values), 4))
        except ValueError:
            medians[field] = statistics.mode(values) if values else ""
    rows = []
    for seat in seats:
        if seat["office"] != "senate":
            continue
        state = seat["jurisdiction"]
        name = STATE_NAMES[state]
        seat_noms = noms.get(name, {})
        rep_names = seat_noms.get("REP", [])
        dem_names = seat_noms.get("DEM", [])
        dem_party = "DEM"
        if not dem_names and seat_noms.get("IND"):
            dem_names, dem_party = seat_noms["IND"], "IND"
        if len(rep_names) != 1 or len(dem_names) != 1:
            raise ValueError(f"{state}: expected one DEM/IND and one REP nominee, got {seat_noms}")
        incumbents = [item.strip() for item in seat["incumbents"].split(";") if item.strip()]
        incumbent_party = seat["incumbent_parties"].split(";")[0].strip()
        open_seat = seat["open_seat"] == "1"
        previous = previous_seat_result(senate, state)
        if previous is None:
            raise ValueError(f"{state}: no previous two-party result in the MIT Senate file")
        prev_label, prev_dem, prev_rep = previous
        dem_share, margin = pres[state]
        old = old_2026.get(state)
        carried = {field: (old or {}).get(field) or medians[field] for field in CARRIED_FIELDS}
        if environment_margin_d is not None:
            reference = HOUSE_POPULAR_VOTE_MARGIN_D[prev_label[:4]]
            carried["national_swing"] = str(
                round((environment_margin_d - SENATE_TEMPER - reference) / 2.0, 4)
            )
        else:
            carried["national_swing"] = (old or {}).get("national_swing") or medians.get(
                "national_swing", ""
            )
        suffix = "I" if dem_party == "IND" else "D"
        race_id = f"US-SEN-{state}-2026"
        rows.append(
            {
                "cycle": "2026",
                "state": state,
                "state_name": name,
                "election_date": "2026-11-03",
                "senate_class": CLASS_BY_STATE.get(state, "II"),
                "dem_name": dem_names[0],
                "rep_name": rep_names[0],
                "dem_incumbent": str(any(_same_person(dem_names[0], i) for i in incumbents)),
                "rep_incumbent": str(any(_same_person(rep_names[0], i) for i in incumbents)),
                "dem_previous_vote_share": str(prev_dem),
                "rep_previous_vote_share": str(prev_rep),
                "dem_vote_share": "",
                "rep_vote_share": "",
                "turnout": "",
                "winner_party": "",
                # Cook PVI (points; negative = REP), same scale as the old synthetic column.
                "partisan_lean": seat["cook_pvi"],
                "incumbency_advantage": "0.0" if open_seat else "3.5",
                "race_id": race_id,
                "dem_option_id": f"{race_id}-{suffix}",
                "rep_option_id": f"{race_id}-R",
                "incumbent_party": "none" if open_seat else incumbent_party,
                "previous_dem_share": str(prev_dem),
                "previous_rep_share": str(prev_rep),
                "dem_party": dem_party,
                "pres_2024_dem_two_party": str(dem_share),
                "pres_2024_margin": str(margin),
                "previous_election": prev_label,
                "open_seat": str(open_seat),
                "special": str(seat["special"] == "1"),
                "synthetic": "False",
                "synthetic_fields": ";".join(CARRIED_FIELDS),
                **carried,
            }
        )
    return sorted(rows, key=lambda row: row["state"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(ROOT / "fixtures" / "senate_state_panel.csv"))
    parser.add_argument("--seats", default=str(ROOT / "fixtures" / "ballot_2026_seats.csv"))
    parser.add_argument("--ballot", default=str(ROOT / "fixtures" / "ballot_2026_candidates.csv"))
    parser.add_argument(
        "--president",
        default=str(ROOT / "data" / "snapshots" / "sources" / "mit_president_1976_2024.csv"),
    )
    parser.add_argument(
        "--senate",
        default=str(ROOT / "data" / "snapshots" / "sources" / "mit_senate_1976_2024.tab"),
    )
    parser.add_argument("--out", default=None, help="defaults to --panel (in place)")
    args = parser.parse_args(argv)
    panel_path = Path(args.panel)
    old_rows = _read(panel_path)
    environment = resolve_environment(ROOT / "fixtures")
    fields = list(old_rows[0].keys())
    for column in NEW_COLUMNS:
        if column not in fields:
            fields.append(column)
    new_2026 = build_rows(
        _read(Path(args.seats)),
        _read(Path(args.ballot)),
        _read(Path(args.president)),
        _read(Path(args.senate), delimiter="\t"),
        old_rows,
        environment["generic_ballot_margin_d"],
    )
    print(
        f"environment: {environment['kind']} D+{environment['generic_ballot_margin_d']} "
        f"({environment.get('source')}, {environment.get('as_of')})"
    )
    history = []
    for row in old_rows:
        if row["cycle"] == "2026":
            continue
        row = dict(row)
        row["dem_party"] = row.get("dem_party") or "DEM"
        row["synthetic"] = "True"
        history.append(row)
    out_path = Path(args.out) if args.out else panel_path
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in history + new_2026:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"wrote {len(history)} synthetic history rows + {len(new_2026)} 2026 rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
