"""Rebuild the 2026 rows of ``fixtures/house_district_panel.csv`` from cited data (audit P0-5).

Only ``cycle == 2026`` rows are replaced. The 2014-2024 rows are the synthetic backtest panel
from ``scripts/generate_house_panel.py``; they are kept and flagged ``synthetic=True``.

Sources (see ``fixtures/house_2026_panel.SOURCES.txt``):
* 2024 presidential result for every district on the 2026 maps (incl. the 10 states with new
  maps): The Downballot, "presidential results by congressional district" (2026 lines),
  Google Sheet 1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0 (free to use with attribution);
* national 2024 two-party share: MIT EDSL "U.S. President 1976-2024" (CC0);
* seats, incumbents, open seats, nominees and Cook PVI: ``fixtures/ballot_2026_seats.csv`` and
  ``fixtures/ballot_2026_candidates.csv`` (Wikipedia 2026 House article, CC BY-SA 4.0);
* incumbent cross-check: ``unitedstates/congress-legislators`` legislators-current.json (CC0);
* environment: ``fixtures/measured_environment.json``.

``partisan_lean`` is the district's 2024 two-party DEM share minus the national 2024 two-party
DEM share, in points. That is a PVI-style lean on the 2026 lines, computed from the
presidential numbers alone. ``previous_*_vote_share`` is the 2024 presidential two-party share
in the district. MIT's 2024 House results are behind a Dataverse guestbook, and most
redistricted seats have no comparable previous House result, so this proxy is used instead
(documented). Carried synthetic fields are listed in ``synthetic_fields``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_senate_2026_panel import _same_person  # noqa: E402

from civic_signal.features.environment import resolve_environment  # noqa: E402

# national_swing is in points of two-party SHARE (= margin change / 2): the change from the
# baseline's environment (2024 national presidential two-party margin, MIT file) to the
# measured 2026 generic-ballot margin. The baseline here is the 2024 presidential share.
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
    "rep_party",
    "pres_2024_dem_two_party",
    "pres_2024_margin",
    "map_changed_2026",
    "open_seat",
    "nominee_note",
    "synthetic",
    "synthetic_fields",
]
SUFFIX = {"DEM": "D", "REP": "R", "IND": "I", "OTH": "O"}
MAX_ABS_LEAN = 60.0


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(text: str) -> float:
    return float(text.replace(",", "").replace("%", "").strip())


def downballot_rows(path: Path) -> dict[str, dict[str, float | bool]]:
    """district (``AK-01`` style) -> Harris/Trump votes, total, map-changed flag."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    out: dict[str, dict[str, float | bool]] = {}
    for row in rows:
        if len(row) < 10 or "-" not in row[0] or not row[4].strip():
            continue
        state, number = row[0].split("-", 1)
        if not (len(state) == 2 and (number.isdigit() or number == "AL")):
            continue
        district = f"{state}-{'01' if number == 'AL' else number.zfill(2)}"
        out[district] = {
            "dem": _num(row[4]),
            "rep": _num(row[5]),
            "total": _num(row[6]),
            # 2020 columns are blank for districts redrawn for 2026.
            "map_changed": len(row) <= 12 or not row[11].strip(),
        }
    return out


def national_two_party(president: list[dict[str, str]]) -> float:
    dem = rep = 0.0
    for row in president:
        if row["year"] == "2024" and row.get("writein", "False") != "True":
            if row["party_simplified"] == "DEMOCRAT":
                dem += float(row["candidatevotes"])
            elif row["party_simplified"] == "REPUBLICAN":
                rep += float(row["candidatevotes"])
    return dem / (dem + rep)


def pick(names: list[str], incumbents: list[str]) -> tuple[str, bool]:
    """Principal candidate among several same-party candidates: the incumbent if present."""
    for name in names:
        if any(_same_person(name, inc) for inc in incumbents):
            return name, True
    return names[0], len(names) == 1


def choose_sides(
    candidates: dict[str, list[str]], incumbents: list[str]
) -> tuple[tuple[str, str], tuple[str, str], str]:
    """((dem_side_party, name), (rep_side_party, name), note)."""
    dems = sorted(candidates.get("DEM", []))
    reps = sorted(candidates.get("REP", []))
    inds = sorted(candidates.get("IND", []))
    note = ""
    if dems and reps:
        dem, dem_sure = pick(dems, incumbents)
        rep, rep_sure = pick(reps, incumbents)
        if not (dem_sure and rep_sure):
            note = "several same-party candidates listed; principal not yet determined TODO(verify)"
        return ("DEM", dem), ("REP", rep), note
    if len(dems) >= 2 and not reps:  # top-two D v D (California)
        first, _ = pick(dems, incumbents)
        second = next(name for name in dems if name != first)
        return ("DEM", first), ("DEM", second), "same-party general (DEM v DEM)"
    if len(reps) >= 2 and not dems:  # top-two R v R (CA-40)
        first, _ = pick(reps, incumbents)
        second = next(name for name in reps if name != first)
        return ("REP", second), ("REP", first), "same-party general (REP v REP)"
    if dems and inds:
        dem, _ = pick(dems, incumbents)
        ind, _ = pick(inds, incumbents)
        return ("DEM", dem), ("IND", ind), "DEM v independent"
    if reps and inds:
        rep, _ = pick(reps, incumbents)
        ind, _ = pick(inds, incumbents)
        return ("IND", ind), ("REP", rep), "independent v REP"
    if dems:
        dem, _ = pick(dems, incumbents)
        return ("DEM", dem), ("OTH", "No major-party opponent"), "no DEM/REP/IND opponent"
    if reps:
        rep, _ = pick(reps, incumbents)
        return ("OTH", "No major-party opponent"), ("REP", rep), "no DEM/REP/IND opponent"
    raise ValueError(f"no DEM/REP/IND candidate: {candidates}")


def build_rows(
    seats: list[dict[str, str]],
    ballot: list[dict[str, str]],
    downballot: dict[str, dict[str, float | bool]],
    national_dem_two_party: float,
    old_rows: list[dict[str, str]],
    environment_margin_d: float | None,
) -> list[dict[str, str]]:
    national_margin = 100 * (2 * national_dem_two_party - 1)
    candidates: dict[str, dict[str, list[str]]] = {}
    for row in ballot:
        if row["office"] == "house":
            candidates.setdefault(row["jurisdiction"], {}).setdefault(row["party"], []).append(
                row["name"]
            )
    old_2026 = {row["district"]: row for row in old_rows if row["cycle"] == "2026"}
    rows = []
    for seat in seats:
        if seat["office"] != "house":
            continue
        district = seat["jurisdiction"]
        state = district.split("-")[0]
        result = downballot[district]
        dem_votes, rep_votes = float(result["dem"]), float(result["rep"])
        dem_two_party = dem_votes / (dem_votes + rep_votes)
        lean = round(100 * (dem_two_party - national_dem_two_party), 2)
        if abs(lean) > MAX_ABS_LEAN:
            raise ValueError(f"{district}: |lean| {lean} > {MAX_ABS_LEAN}")
        incumbents = [item.strip() for item in seat["incumbents"].split(";") if item.strip()]
        inc_parties = [p.strip() for p in seat["incumbent_parties"].split(";") if p.strip()]
        open_seat = seat["open_seat"] == "1" or not incumbents
        (dem_party, dem_name), (rep_party, rep_name), note = choose_sides(
            candidates.get(district, {}), incumbents
        )
        dem_suffix = SUFFIX[dem_party] + ("2" if dem_party == rep_party else "")
        rep_suffix = SUFFIX[rep_party]
        race_id = f"US-HOUSE-{district}-2026"
        previous_dem = round(dem_two_party, 4)
        if rep_party == "OTH":  # unopposed DEM: the placeholder gets no previous share
            previous_dem = 1.0
        elif dem_party == "OTH":
            previous_dem = 0.0
        old = old_2026.get(district, {})
        carried = {field: old.get(field, "") for field in CARRIED_FIELDS}
        swing = (
            str(round((environment_margin_d - national_margin) / 2.0, 4))
            if environment_margin_d is not None
            else old.get("national_swing", "")
        )
        rows.append(
            {
                "cycle": "2026",
                "state": state,
                "state_name": old.get("state_name", state),
                "district": district,
                "election_date": "2026-11-03",
                "redistricting_era": "2022_plus",
                "cook_pvi": seat["cook_pvi"],
                "competitive": str(abs(lean) <= 8),
                "dem_name": dem_name,
                "rep_name": rep_name,
                "dem_incumbent": str(any(_same_person(dem_name, i) for i in incumbents)),
                "rep_incumbent": str(any(_same_person(rep_name, i) for i in incumbents)),
                "dem_previous_vote_share": str(previous_dem),
                "rep_previous_vote_share": str(round(1 - previous_dem, 4)),
                "dem_vote_share": "",
                "rep_vote_share": "",
                "turnout": "",
                "winner_party": "",
                "partisan_lean": str(lean),
                "incumbency_advantage": "0.0" if open_seat else "4.0",
                "national_swing": swing,
                "race_id": race_id,
                "dem_option_id": f"{race_id}-{dem_suffix}",
                "rep_option_id": f"{race_id}-{rep_suffix}",
                "incumbent_party": "none"
                if open_seat
                else (inc_parties[0] if inc_parties else "none"),
                "previous_dem_share": str(previous_dem),
                "previous_rep_share": str(round(1 - previous_dem, 4)),
                "dem_party": dem_party,
                "rep_party": rep_party,
                "pres_2024_dem_two_party": str(round(dem_two_party, 4)),
                "pres_2024_margin": str(
                    round(100 * (dem_votes - rep_votes) / float(result["total"]), 2)
                ),
                "map_changed_2026": str(bool(result["map_changed"])),
                "open_seat": str(open_seat),
                "nominee_note": note,
                "synthetic": "False",
                "synthetic_fields": ";".join(CARRIED_FIELDS),
                **carried,
            }
        )
    return sorted(rows, key=lambda row: row["district"])


def qa(rows: list[dict[str, str]], seats: list[dict[str, str]]) -> dict[str, object]:
    """Sanity checks; raises on failure, returns a summary."""
    house_seats = [seat for seat in seats if seat["office"] == "house"]
    if len(rows) != 435 or len(house_seats) != 435:
        raise ValueError(f"expected 435 rows, got {len(rows)} (seats {len(house_seats)})")
    if len({row["race_id"] for row in rows}) != 435:
        raise ValueError("duplicate race ids")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["incumbent_party"]] = counts.get(row["incumbent_party"], 0) + 1
    expected: dict[str, int] = {}
    for seat in house_seats:
        party = seat["incumbent_parties"].split(";")[0].strip()
        key = "none" if seat["open_seat"] == "1" or not seat["incumbents"].strip() else party
        expected[key] = expected.get(key, 0) + 1
    if counts != expected:
        raise ValueError(f"incumbent counts {counts} != source {expected}")
    worst = max(abs(float(row["partisan_lean"])) for row in rows)
    return {"rows": len(rows), "incumbent_party_counts": counts, "max_abs_lean": worst}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = ROOT / "data" / "snapshots" / "sources"
    parser.add_argument("--panel", default=str(ROOT / "fixtures" / "house_district_panel.csv"))
    parser.add_argument("--seats", default=str(ROOT / "fixtures" / "ballot_2026_seats.csv"))
    parser.add_argument("--ballot", default=str(ROOT / "fixtures" / "ballot_2026_candidates.csv"))
    parser.add_argument(
        "--downballot", default=str(sources / "downballot_pres2024_by_cd_2026_maps.csv")
    )
    parser.add_argument("--president", default=str(sources / "mit_president_1976_2024.csv"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    panel_path = Path(args.panel)
    old_rows = _read(panel_path)
    fields = list(old_rows[0].keys())
    for column in NEW_COLUMNS:
        if column not in fields:
            fields.append(column)
    seats = _read(Path(args.seats))
    environment = resolve_environment(ROOT / "fixtures")
    president = _read(Path(args.president))
    new_rows = build_rows(
        seats,
        _read(Path(args.ballot)),
        downballot_rows(Path(args.downballot)),
        national_two_party(president),
        old_rows,
        environment["generic_ballot_margin_d"],
    )
    summary = qa(new_rows, seats)
    history = []
    for row in old_rows:
        if row["cycle"] == "2026":
            continue
        row = dict(row)
        row["dem_party"] = row.get("dem_party") or "DEM"
        row["rep_party"] = row.get("rep_party") or "REP"
        row["synthetic"] = "True"
        history.append(row)
    out_path = Path(args.out) if args.out else panel_path
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in history + new_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"QA: {summary}; national 2024 D two-party {national_two_party(president):.4f}")
    print(f"wrote {len(history)} synthetic rows + {len(new_rows)} 2026 rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
