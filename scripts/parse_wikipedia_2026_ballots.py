"""Parse the 2026 Senate/House Wikipedia summary tables into ballot and seat CSVs.

Source: the ``?action=raw`` wikitext of "2026 United States Senate elections" and
"2026 United States House of Representatives elections" (Wikipedia, CC BY-SA 4.0).
The live configs already fetch these pages (``wikipedia_*_race_presence`` sources), so
the same snapshot used for a run can be parsed here. Outputs (both CC BY-SA derived):

* ``fixtures/ballot_2026_candidates.csv``: office, jurisdiction, name, party, party_label
  (general-election candidates as listed on the page on the snapshot date)
* ``fixtures/ballot_2026_seats.csv``: office, jurisdiction, cook_pvi (D positive),
  incumbent names/parties, incumbent status text, open_seat flag, special flag

Nothing is inferred: rows/fields the parser cannot read are left blank.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip
STATE_NAME = {abbr: name for name, abbr in STATE_ABBR.items()}

PARTY_CODES = {
    "democratic": "DEM",
    "republican": "REP",
    "independent": "IND",
    "libertarian": "LIB",
    "green": "GRN",
    "dfl": "DEM",
    "democratic\u2013farmer\u2013labor": "DEM",
    "democratic-npl": "DEM",
}

_SENATE_HEADER = re.compile(
    r"^!\s*\[\[2026 United States Senate (special )?election in ([A-Za-z .]+)\|"
)
_HOUSE_HEADER = re.compile(r"^!(?:rowspan=\d+\s*\|)?\s*\{\{ushr\|([A-Z]{2})\|(AL|\d+)\|", re.I)
_PVI = re.compile(r"\{\{\s*shading PVI\|(D|R|EVEN)\|?(\d+)?", re.I)
_CANDIDATE = re.compile(r"^\*\s*\{\{Party stripe\|[^}]*\}\}(.+?)\s*\(([^()]*)\)\s*(?:<ref|$)", re.I)
_SORTNAME = re.compile(r"\{\{sortname\|([^|}]+)\|([^|}]+)")
_LINK = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
_PARTY_CELL = re.compile(r"\|\s*(Democratic|Republican|Independent|DFL)\s*$")


@dataclass
class Seat:
    office: str
    jurisdiction: str
    special: bool = False
    cook_pvi: float | None = None
    incumbents: list[tuple[str, str]] = field(default_factory=list)
    status: list[str] = field(default_factory=list)
    candidates: list[tuple[str, str, str]] = field(default_factory=list)


def _clean_name(raw: str) -> str:
    text = _LINK.sub(lambda m: m.group(1), raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def party_code(label: str) -> str:
    key = label.strip().lower()
    return PARTY_CODES.get(key, "OTH")


_STATUS = re.compile(r"(?i)(incumbent|new seat|open seat|interim appointee)[^|]*$")


def _status_text(line: str) -> str | None:
    if not line.startswith("|"):
        return None
    text = _clean_name(re.sub(r"<br\s*/?>", " ", re.sub(r"<ref.*", "", line)))
    match = _STATUS.search(text)
    return match.group(0).strip().rstrip(".") if match else None


def parse_rows(text: str, office: str) -> list[Seat]:
    seats: list[Seat] = []
    current: Seat | None = None
    in_candidates = False
    continuation = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        header = _SENATE_HEADER.match(line) if office == "senate" else _HOUSE_HEADER.match(line)
        if header:
            if office == "senate":
                state = STATE_ABBR.get(header.group(2).strip())
                if state is None:
                    current = None
                    continue
                current = Seat("senate", state, special=bool(header.group(1)))
            else:
                state, number = header.group(1).upper(), header.group(2).upper()
                if state not in STATE_NAME:
                    current = None
                    continue
                nn = "01" if number == "AL" else f"{int(number):02d}"
                current = Seat("house", f"{state}-{nn}")
            seats.append(current)
            in_candidates = False
            continuation = False
            continue
        if current is None:
            continue
        if line.startswith("|}") or (line.startswith("!") and not in_candidates):
            if line.startswith("|}"):
                current = None
            in_candidates = False
            continue
        if line == "|-":
            in_candidates = False
            continuation = True
            continue
        pvi = _PVI.search(line)
        if pvi and current.cook_pvi is None:
            sign = pvi.group(1).upper()
            current.cook_pvi = (
                0.0 if sign == "EVEN" else float(pvi.group(2) or 0) * (1.0 if sign == "D" else -1.0)
            )
            continue
        if re.search(r"(?i)plainlist", line) and not line.startswith("*"):
            in_candidates = not re.search(r"(?i)endplainlist", line)
            inline = re.search(r"\*\s*\{\{Party stripe.*", line)
            if not inline:
                continue
            line = inline.group(0)
        if line.startswith("}}") or re.search(r"(?i)\{\{endplainlist\}\}", line):
            in_candidates = False
            continue
        if line.startswith("*"):
            match = _CANDIDATE.match(line)
            if match and in_candidates:
                name = _clean_name(match.group(1))
                label = match.group(2).strip()
                if name:
                    current.candidates.append((name, party_code(label), label))
            continue
        sort = _SORTNAME.search(line)
        if line.startswith("|") and sort and (not current.status or continuation):
            current.incumbents.append((f"{sort.group(1)} {sort.group(2)}".strip(), ""))
            continuation = False
            continue
        if (
            line.startswith("|")
            and not current.incumbents
            and _LINK.fullmatch(re.sub(r"\{\{efn.*", "", line.lstrip("| ")).strip())
        ):
            cell = re.sub(r"\{\{efn.*", "", line.lstrip("| "))
            current.incumbents.append((_clean_name(cell), ""))
            continue
        party = _PARTY_CELL.search(line)
        if party and current.incumbents and current.incumbents[-1][1] == "":
            name, _ = current.incumbents[-1]
            current.incumbents[-1] = (name, party_code(party.group(1)))
            continue
        status = _status_text(line)
        if status:
            current.status.append(status)
    return seats


def summary_seats(seats: list[Seat]) -> list[Seat]:
    """Keep one row per seat: the last summary-table row that lists candidates."""
    chosen: dict[tuple[str, str, bool], Seat] = {}
    for seat in seats:
        if seat.candidates:
            chosen[(seat.office, seat.jurisdiction, seat.special)] = seat
    return sorted(chosen.values(), key=lambda s: (s.office, s.jurisdiction, s.special))


def write_outputs(seats: list[Seat], candidates_path: Path, seats_path: Path) -> None:
    with candidates_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["office", "jurisdiction", "name", "party", "party_label"])
        for seat in seats:
            jurisdiction = (
                STATE_NAME[seat.jurisdiction] if seat.office == "senate" else (seat.jurisdiction)
            )
            for name, code, label in seat.candidates:
                writer.writerow([seat.office, jurisdiction, name, code, label])
    with seats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "office",
                "jurisdiction",
                "special",
                "cook_pvi",
                "incumbents",
                "incumbent_parties",
                "incumbent_status",
                "open_seat",
                "n_candidates",
            ]
        )
        for seat in seats:
            status = "; ".join(seat.status)
            running = [s for s in seat.status if not re.match(r"(?i)^incumbent retir", s)]
            open_seat = bool(seat.status) and not any(
                re.match(
                    r"(?i)^(incumbent (renominated|advanced|running|re-elected)"
                    r"|interim appointee nominated)",
                    s,
                )
                for s in running
            )
            writer.writerow(
                [
                    seat.office,
                    seat.jurisdiction,
                    int(seat.special),
                    "" if seat.cook_pvi is None else seat.cook_pvi,
                    "; ".join(name for name, _ in seat.incumbents),
                    "; ".join(party for _, party in seat.incumbents),
                    status,
                    int(open_seat),
                    len(seat.candidates),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--senate-wikitext", type=Path, required=True)
    parser.add_argument("--house-wikitext", type=Path, required=True)
    parser.add_argument(
        "--candidates-out", type=Path, default=ROOT / "fixtures/ballot_2026_candidates.csv"
    )
    parser.add_argument("--seats-out", type=Path, default=ROOT / "fixtures/ballot_2026_seats.csv")
    args = parser.parse_args(argv)
    seats = parse_rows(args.senate_wikitext.read_text(encoding="utf-8"), "senate")
    seats += parse_rows(args.house_wikitext.read_text(encoding="utf-8"), "house")
    seats = summary_seats(seats)
    write_outputs(seats, args.candidates_out, args.seats_out)
    senate = sum(1 for s in seats if s.office == "senate")
    print(f"senate seats={senate} house seats={len(seats) - senate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
