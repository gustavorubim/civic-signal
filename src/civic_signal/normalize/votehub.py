"""VoteHub race-poll parser (``votehub-race-polls-v1`` / ``-v2``).

v2 behaviour (the live default since the Oct 2026 audit fixes):

* answers whose name is not in the party map become ``OTHER`` instead of discarding
  the whole poll (P0-3); unmapped names are counted in the accounting output;
* a poll is rejected when two answers map to the same modeled party
  (``dropped_same_party``) or when a modeled side is missing (``dropped_missing_side``)
  (P0-2);
* with ``require_nominees`` the modeled-party answers must be general-election
  nominees per the party map ``status`` column (``dropped_non_nominee``) (P0-1a);
* versions of the same poll (same race, pollster and field dates) are collapsed:
  LV > RV > A, then full ballot over head-to-head, then candidate-named over a
  generic party ballot (``dropped_duplicate``) (P0-7);
* the sponsor party of partisan polls is kept in ``sponsor_party`` (P1-4).

v1 is kept for reproducibility of earlier runs: one unmapped answer drops the poll and
same-party answers are summed, with no nominee filter and no de-duplication.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

GENERIC_LABELS: dict[str, str] = {
    "dem": "DEM",
    "democrat": "DEM",
    "democratic": "DEM",
    "rep": "REP",
    "republican": "REP",
    "gop": "REP",
}
NAME_ALIASES: dict[str, str] = {
    # VoteHub choice-string typos / variants -> canonical public name.
    "dan sullvian": "dan sullivan",
    "thom thillis": "thom tillis",
    "janelle stetson": "janelle stelson",
    "nick begich iii": "nick begich",
    "brian steil": "bryan steil",
    "gregory levy": "greg levy",
    "bill redpath": "william redpath",
    "maria elvira salazar": "maría elvira salazar",
    "christopher schmidt": "chris schmidt",
}
NOMINEE_STATUSES = {"nominee", "generic"}
POPULATION_RANK = {"lv": 0, "v": 0, "rv": 1, "a": 2}
POLL_SCHEMA: dict[str, pl.DataType] = {
    "poll_id": pl.Utf8,
    "cycle": pl.Int64,
    "office_type": pl.Utf8,
    "race_id": pl.Utf8,
    "pollster": pl.Utf8,
    "start_date": pl.Utf8,
    "end_date": pl.Utf8,
    "population": pl.Utf8,
    "sample_size": pl.Float64,
    "sponsor_class": pl.Utf8,
    "sponsor_party": pl.Utf8,
    "methodology": pl.Utf8,
    "option_id": pl.Utf8,
    "pct": pl.Float64,
}


@dataclass(frozen=True)
class PartyEntry:
    party: str
    status: str


@dataclass
class PartyMap:
    by_key: dict[tuple[str, str], PartyEntry] = field(default_factory=dict)
    conflicts: set[tuple[str, str]] = field(default_factory=set)

    def lookup(self, jurisdiction: str, choice: str) -> PartyEntry | None:
        name = normalize_name(choice)
        for juris in (jurisdiction, ""):
            if (juris, name) in self.conflicts:
                return None
            entry = self.by_key.get((juris, name))
            if entry is not None:
                return entry
        ascii_name = _ascii(name)
        for (juris, key), entry in self.by_key.items():
            if juris == jurisdiction and _ascii(key) == ascii_name:
                return entry
        if name in GENERIC_LABELS:
            return PartyEntry(GENERIC_LABELS[name], "generic")
        return None


def _ascii(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_name(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    return NAME_ALIASES.get(text, text)


def jurisdiction_key(office: str, jurisdiction: str, state_lookup: dict[str, str]) -> str:
    """Senate rows key on the state abbreviation; House rows on ``ST-NN``."""
    text = str(jurisdiction or "").strip()
    if not text:
        return ""
    if office == "senate":
        return state_lookup.get(text.upper(), text.upper())
    match = re.fullmatch(r"([A-Za-z]{2})[-\s]?(\d{1,2}|AL)", text)
    if not match:
        return text.upper()
    number = match.group(2).upper()
    nn = "01" if number == "AL" else f"{int(number):02d}"
    return f"{match.group(1).upper()}-{nn}"


def load_party_map(path: Path, office: str, state_lookup: dict[str, str]) -> PartyMap:
    if not path.exists():
        raise FileNotFoundError(f"votehub party map not found: {path}")
    result = PartyMap()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "party"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("votehub party map must include at least name,party columns")
        for row in reader:
            name = str(row.get("name") or "").strip()
            party = str(row.get("party") or "").strip().upper()
            row_office = str(row.get("office") or "").strip().lower()
            if not name or not party:
                continue
            if row_office and row_office != office:
                continue
            status = str(row.get("status") or "nominee").strip().lower() or "nominee"
            juris = jurisdiction_key(office, str(row.get("jurisdiction") or ""), state_lookup)
            key = (juris, normalize_name(name))
            if key in result.conflicts:
                continue
            existing = result.by_key.get(key)
            if existing is not None and existing.party != party:
                result.by_key.pop(key)
                result.conflicts.add(key)
                continue
            result.by_key[key] = PartyEntry(party, status)
    return result


@dataclass
class VoteHubAccounting:
    polls_in_payload: int = 0
    skipped_other_poll_type: int = 0
    skipped_primary: int = 0
    skipped_unknown_race: int = 0
    skipped_same_party_race: int = 0
    dropped_malformed: int = 0
    dropped_unmapped: int = 0
    dropped_same_party: int = 0
    dropped_missing_side: int = 0
    dropped_non_nominee: int = 0
    dropped_duplicate: int = 0
    polls_used: int = 0
    one_sided_polls_used: int = 0
    unmapped_names: Counter[str] = field(default_factory=Counter)
    non_nominee_names: Counter[str] = field(default_factory=Counter)
    used_by_race: Counter[str] = field(default_factory=Counter)
    outcomes: dict[str, str] = field(default_factory=dict)

    def mark(self, poll: object, outcome: str) -> None:
        """Count one raw poll under ``outcome`` and remember its id."""
        setattr(self, outcome, int(getattr(self, outcome)) + 1)
        if isinstance(poll, dict):
            self.outcomes[f"votehub-{poll.get('id')}"] = outcome

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            key: value
            for key, value in self.__dict__.items()
            if not isinstance(value, (Counter, dict))
        }
        payload["unmapped_names"] = dict(self.unmapped_names.most_common())
        payload["non_nominee_names"] = dict(self.non_nominee_names.most_common())
        payload["used_by_race"] = dict(sorted(self.used_by_race.items()))
        return payload


def is_primary_subject(subject: str) -> bool:
    return bool(re.search(r"(?i)\b(democratic|democrat|republican|gop)\b", subject or ""))


def votehub_race_id(
    *,
    subject: str,
    seat_name: object,
    office: str,
    cycle: int,
    house_format: str,
    state_lookup: dict[str, str],
    state_abbreviations: set[str],
) -> str | None:
    text = re.sub(r"(?i)^20\d{2}\s+", "", str(subject or "").strip())
    text = re.sub(r"(?i)\s*(democratic|democrat|republican|gop)\s*$", "", text).strip()
    if office == "senate":
        state = state_lookup.get(text.upper()) or state_lookup.get(text.upper().replace(".", ""))
        if state is None:
            return None
        return f"US-SEN-{state}-{cycle}"
    seat = str(seat_name or "").strip().upper() or text.upper()
    match = re.fullmatch(r"([A-Z]{2})[-\s]?(\d{1,2})", seat.replace(" ", ""))
    if not match:
        match = re.search(r"\b([A-Z]{2})-(\d{1,2})\b", seat)
    if not match:
        return None
    state, number = match.group(1), int(match.group(2))
    if state not in state_abbreviations:
        return None
    if house_format == "hyphenated":
        return f"US-HOUSE-{state}-{number:02d}-{cycle}"
    if house_format != "compact":
        raise ValueError("house_race_id_format must be compact or hyphenated")
    return f"US-HOUSE-{state}{number}-{cycle}"


def _race_jurisdiction(race_id: str, office: str) -> str:
    parts = race_id.split("-")
    if office == "senate":
        return parts[2] if len(parts) > 2 else ""
    if len(parts) >= 5:
        return f"{parts[2]}-{parts[3]}"
    match = re.fullmatch(r"([A-Z]{2})(\d+)", parts[2] if len(parts) > 2 else "")
    return f"{match.group(1)}-{int(match.group(2)):02d}" if match else ""


def parse_votehub_polls(
    raw_text: str,
    parser_args: dict[str, object],
    *,
    root: Path,
    state_lookup: dict[str, str],
    state_abbreviations: set[str],
    parser_version: str = "votehub-race-polls-v2",
) -> tuple[pl.DataFrame, VoteHubAccounting]:
    required_args = {"cycle", "office", "poll_type", "parties", "party_map_path"}
    missing_args = required_args.difference(parser_args)
    if missing_args:
        raise ValueError(
            f"{parser_version} parser_args missing required keys: {sorted(missing_args)}"
        )
    legacy = parser_version == "votehub-race-polls-v1"
    cycle = int(parser_args["cycle"])
    office = str(parser_args["office"]).lower().strip()
    poll_type = str(parser_args["poll_type"]).strip()
    parties = [str(party).upper() for party in list(parser_args["parties"])]
    if office not in {"senate", "house"}:
        raise ValueError(f"{parser_version} parser_args.office must be senate or house")
    if not parties:
        raise ValueError(f"{parser_version} parser_args.parties must be non-empty")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{parser_version} raw payload is not valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{parser_version} expects a JSON array of poll objects")
    map_path = Path(str(parser_args["party_map_path"]))
    if not map_path.is_absolute():
        map_path = root / map_path
    party_map = load_party_map(map_path, office, state_lookup)
    include_primaries = bool(parser_args.get("include_primaries", False))
    house_format = str(parser_args.get("house_race_id_format", "hyphenated"))
    require_nominees = bool(parser_args.get("require_nominees", False)) and not legacy
    dedupe = bool(parser_args.get("dedupe_versions", True)) and not legacy
    race_parties = {
        str(race): [str(p).upper() for p in value]
        for race, value in dict(parser_args.get("race_parties") or {}).items()
    }
    same_party_races = {str(race) for race in parser_args.get("same_party_races") or []}

    accounting = VoteHubAccounting(polls_in_payload=len(payload))
    candidates: list[dict[str, object]] = []
    for poll in payload:
        if not isinstance(poll, dict):
            accounting.mark(poll, "dropped_malformed")
            continue
        if str(poll.get("poll_type") or "").strip() != poll_type:
            accounting.mark(poll, "skipped_other_poll_type")
            continue
        subject = str(poll.get("subject") or "").strip()
        if not subject:
            accounting.mark(poll, "dropped_malformed")
            continue
        if not include_primaries and is_primary_subject(subject):
            accounting.mark(poll, "skipped_primary")
            continue
        race_id = votehub_race_id(
            subject=subject,
            seat_name=poll.get("seat_name"),
            office=office,
            cycle=cycle,
            house_format=house_format,
            state_lookup=state_lookup,
            state_abbreviations=state_abbreviations,
        )
        if race_id is None:
            accounting.mark(poll, "skipped_unknown_race")
            continue
        if race_id in same_party_races:
            accounting.mark(poll, "skipped_same_party_race")
            continue
        answers = poll.get("answers")
        if not isinstance(answers, list) or not answers:
            accounting.mark(poll, "dropped_malformed")
            continue
        modeled = race_parties.get(race_id, parties)
        jurisdiction = _race_jurisdiction(race_id, office)
        parsed = _parse_answers(
            answers, jurisdiction, party_map, modeled, legacy=legacy, accounting=accounting
        )
        if isinstance(parsed, str):
            accounting.mark(poll, parsed)
            continue
        party_pct, party_counts, non_nominees, has_generic, n_answers = parsed
        if not legacy:
            if any(count > 1 for count in party_counts.values()):
                accounting.mark(poll, "dropped_same_party")
                continue
            if any(party not in party_pct for party in modeled):
                accounting.mark(poll, "dropped_missing_side")
                continue
            if require_nominees and non_nominees:
                accounting.mark(poll, "dropped_non_nominee")
                accounting.non_nominee_names.update(non_nominees)
                continue
        if not party_pct:
            accounting.mark(poll, "dropped_malformed")
            continue
        candidates.append(
            {
                "poll": poll,
                "race_id": race_id,
                "party_pct": party_pct,
                "generic": has_generic,
                "n_answers": n_answers,
            }
        )
    if dedupe:
        candidates = _dedupe_versions(candidates, accounting)
    rows: list[dict[str, object]] = []
    for item in candidates:
        rows.extend(_poll_rows(item, cycle, office))
        accounting.mark(item["poll"], "polls_used")
        accounting.used_by_race[str(item["race_id"])] += 1
        if len(item["party_pct"]) < 2:
            accounting.one_sided_polls_used += 1
    if not rows:
        return pl.DataFrame(schema=POLL_SCHEMA), accounting
    frame = pl.DataFrame(rows, schema=POLL_SCHEMA)
    frame = frame.unique(
        subset=["poll_id", "race_id", "option_id"], keep="last", maintain_order=True
    )
    return frame.sort(["race_id", "option_id", "end_date", "poll_id"]), accounting


def _parse_answers(
    answers: list[object],
    jurisdiction: str,
    party_map: PartyMap,
    modeled: list[str],
    *,
    legacy: bool,
    accounting: VoteHubAccounting,
) -> tuple[dict[str, float], Counter[str], list[str], bool, int] | str:
    """Map answers to modeled parties, or return the drop reason for the whole poll."""
    party_pct: dict[str, float] = {}
    party_counts: Counter[str] = Counter()
    non_nominees: list[str] = []
    has_generic = False
    for answer in answers:
        if not isinstance(answer, dict):
            return "dropped_malformed"
        choice = str(answer.get("choice") or "").strip()
        pct_raw = answer.get("pct")
        try:
            pct = float(pct_raw) if pct_raw is not None else None
        except (TypeError, ValueError):
            pct = None
        if choice == "" or pct is None:
            return "dropped_malformed"
        entry = party_map.lookup(jurisdiction, choice)
        if entry is None or entry.party not in modeled:
            if legacy:
                if entry is None:
                    accounting.unmapped_names[f"{jurisdiction}: {choice}"] += 1
                return "dropped_unmapped"
            if entry is None:
                accounting.unmapped_names[f"{jurisdiction}: {choice}"] += 1
            continue  # OTHER: not added to any modeled party
        party_counts[entry.party] += 1
        party_pct[entry.party] = party_pct.get(entry.party, 0.0) + pct
        if entry.status == "generic":
            has_generic = True
        elif entry.status not in NOMINEE_STATUSES:
            non_nominees.append(f"{jurisdiction}: {choice}")
    return party_pct, party_counts, non_nominees, has_generic, len(answers)


def _version_rank(item: dict[str, object]) -> tuple[int, int, int, str]:
    poll = item["poll"]
    population = str(poll.get("population") or "a").strip().lower()
    return (
        POPULATION_RANK.get(population, 3),
        -int(item["n_answers"]),
        int(bool(item["generic"])),
        str(poll.get("id") or ""),
    )


def _dedupe_versions(
    candidates: list[dict[str, object]], accounting: VoteHubAccounting
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    order: list[tuple[str, str, str, str]] = []
    for item in candidates:
        poll = item["poll"]
        key = (
            str(item["race_id"]),
            str(poll.get("pollster") or "").strip().lower(),
            str(poll.get("start_date") or ""),
            str(poll.get("end_date") or ""),
        )
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(item)
    kept: list[dict[str, object]] = []
    for key in order:
        versions = sorted(groups[key], key=_version_rank)
        kept.append(versions[0])
        for duplicate in versions[1:]:
            accounting.mark(duplicate["poll"], "dropped_duplicate")
    return kept


def _poll_rows(item: dict[str, object], cycle: int, office: str) -> list[dict[str, object]]:
    poll = item["poll"]
    race_id = str(item["race_id"])
    poll_id_base = str(poll.get("id") or "").strip() or "unknown"
    population = str(poll.get("population") or "a").strip().lower() or "a"
    if population == "v":
        population = "lv"
    partisan = poll.get("partisan")
    sponsor_party = str(partisan).upper() if partisan not in (None, "", False) else ""
    if poll.get("internal") is True:
        sponsor_class = "internal"
    elif sponsor_party:
        sponsor_class = "partisan"
    else:
        sponsor_class = "nonpartisan"
    sample_size = poll.get("sample_size")
    try:
        sample_size_f = float(sample_size) if sample_size is not None else None
    except (TypeError, ValueError):
        sample_size_f = None
    start_date = str(poll.get("start_date") or "").strip() or None
    end_date = str(poll.get("end_date") or "").strip() or None
    pollster = str(poll.get("pollster") or "unknown").strip() or "unknown"
    rows = []
    for party, pct in sorted(dict(item["party_pct"]).items()):
        rows.append(
            {
                "poll_id": f"votehub-{poll_id_base}-{party}",
                "cycle": cycle,
                "office_type": office,
                "race_id": race_id,
                "pollster": pollster,
                "start_date": start_date,
                "end_date": end_date,
                "population": population,
                "sample_size": sample_size_f,
                "sponsor_class": sponsor_class,
                "sponsor_party": sponsor_party or None,
                "methodology": "mixed",
                "option_id": f"{race_id}-{party[:1]}",
                "pct": pct,
            }
        )
    return rows
