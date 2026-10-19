"""Measure the 2026 national environment from the generic-ballot aggregators.

Fetches the section-tagged GenericBallotAgg table on Wikipedia's 2026 House
election page (six professional aggregators: Decision Desk HQ, FiftyPlusOne,
RealClearPolitics, Silver Bulletin, VoteHub, Race to the WH), extracts each
aggregator's margin, and writes the median to fixtures/measured_environment.json.

The panel generators read that file when present, so the 2026 environment is a
measured, refreshable input rather than a hand-set assumption. Rerun this
script before regenerating panels to pick up the latest polling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "measured_environment.json"
SOURCE_URL = (
    "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections?action=raw"
)
USER_AGENT = "civic-signal/0.1 (election forecasting research; +https://github.com/gustavorubim/civic-signal)"
SECTION_PATTERN = re.compile(
    r'<section begin="GenericBallotAgg"\s*/>(.*?)<section end="GenericBallotAgg"\s*/>',
    re.DOTALL,
)
MARGIN_PATTERN = re.compile(r"'''(Democrats|Republicans) \+([0-9.]+)%'''")


def measure(
    text: str | None = None, measured_at: str | None = None, source: str = SOURCE_URL
) -> dict[str, object]:
    """Median generic-ballot margin. ``text`` lets a frozen wikitext snapshot be measured."""
    if text is None:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
    section = SECTION_PATTERN.search(text)
    if section is None:
        raise RuntimeError("GenericBallotAgg section not found on the 2026 House page")
    rows = section.group(1).split("|-")
    margins: dict[str, float] = {}
    for row in rows:
        if "'''Average'''" in row:
            continue
        match = MARGIN_PATTERN.search(row)
        if match is None:
            continue
        name_match = re.search(r"\|\s*(?:\[\[)?([A-Za-z][A-Za-z0-9 .]+?)(?:\]\]|\||<ref)", row)
        label = name_match.group(1).strip() if name_match else f"aggregator_{len(margins)}"
        value = float(match.group(2))
        margins[label] = value if match.group(1) == "Democrats" else -value
    # Drop the table's own summary row so the median is over the aggregators.
    aggregators = {name: value for name, value in margins.items() if name.lower() != "average"}
    if not aggregators:
        raise RuntimeError("No aggregator margins parsed from the GenericBallotAgg table")
    measured = round(median(aggregators.values()), 1)
    return {
        "generic_ballot_margin_d": measured,
        "aggregators": aggregators,
        "aggregator_count": len(aggregators),
        "source": source,
        "measured_at": measured_at or datetime.now(UTC).isoformat(),
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "method": "median_of_public_aggregators",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="frozen wikitext snapshot instead of a live fetch")
    parser.add_argument("--measured-at", help="timestamp to record (snapshot fetch time)")
    args = parser.parse_args(argv)
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        payload = measure(text, args.measured_at, source=f"{SOURCE_URL} (snapshot {args.input})")
    else:
        payload = measure()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"measured generic ballot: D+{payload['generic_ballot_margin_d']} "
        f"({payload['aggregator_count']} aggregators) -> {OUTPUT}"
    )


if __name__ == "__main__":
    main()
