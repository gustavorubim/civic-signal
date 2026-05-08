from __future__ import annotations

import html
import json
from typing import Any

import polars as pl


class DiagnosticsReport:
    def render(
        self,
        run_id: str,
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        source_manifest: pl.DataFrame,
        backtest_payload: dict[str, Any],
        reward_card: dict[str, Any] | None = None,
        plot_manifest: dict[str, list[dict[str, str]]] | None = None,
    ) -> str:
        tier_counts = race_catalog.group_by("tier").agg(pl.len().alias("count")).to_dicts()
        rewards = (reward_card or {}).get("rewards", {})
        reward_rows = "".join(
            f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value.get('passed')))}</td>"
            f"<td><code>{html.escape(json.dumps(value.get('metric'), sort_keys=True))}"
            "</code></td></tr>"
            for key, value in rewards.items()
        )
        plot_sections = self._plot_sections(plot_manifest or {})
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Diagnostics {html.escape(run_id)}</title></head>
<body>
<h1>Forecast Diagnostics: {html.escape(run_id)}</h1>
<h2>Coverage</h2>
<pre>{html.escape(json.dumps(tier_counts, indent=2, sort_keys=True))}</pre>
<p>Forecast rows: {race_forecasts.height}</p>
<p>Source rows: {source_manifest.height}</p>
<h2>Backtest Metrics</h2>
<pre>{html.escape(json.dumps(backtest_payload, indent=2, sort_keys=True))}</pre>
<h2>Reward Card</h2>
<table><thead><tr><th>Reward</th><th>Passed</th><th>Metric</th></tr></thead>
<tbody>{reward_rows}</tbody></table>
<h2>Plots</h2>
{plot_sections}
</body>
</html>
"""

    @staticmethod
    def _plot_sections(plot_manifest: dict[str, list[dict[str, str]]]) -> str:
        sections = []
        for category, entries in plot_manifest.items():
            figures = []
            for entry in entries:
                title = html.escape(entry["title"])
                path = html.escape(entry["path"])
                figures.append(
                    f'<figure><img src="{path}" alt="{title}" width="900">'
                    f"<figcaption>{title}</figcaption></figure>"
                )
            if figures:
                sections.append(f"<h3>{html.escape(category.title())}</h3>{''.join(figures)}")
        return "\n".join(sections) if sections else "<p>No plots generated.</p>"
