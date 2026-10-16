#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a single dataset, without regressions, improvements, or a baseline."""

import argparse
import json
from pathlib import Path

from ci.compile_time.render_pr_comment import md_code_span, md_escape, render_event_name
from ci.compile_time.summarize_events import primary_template_name

COMMENT_LIMIT = 60_000
ROW_LIMIT = 10_000
TABLE_HEADER = (
    "| Rank | Total (s) | Mean / event (s) | Events | Root TUs | Event |\n"
    "| ---: | ---: | ---: | ---: | ---: | --- |"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render_row(row: dict, artifacts_url: str) -> str:
    event = render_event_name(row)
    # Keep every ranked item. Specializations can exceed GitHub's entire
    # comment limit on their own; their full, unchanged names remain in CSV/JSON.
    if len(event.encode()) > ROW_LIMIT:
        name = primary_template_name(row.get("event_key", ""))
        event = f"{md_code_span(name)} ([complete event name in report CSV]({artifacts_url}))"
        if len(event.encode()) > ROW_LIMIT:
            event = (
                f"[Event {row['rank']}: complete name in report CSV]({artifacts_url})"
            )
    event = event.replace("|", "\\|")
    return "| {rank} | `{total}` | `{mean}` | {events} | {tus} | {event} |".format(
        rank=md_escape(row["rank"]),
        total=md_escape(row["selected_total_s"]),
        mean=md_escape(row["selected_avg_per_event_s"]),
        events=md_escape(row["event_count"]),
        tus=md_escape(row["root_tu_count"]),
        event=event,
    )


def render_reports(
    dataset: Path, output: Path, traces_url: str, artifacts_url: str
) -> dict:
    config = load_json(dataset / "config.json")
    metadata = load_json(dataset / "dataset.json")
    summary = load_json(dataset / "event_reports" / "summary.json")
    output.mkdir(parents=True, exist_ok=True)
    title = f"## ⏱️ {md_escape(config['name'])}"
    overview = "\n".join(
        [
            title,
            "",
            f"CCCL: {md_code_span(metadata.get('cccl_sha') or 'unavailable')}  ",
            f"Dataset: **{md_escape(metadata.get('status', 'unavailable'))}**, "
            f"{metadata.get('trace_count', 0)} original trace file(s).  ",
            f"**[Original traces and provenance]({traces_url or artifacts_url})** · "
            f"[Full CSV/JSON reports]({artifacts_url})",
            "",
            "Requested artifact retention: **90 days**, subject to repository/organization policy.",
            "",
        ]
    )
    if not summary:
        overview += "\nReport generation did not complete. Any collected original traces are saved independently.\n"
    (output / "overview.md").write_text(overview, encoding="utf-8")
    sections = []
    section_limit = COMMENT_LIMIT - len(overview.encode()) - 2
    slices = {s["id"]: s for s in summary.get("slices", [])}
    for spec in config["slices"]["slices"]:
        slice_id = spec["id"]
        data = slices.get(slice_id, {})
        rows = data.get("reports", {}).get("current", {}).get("rows", [])
        heading = f"### {md_escape(spec['title'])}"
        subtitle = f"Top {spec['top']} · {md_code_span(spec['timing'])} · ranked by total time."
        prefix = f"{heading}\n\n{subtitle}\n\n"
        if data.get("warnings"):
            prefix += "\n".join(f"- {md_escape(w)}" for w in data["warnings"]) + "\n\n"
        table_rows = [render_row(row, artifacts_url) for row in rows]
        if not rows:
            table_rows = ["No matching events." if summary else "Report unavailable."]
        # Each summary is written by a separate workflow step (<1 MiB per step).
        full = prefix + TABLE_HEADER + "\n" + "\n".join(table_rows) + "\n"
        (output / f"summary-{slice_id}.md").write_text(full, encoding="utf-8")
        page_prefix = (
            prefix
            + f"[Original traces]({traces_url or artifacts_url}) · [Full report]({artifacts_url})\n\n"
            + TABLE_HEADER
            + "\n"
        )
        page_rows = []
        pages = []
        for row in table_rows:
            candidate = page_prefix + "\n".join([*page_rows, row]) + "\n"
            if len(candidate.encode()) > section_limit and page_rows:
                pages.append(page_prefix + "\n".join(page_rows) + "\n")
                page_rows = []
            page_rows.append(row)
        pages.append(page_prefix + "\n".join(page_rows) + "\n")
        sections.extend(pages)
    # Prefer one comment per configuration; split only when the tables need it.
    comments = []
    body = overview
    for section in sections:
        if len((body + "\n" + section).encode()) > COMMENT_LIMIT:
            comments.append(body)
            body = overview
        body += "\n" + section
    comments.append(body)
    index = {"include": []}
    for part, body in enumerate(comments, 1):
        filename = f"report-{part}.md"
        assert len(body.encode()) <= COMMENT_LIMIT
        (output / filename).write_text(body, encoding="utf-8")
        index["include"].append(
            {"header": f"third-party-traces-{config['id']}-{part}", "file": filename}
        )
    (output / "index.json").write_text(json.dumps(index) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces-url", default="")
    parser.add_argument("--artifacts-url", required=True)
    args = parser.parse_args()
    render_reports(args.dataset, args.output, args.traces_url, args.artifacts_url)


if __name__ == "__main__":
    main()
