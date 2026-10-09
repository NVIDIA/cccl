#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def md_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def md_code_span(value: object) -> str:
    text = str(value).replace("\n", " ")
    max_backtick_run = 0
    current_backtick_run = 0
    for char in text:
        if char == "`":
            current_backtick_run += 1
            max_backtick_run = max(max_backtick_run, current_backtick_run)
        else:
            current_backtick_run = 0
    delimiter = "`" * (max_backtick_run + 1)
    if text.startswith("`") or text.endswith("`"):
        text = f" {text} "
    return f"{delimiter}{text}{delimiter}"


def render_event_name(row: dict[str, Any]) -> str:
    event_name = row.get("event_name", "")
    event_key = row.get("event_key", "")
    if event_key:
        return f"{md_escape(event_name)}: {md_code_span(event_key)}"
    return md_escape(event_name)


def render_rows(rows: list[dict[str, Any]], *, direction: str) -> str:
    delta_heading = (
        "Regression impact" if direction == "worse" else "Improvement impact"
    )
    lines = [
        f"| Rank | {delta_heading} | Selected Δ | Baseline | Current | Event | Matched traces |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{impact}` | `{selected_delta}` | `{baseline}` | `{current}` | {event} | {traces} |".format(
                rank=md_escape(row.get("rank", "")),
                impact=md_escape(row.get("impact_magnitude_s", "")),
                selected_delta=md_escape(row.get("selected_delta_s", "")),
                baseline=md_escape(row.get("baseline_selected_s", "")),
                current=md_escape(row.get("current_selected_s", "")),
                event=render_event_name(row),
                traces=md_escape(row.get("matched_trace_count", "")),
            )
        )
    return "\n".join(lines)


def render_direction_details(
    slice_title: str,
    direction: str,
    rows: list[dict[str, Any]],
) -> str:
    if not rows:
        return ""
    label = "Regressions" if direction == "worse" else "Improvements"
    icon = "🔴" if direction == "worse" else "🟢"
    return "\n".join(
        [
            "<details>",
            f"<summary><strong>{icon} {md_escape(slice_title)} — {label}</strong></summary>",
            "",
            render_rows(rows, direction=direction),
            "",
            "</details>",
        ]
    )


def render_warning_details(slice_title: str, warnings: list[Any]) -> str:
    if not warnings:
        return ""
    lines = [
        "<details open>",
        f"<summary><strong>⚠️ {md_escape(slice_title)} — Warnings</strong></summary>",
        "",
    ]
    lines.extend(f"- {md_escape(warning)}" for warning in warnings)
    lines.extend(["", "</details>"])
    return "\n".join(lines)


def render_slice(slice_data: dict[str, Any], *, level: int = 3) -> str:
    comparison = slice_data.get("comparison", {})
    worse_rows = comparison.get("worse", {}).get("rows", [])
    better_rows = comparison.get("better", {}).get("rows", [])
    warnings = slice_data.get("warnings", [])
    child_sections = [
        rendered
        for child in slice_data.get("children", [])
        if (rendered := render_slice(child, level=level + 1))
    ]
    direct_sections = [
        section
        for section in (
            render_warning_details(slice_data.get("title", "Slice"), warnings),
            render_direction_details(
                slice_data.get("title", "Slice"), "worse", worse_rows
            ),
            render_direction_details(
                slice_data.get("title", "Slice"), "better", better_rows
            ),
        )
        if section
    ]
    if not direct_sections and not child_sections:
        return ""

    heading_prefix = "#" * min(level, 6)
    subtitle = (
        f"`-f {slice_data.get('filter', '')}` "
        f"`{slice_data.get('timing', '')}` "
        f"`--sort {slice_data.get('sort', '')}`"
    )
    lines = [
        f"{heading_prefix} {md_escape(slice_data.get('title', 'Slice'))}",
        "",
        subtitle,
        "",
    ]
    lines.extend(join_sections(direct_sections))
    if child_sections:
        lines.extend(["", *join_sections(child_sections)])
    return "\n".join(lines).strip()


def join_sections(sections: list[str]) -> list[str]:
    lines: list[str] = []
    for section in sections:
        if lines:
            lines.append("")
        lines.append(section)
    return lines


def count_rows(slice_data: dict[str, Any], direction: str) -> int:
    comparison = slice_data.get("comparison", {})
    total = len(comparison.get(direction, {}).get("rows", []))
    return total + sum(
        count_rows(child, direction) for child in slice_data.get("children", [])
    )


def count_warnings(slice_data: dict[str, Any]) -> int:
    return len(slice_data.get("warnings", [])) + sum(
        count_warnings(child) for child in slice_data.get("children", [])
    )


def render_comment(
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    artifacts_url: str,
) -> str:
    if summary.get("overall", {}).get("stability_filter"):
        return render_stability_comment(summary, config, artifacts_url=artifacts_url)
    config_id = str(config["id"])
    slices = summary.get("slices", [])
    sections = [
        section for slice_data in slices if (section := render_slice(slice_data))
    ]
    worse_count = sum(count_rows(slice_data, "worse") for slice_data in slices)
    better_count = sum(count_rows(slice_data, "better") for slice_data in slices)
    warning_count = sum(count_warnings(slice_data) for slice_data in slices)
    result = (
        f"**Result:** {worse_count} regression row(s), "
        f"{better_count} improvement row(s) above threshold."
    )
    if warning_count:
        result += f" {warning_count} warning(s)."

    lines = [
        f"<!-- cccl-compile-time-bench: {md_escape(config_id)} -->",
        f"## ⏱️ CCCL compile-time benchmark comparison: {md_escape(config.get('name', config_id))}",
        "",
        result,
        "",
        "| Run | Value |",
        "| --- | --- |",
        f"| Config | {md_code_span(config_id)} |",
        f"| Baseline | {md_code_span(config.get('baseline_ref', ''))} |",
        f"| Preset | {md_code_span(config.get('preset', ''))} |",
        f"| Targets | {md_code_span(', '.join(config.get('targets', [])))} |",
        f"| GPU / launch args | {md_code_span(config.get('gpu', ''))} / {md_code_span(config.get('launch_args', ''))} |",
        "",
        f"**Artifacts:** [reports and traces]({artifacts_url})",
        "",
    ]
    if sections:
        lines.extend(join_sections(sections))
    else:
        lines.append(
            "No compile-time benchmark changes exceeded the configured thresholds."
        )
    return "\n".join(lines).rstrip() + "\n"


def iter_slices(slices: list[dict[str, Any]]):
    for slice_data in slices:
        yield slice_data
        yield from iter_slices(slice_data.get("children", []))


def render_diagnostics(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Event | Raw Δ (s) | Beyond build drift (s) | TUs changing in this direction | Adjusted p-value | Consistency across TUs |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        consistency = {
            "consistent": "Reliable across TU contexts",
            "localized": "Context-dependent; not established",
            "insufficient-tus": "Too few TU contexts",
            "inconsistent": "Directional consistency not established",
            "aggregate-only": "Not assessed",
        }.get(row["stability"], row["stability"])
        lines.append(
            f"| {render_event_name(row)} | {md_escape(row['impact_delta_s'])} | "
            f"{md_escape(row['adjusted_delta_s'])} | "
            f"{row.get('same_direction_tus', 0)}/{row['matched_tu_count']} | "
            f"{row.get('consistency_adjusted_pvalue', 1.0):.3g} | "
            f"{md_escape(consistency)} |"
        )
    return "\n".join(lines)


def render_stability_comment(
    summary: dict[str, Any], config: dict[str, Any], *, artifacts_url: str
) -> str:
    overall = summary["overall"]
    relative = overall["relative_delta_pct"]
    relative_text = (
        f"{relative:+.2f}%" if relative is not None else "relative change unavailable"
    )
    slices = list(iter_slices(summary.get("slices", [])))
    warnings = [warning for data in slices for warning in data.get("warnings", [])]

    def ranked(direction: str, kind: str = "rows") -> list[dict[str, Any]]:
        rows = [
            row
            for data in slices
            if data.get("filter") != "total-compilation"
            for row in data.get("comparison", {}).get(direction, {}).get(kind, [])
        ]
        rows.sort(
            key=lambda row: (
                -abs(float(row["adjusted_delta_s"])),
                row["event_name"],
                row["event_key"],
            )
        )
        return rows

    worse = ranked("worse")[:5]
    better = ranked("better")[:3]
    common = sorted(
        ranked("worse", "common_headers") + ranked("better", "common_headers"),
        key=lambda row: -abs(float(row["adjusted_delta_s"])),
    )[:3]
    unconfirmed = sorted(
        ranked("worse", "unconfirmed") + ranked("better", "unconfirmed"),
        key=lambda row: -abs(float(row["adjusted_delta_s"])),
    )[:3]
    lines = [
        f"<!-- cccl-compile-time-bench: {md_escape(config['id'])} -->",
        f"## ⏱️ {md_escape(config.get('name', config['id']))}",
        "",
        f"**Total compilation:** {overall['baseline_s']} → {overall['current_s']} s "
        f"(**{float(overall['delta_s']):+.3f} s**, {relative_text}).",
        f"Matched corpus: {overall['matched_tu_count']} TUs, {overall['matched_trace_count']} traces. "
        "Times sum compiler trace durations; they are not elapsed build time.",
        "",
        f"Baseline: {md_code_span(config.get('baseline_ref', ''))}. "
        f"[Full CSV reports and raw traces]({artifacts_url}).",
        "",
        "Consistency tests look for more than 75% of TU contexts moving in the same direction, using exact binomial tests and report-wide Holm adjustment at 5%. "
        "They assume independent TU directions and cannot distinguish shared runner bias from code effects. Raw totals retain observed impact; diagnostic ranks subtract median build drift. Build-total confidence intervals are unavailable from one build pair.",
        "",
    ]
    if warnings:
        lines.append("**Warnings:**")
        lines.append("")
        lines.extend(f"- {md_escape(warning)}" for warning in dict.fromkeys(warnings))
        lines.append("")
    if worse:
        lines.extend(
            ["**Largest regression candidates**", "", render_diagnostics(worse), ""]
        )
    else:
        message = (
            "Regression candidates affecting common headers passed the tests and appear below."
            if any(float(row["adjusted_delta_s"]) > 0 for row in common)
            else "No diagnostic regression candidates passed the impact and directional consistency tests."
        )
        lines.extend(
            [
                message,
                "",
            ]
        )
    for title, rows in (
        ("Largest improvement candidates", better),
        ("Common headers (present in at least 80% of matched TUs)", common),
        ("Context-dependent or sparse changes", unconfirmed),
    ):
        if rows:
            lines.extend(
                [
                    "<details>",
                    f"<summary>{title}</summary>",
                    "",
                    render_diagnostics(rows),
                    "",
                    "</details>",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a GitHub PR comment from compile-time report JSON."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts-url", required=True)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    comment = render_comment(
        load_json(args.summary),
        load_json(args.config),
        artifacts_url=args.artifacts_url,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(comment, encoding="utf-8")
    else:
        print(comment, end="")


if __name__ == "__main__":
    main()
