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
