# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA = "cuda.compute.host_benchmark.v1"


class _Color(StrEnum):
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    NONE = ""


class _Emoji(StrEnum):
    YELLOW = "\U0001f7e1"
    BLUE = "\U0001f535"
    GREEN = "\U0001f7e2"
    RED = "\U0001f534"
    NONE = ""


def _colorize(label: str, color: _Color, emoji: _Emoji, no_color: bool) -> str:
    if no_color:
        if emoji:
            return f"{emoji} {label}"
        return label
    return f"{color}{label}{_Color.RESET}"


def _format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} us"
    return f"{ns / 1_000_000:.2f} ms"


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "inf"
    return f"{value * 100.0:.2f}%"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {schema!r}")
    return payload


def _result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result["name"]: result for result in payload["results"]}


def _minimum_noise(ref_noise: float | None, cmp_noise: float | None) -> float | None:
    if ref_noise is not None and cmp_noise is not None:
        return min(ref_noise, cmp_noise)
    if ref_noise is not None:
        return ref_noise
    return cmp_noise


def _status(
    ref_mean: float,
    cmp_mean: float,
    ref_noise: float | None,
    cmp_noise: float | None,
) -> tuple[str, float, float, float | None]:
    diff = cmp_mean - ref_mean
    frac_diff = diff / ref_mean
    min_noise = _minimum_noise(ref_noise, cmp_noise)

    if min_noise is None:
        return "????", diff, frac_diff, min_noise
    if abs(frac_diff) <= min_noise:
        return "SAME", diff, frac_diff, min_noise
    if diff < 0:
        return "FAST", diff, frac_diff, min_noise
    return "SLOW", diff, frac_diff, min_noise


def _format_status(status: str, *, no_color: bool) -> str:
    if status == "SAME":
        return _colorize(status, _Color.BLUE, _Emoji.BLUE, no_color)
    if status == "FAST":
        return _colorize(status, _Color.GREEN, _Emoji.GREEN, no_color)
    if status == "SLOW":
        return _colorize(status, _Color.RED, _Emoji.RED, no_color)
    return _colorize(status, _Color.YELLOW, _Emoji.YELLOW, no_color)


def _print_table(rows: list[list[str]]) -> None:
    headers = [
        "case",
        "ref mean",
        "ref noise",
        "cmp mean",
        "cmp noise",
        "diff",
        "%diff",
        "status",
    ]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    def print_row(row: list[str]) -> None:
        formatted = []
        for index, value in enumerate(row):
            if index in (0, 7):
                formatted.append(value.ljust(widths[index]))
            else:
                formatted.append(value.rjust(widths[index]))
        print("  ".join(formatted))

    print_row(headers)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print_row(row)


def compare(
    ref_payload: dict[str, Any],
    cmp_payload: dict[str, Any],
    *,
    no_color: bool,
    threshold: float,
) -> dict[str, int]:
    ref_results = _result_map(ref_payload)
    cmp_results = _result_map(cmp_payload)
    common_names = sorted(set(ref_results) & set(cmp_results))

    rows = []
    counts = {"total": 0, "same": 0, "unknown": 0, "fast": 0, "slow": 0}
    for name in common_names:
        ref_result = ref_results[name]
        cmp_result = cmp_results[name]
        ref_mean = float(ref_result["mean"])
        cmp_mean = float(cmp_result["mean"])
        ref_noise = ref_result["relative_noise"]
        cmp_noise = cmp_result["relative_noise"]
        status, diff, frac_diff, _ = _status(
            ref_mean,
            cmp_mean,
            ref_noise,
            cmp_noise,
        )

        counts["total"] += 1
        if status == "SAME":
            counts["same"] += 1
        elif status == "FAST":
            counts["fast"] += 1
        elif status == "SLOW":
            counts["slow"] += 1
        else:
            counts["unknown"] += 1

        if abs(frac_diff) < threshold:
            continue

        rows.append(
            [
                name,
                _format_ns(ref_mean),
                _format_percentage(ref_noise),
                _format_ns(cmp_mean),
                _format_percentage(cmp_noise),
                _format_ns(diff),
                _format_percentage(frac_diff),
                _format_status(status, no_color=no_color),
            ]
        )

    if rows:
        _print_table(rows)
    else:
        print("No matching benchmark cases exceeded the display threshold.")

    missing_in_cmp = sorted(set(ref_results) - set(cmp_results))
    missing_in_ref = sorted(set(cmp_results) - set(ref_results))
    if missing_in_cmp:
        print(f"\nMissing from compare: {', '.join(missing_in_cmp)}")
    if missing_in_ref:
        print(f"\nMissing from reference: {', '.join(missing_in_ref)}")

    print("\n# Summary\n")
    print(f"- Total Matches: {counts['total']}")
    print(f"  - Same    (diff <= min noise): {counts['same']}")
    print(f"  - Fast    (cmp faster):        {counts['fast']}")
    print(f"  - Slow    (cmp slower):        {counts['slow']}")
    print(f"  - Unknown (missing noise):     {counts['unknown']}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two cuda.compute host benchmark JSON outputs."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("compare", type=Path)
    parser.add_argument(
        "--threshold-diff",
        type=float,
        default=0.0,
        help="Only show rows where absolute relative diff is at least this value.",
    )
    parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Return nonzero if any case is classified FAST or SLOW.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Use emoji instead of ANSI color codes.",
    )
    args = parser.parse_args()

    ref_payload = _load(args.reference)
    cmp_payload = _load(args.compare)
    ref_benchmark = ref_payload["benchmark"]
    cmp_benchmark = cmp_payload["benchmark"]
    if ref_benchmark != cmp_benchmark:
        print(
            f"Benchmark types do not match: {ref_benchmark!r} vs {cmp_benchmark!r}",
            file=sys.stderr,
        )
        return 1

    print(f"# {ref_benchmark}\n")
    counts = compare(
        ref_payload,
        cmp_payload,
        no_color=args.no_color,
        threshold=args.threshold_diff,
    )
    if args.fail_on_change:
        return counts["fast"] + counts["slow"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
