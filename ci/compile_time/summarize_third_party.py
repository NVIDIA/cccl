#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep bounded CCCL reports from one third-party build, without uploading traces."""

import argparse
import posixpath
import re
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath

from ci.compile_time import summarize_events as events

REPORT_ROOT = Path("/__cccl__")
CCCL_SCOPE = re.compile(r"^(?:::)?(?:cuda|thrust|cub|cccl)::")
TOP = 100


def cccl_header(detail: str, source_root: PurePosixPath) -> str | None:
    """Map fetched/installed benchmark headers to CCCL source-tree paths."""
    path = PurePosixPath(posixpath.normpath(detail))
    if not path.is_absolute():
        return None
    try:
        parts = path.relative_to(source_root).parts
    except ValueError:
        return None
    for index in range(len(parts) - 1, 0, -1):
        if parts[index - 1 : index + 1] == ("_deps", "cccl-src"):
            relative = parts[index + 1 :]
            if relative and "build" not in relative:
                return PurePosixPath(*relative).as_posix()
            return None
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("include", "cccl") and "cuda" in parts[:index]:
            relative = parts[index + 2 :]
            break
        if parts[index : index + 2] == ("cccl-install", "include"):
            relative = parts[index + 2 :]
            break
    else:
        return None
    if relative and relative[0] in ("cub", "thrust"):
        return PurePosixPath(relative[0], *relative).as_posix()
    if relative and relative[0] in ("cuda", "nv"):
        return PurePosixPath("libcudacxx", "include", *relative).as_posix()
    return None


def cccl_symbol(detail: str) -> bool:
    # Match the owning name, never a CCCL type appearing in another project's
    # template arguments, parameters, or return type.
    primary = detail.partition(" [")[0].strip()
    if primary.startswith("_ZN"):
        nested = primary[3:].lstrip("KOVR")
        length = re.match(r"\d+", nested)
        if length:
            start = length.end()
            owner = nested[start : start + int(length.group())]
            return owner in ("cuda", "thrust", "cub", "cccl")
        return False
    return bool(CCCL_SCOPE.match(events.symbol_name_prefix(primary).lstrip("*&")))


def primary_template(detail: str) -> str:
    # NVCC supplies its primary-template label before the bracketed specialization.
    primary, separator, _ = detail.partition(" [")
    return primary.strip() if separator else events.symbol_name_prefix(detail)


def report_configs() -> dict[str, events.ReportConfig]:
    specs = (
        (
            "total-compilation",
            "Whole-TU compilation (context: includes non-CCCL work)",
            "total-compilation",
            "inclusive",
        ),
        (
            "file-processing",
            "CCCL direct header processing",
            "file-processing",
            "exclusive",
        ),
        (
            "function-bodies",
            "CCCL function-body parsing",
            "scanning-function-body",
            "inclusive",
        ),
        (
            "template-instantiation",
            "CCCL template instantiations",
            "template-instantiation",
            "inclusive",
        ),
    )
    configs = {}
    for slice_id, title, filter_name, timing in specs:
        spec = events.resolve_filter(filter_name)
        if slice_id in ("function-bodies", "template-instantiation"):
            match = spec.matches
            spec = replace(
                spec,
                matches=lambda event, match=match: (
                    match(event) and cccl_symbol(event.detail)
                ),
            )
        configs[slice_id] = events.ReportConfig(
            slice_id=slice_id,
            title=title,
            spec=spec,
            timing=timing,
            exclusive_scope=spec.default_exclusive_scope,
            sort_by="total",
            top_n=TOP,
            tag=None,
        )
    configs["primary-template-instantiation"] = replace(
        configs["template-instantiation"],
        slice_id="primary-template-instantiation",
        title="CCCL template instantiations by primary template",
    )
    return configs


def summarize(
    trace_dir: Path,
    output: Path,
    source_root: PurePosixPath,
    project: str,
    phase: str,
    delete_traces: bool = False,
) -> int:
    paths = sorted(
        p for p in trace_dir.rglob("*.o.json") if p.is_file() and not p.is_symlink()
    )
    configs = report_configs()
    aggregates = {slice_id: {} for slice_id in configs}
    failed = []
    trace_count = 0
    # Individual third-party traces can be large; parse one at a time and reuse
    # its event tree for every slice rather than multiplying peak memory usage.
    for path in paths:
        relative_path = path.relative_to(trace_dir).as_posix()
        try:
            parsed = events.read_trace_events(path, REPORT_ROOT)
            if not parsed:
                raise ValueError("No duration events found")
            for event in parsed:
                if event.name == "Processing Header File":
                    header = cccl_header(event.detail, source_root)
                    # Keep non-CCCL children for correct exclusive-time subtraction,
                    # but prevent them from appearing in the header report.
                    event.detail = str(REPORT_ROOT / header) if header else ""
                tu = PurePosixPath(posixpath.normpath(event.root_tu))
                if tu.is_relative_to(source_root):
                    event.root_tu = tu.relative_to(source_root).as_posix()
                if event.synthetic:
                    event.detail = event.root_tu
            trace_stats = {}
            for slice_id, config in configs.items():
                if slice_id == "primary-template-instantiation":
                    continue
                stats = events.side_stats_for_events(
                    parsed, relative_path, REPORT_ROOT, config
                )
                trace_stats[slice_id] = stats
                if slice_id == "template-instantiation":
                    grouped = {}
                    for stat in stats.values():
                        key = primary_template(stat.event_key)
                        events.merge_side_stats(
                            grouped,
                            {
                                (stat.event_name, key): replace(
                                    stat,
                                    event_key=key,
                                    trace_paths=set(stat.trace_paths),
                                    root_tus=set(stat.root_tus),
                                )
                            },
                        )
                    trace_stats["primary-template-instantiation"] = grouped
            for slice_id, stats in trace_stats.items():
                events.merge_side_stats(aggregates[slice_id], stats)
            trace_count += 1
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            failed.append({"trace": relative_path, "error": str(error)})
            print(f"Cannot summarize {relative_path}: {error}", file=sys.stderr)
        if (trace_count + len(failed)) % 25 == 0:
            print(f"Summarized {trace_count}/{len(paths)} traces", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    warnings = []
    if not paths:
        warnings.append("No object-adjacent traces found")
    if not aggregates["file-processing"] and not aggregates["template-instantiation"]:
        warnings.append(
            "No CCCL headers or templates matched; check trace paths and filtering"
        )
    manifest = {
        "schema_version": 1,
        "mode": "single",
        "project": project,
        "phase": phase,
        "top": TOP,
        "trace_count": trace_count,
        "failed_traces": failed,
        "warnings": warnings,
        "slices": [],
    }
    for slice_id, config in configs.items():
        rows = events.sorted_rows(aggregates[slice_id], config)
        filename = f"{slice_id}.csv"
        events.write_csv(output / filename, rows, config.timing)
        manifest["slices"].append(
            {
                "id": slice_id,
                "title": config.title,
                "timing": config.timing,
                "csv": filename,
                "rows": [
                    events.event_stats_csv_row(rank, row, config.timing)
                    for rank, row in enumerate(rows, 1)
                ],
            }
        )
    events.write_json(output / "summary.json", manifest)
    if delete_traces:
        # After RAPIDS' OFF report is saved, prevent unchanged OFF traces from
        # being counted again as ON work, and release their local disk space.
        for path in paths:
            path.unlink()
    print(f"Saved top-{TOP} reports for {trace_count} traces in {output}", flush=True)
    return int(bool(failed or warnings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        required=True,
        help="container path containing the benchmark's CCCL checkouts/installations",
    )
    parser.add_argument(
        "--project", choices=("matx", "pytorch", "rapids"), required=True
    )
    parser.add_argument("--phase", default="build")
    parser.add_argument("--delete-traces", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        summarize(
            args.trace_dir,
            args.output,
            PurePosixPath(posixpath.normpath(args.source_root)),
            args.project,
            args.phase,
            args.delete_traces,
        )
    )


if __name__ == "__main__":
    main()
