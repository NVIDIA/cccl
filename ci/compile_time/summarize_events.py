#!/usr/bin/env python3

import argparse
import csv
import heapq
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class FilterSpec:
    label: str
    description: str
    matches: Callable[["TraceEvent"], bool]
    default_exclusive_scope: str = "all"


@dataclass(frozen=True)
class ReportConfig:
    spec: FilterSpec
    timing: str
    exclusive_scope: str
    sort_by: str
    top_n: int
    tag: str | None
    threshold_us: float = 0.0


@dataclass
class TraceEvent:
    name: str
    detail: str
    start_us: int
    end_us: int
    pid: int
    tid: int
    root_tu: str
    synthetic: bool = False
    children: list["TraceEvent"] = field(default_factory=list)

    @property
    def inclusive_us(self) -> int:
        return self.end_us - self.start_us

    def key(self, repo_root: Path) -> str:
        if self.detail:
            return normalize_detail(self.detail, repo_root)
        return self.name


@dataclass
class EventStats:
    event_name: str
    event_key: str
    event_count: int = 0
    total_inclusive_us: int = 0
    total_exclusive_us: int = 0
    max_inclusive_us: int = 0
    max_exclusive_us: int = 0
    trace_paths: set[str] = field(default_factory=set)
    root_tus: set[str] = field(default_factory=set)


@dataclass
class ComparisonStats:
    event_name: str
    event_key: str
    baseline: EventStats
    current: EventStats
    matched_trace_paths: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ComparisonRow:
    stats: ComparisonStats
    baseline_metric_us: float
    current_metric_us: float
    magnitude_us: float

    @property
    def delta_us(self) -> float:
        return self.current_metric_us - self.baseline_metric_us


@dataclass(frozen=True)
class ComparisonSide:
    name: str
    repo_root: Path
    events: list[TraceEvent]
    report_ids: set[tuple[str, str]]
    child_ids: set[tuple[str, str]]


@dataclass(frozen=True)
class ComparisonInput:
    name: str
    trace_paths: dict[Path, Path]
    repo_root: Path


@dataclass(frozen=True)
class ReportSide:
    name: str
    trace_paths: list[Path]
    repo_root: Path
    output_dir: Path


def merged_interval_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0

    sorted_intervals = sorted(intervals)
    total = 0
    merged_start, merged_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start > merged_end:
            total += merged_end - merged_start
            merged_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)

    total += merged_end - merged_start
    return total


def generated_tu_input(tu: str) -> str:
    marker = "/headers/"
    if marker not in tu:
        return tu

    rel = tu.split(marker, 1)[1]
    parts = rel.split("/", 1)
    if len(parts) != 2:
        return tu

    tu_input = parts[1]
    for suffix in (".cu", ".cpp", ".cxx", ".cc", ".c"):
        if tu_input.endswith(suffix):
            return tu_input[: -len(suffix)]

    return tu_input


def normalize_detail(detail: str, repo_root: Path) -> str:
    detail_path = Path(detail)
    if detail_path.is_absolute():
        try:
            detail = detail_path.resolve(strict=False).relative_to(repo_root).as_posix()
        except ValueError:
            pass

    return detail


def normalize_project_file(detail: str, repo_root: Path) -> str | None:
    detail_path = Path(detail)
    if not detail_path.is_absolute():
        return None

    try:
        detail = detail_path.resolve(strict=False).relative_to(repo_root).as_posix()
    except ValueError:
        return None

    if detail.startswith("build/"):
        return None
    return detail


def general_event_identity(event: TraceEvent, repo_root: Path) -> tuple[str, str]:
    return (event.name, event.key(repo_root))


def report_event_identity(
    event: TraceEvent, spec: FilterSpec, repo_root: Path
) -> tuple[str, str] | None:
    if not spec.matches(event):
        return None

    if spec.label == "file-processing":
        event_key = normalize_project_file(event.detail, repo_root)
        if event_key is None:
            return None
    else:
        event_key = event.key(repo_root)

    return (event.name, event_key)


def filter_event_identity(
    event: TraceEvent, spec: FilterSpec, repo_root: Path
) -> tuple[str, str] | None:
    if not spec.matches(event):
        return None
    return general_event_identity(event, repo_root)


def trace_root_tu(trace: dict, trace_path: Path) -> str:
    input_files = trace.get("otherData", {}).get("inputFiles", [])
    if input_files:
        return generated_tu_input(input_files[0])
    return trace_path.as_posix()


def iter_trace_paths(trace_dir: Path) -> list[Path]:
    return sorted(p for p in trace_dir.rglob("*.json") if p.is_file())


def iter_duration_events(trace_path: Path, repo_root: Path) -> list[TraceEvent]:
    with trace_path.open(encoding="utf-8") as f:
        trace = json.load(f)

    root_tu = normalize_detail(trace_root_tu(trace, trace_path), repo_root)
    events: list[TraceEvent] = []
    for event in trace.get("traceEvents", []):
        if event.get("ph") not in (None, "X"):
            continue
        if "ts" not in event or "dur" not in event:
            continue

        name = str(event.get("name", ""))
        if not name:
            continue

        args = event.get("args", {})
        detail = ""
        if isinstance(args, dict):
            detail = str(args.get("detail", "") or "")

        start_us = int(event["ts"])
        dur_us = int(event["dur"])
        events.append(
            TraceEvent(
                name=name,
                detail=detail,
                start_us=start_us,
                end_us=start_us + dur_us,
                pid=int(event.get("pid", 0)),
                tid=int(event.get("tid", 0)),
                root_tu=root_tu,
            )
        )

    if events:
        trace_start_us = min(event.start_us for event in events)
        trace_end_us = max(event.end_us for event in events)
        events.append(
            TraceEvent(
                name="Total Compilation Time",
                detail=root_tu,
                start_us=trace_start_us,
                end_us=trace_end_us,
                pid=-1,
                tid=-1,
                root_tu=root_tu,
                synthetic=True,
            )
        )

    return events


def link_child_events(events: list[TraceEvent]) -> None:
    grouped: dict[tuple[int, int], list[TraceEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.pid, event.tid)].append(event)

    for thread_events in grouped.values():
        stack: list[TraceEvent] = []
        for event in sorted(thread_events, key=lambda e: (e.start_us, -e.end_us)):
            # Example: A=[0,10], B=[10,20]. A ended before B starts, so it is
            # not B's parent.
            while stack and stack[-1].end_us <= event.start_us:
                stack.pop()
            # Example: A=[0,100], B=[50,150]. B overlaps A but is not fully
            # contained by A, so A is not B's parent.
            while stack and not (
                stack[-1].start_us <= event.start_us
                and event.end_us <= stack[-1].end_us
            ):
                stack.pop()

            # Example: A=[0,100], B=[0,100]. Identical-span events are not
            # treated as nested children of each other.
            if stack and (
                stack[-1].start_us != event.start_us or stack[-1].end_us != event.end_us
            ):
                stack[-1].children.append(event)

            stack.append(event)


def read_trace_events(trace_path: Path, repo_root: Path) -> list[TraceEvent]:
    events = iter_duration_events(trace_path, repo_root)
    link_child_events(events)
    return events


def event_name_filter(
    *, label: str, description: str, event_names: tuple[str, ...]
) -> FilterSpec:
    name_set = set(event_names)
    return FilterSpec(
        label=label,
        description=description,
        matches=lambda event: event.name in name_set,
    )


def any_event_filter() -> FilterSpec:
    return FilterSpec(
        label="all",
        description="all raw duration events",
        matches=lambda event: not event.synthetic,
    )


def regex_filter(pattern: str) -> FilterSpec:
    compiled = re.compile(pattern, re.IGNORECASE)
    return FilterSpec(
        label=f"regex-{slugify(pattern)}",
        description=f"event name or detail matches /{pattern}/i",
        matches=lambda event: (
            bool(compiled.search(event.name)) or bool(compiled.search(event.detail))
        )
        and not event.synthetic,
    )


def builtin_filters() -> dict[str, FilterSpec]:
    filters: dict[str, FilterSpec] = {}

    def add(spec: FilterSpec) -> None:
        filters[spec.label] = spec

    add(any_event_filter())
    add(
        FilterSpec(
            label="file-processing",
            description=(
                "PHF trace events; exclusive time subtracts nested PHF events "
                "to match the direct file-processing metric"
            ),
            matches=lambda event: event.name == "Processing Header File",
            default_exclusive_scope="same-filter",
        ),
    )
    add(
        event_name_filter(
            label="scanning-function-body",
            description="Scanning Function Body events",
            event_names=("Scanning Function Body",),
        ),
    )
    add(
        event_name_filter(
            label="template-instantiation",
            description="template class/function instantiation events",
            event_names=(
                "Instantiating Template Class",
                "Instantiating Template Function",
            ),
        ),
    )
    add(
        event_name_filter(
            label="template-class-instantiation",
            description="template class instantiation events",
            event_names=("Instantiating Template Class",),
        ),
    )
    add(
        event_name_filter(
            label="template-function-instantiation",
            description="template function instantiation events",
            event_names=("Instantiating Template Function",),
        ),
    )
    add(
        event_name_filter(
            label="pending-instantiations",
            description="pending template instantiation phase events",
            event_names=("Generating Needed Template Instantiations",),
        ),
    )
    add(
        event_name_filter(
            label="frontend",
            description="front-end phase events",
            event_names=(
                "Front End Cleanup",
                "CUDA C++ Front-End",
            ),
        ),
    )
    add(
        event_name_filter(
            label="host-compiler",
            description="host compiler preprocessing and compiling events",
            event_names=(
                "g++ (preprocessing 1)",
                "g++ (preprocessing 4)",
                "g++ (compiling)",
                "gcc (preprocessing 1)",
                "gcc (preprocessing 4)",
                "gcc (compiling)",
            ),
        ),
    )
    add(
        event_name_filter(
            label="code-generation",
            description="code generation events",
            event_names=(
                "Generating Function IR",
                "Generating NVVM IR",
                "NVVM CodeGen",
            ),
        ),
    )
    add(
        event_name_filter(
            label="optimizer",
            description="optimizer events",
            event_names=(
                "OptFunction",
                "NVVM Optimizer",
            ),
        ),
    )
    add(
        event_name_filter(
            label="total-compilation",
            description=(
                "synthetic per-trace wall-clock span from first to last timed "
                "trace event"
            ),
            event_names=("Total Compilation Time",),
        ),
    )

    return filters


def resolve_filter(filter_name: str) -> FilterSpec:
    filters = builtin_filters()
    normalized = filter_name.strip().lower()
    if normalized in filters:
        return filters[normalized]

    return regex_filter(filter_name)


def exclusive_child_events(
    event: TraceEvent, spec: FilterSpec, exclusive_scope: str
) -> list[TraceEvent]:
    if exclusive_scope == "all":
        return event.children
    if exclusive_scope == "same-filter":
        return [child for child in event.children if spec.matches(child)]
    raise ValueError(f"unknown exclusive scope: {exclusive_scope}")


def event_exclusive_us(
    event: TraceEvent, spec: FilterSpec, exclusive_scope: str
) -> int:
    child_intervals = [
        (child.start_us, child.end_us)
        for child in exclusive_child_events(event, spec, exclusive_scope)
    ]
    return max(0, event.inclusive_us - merged_interval_duration(child_intervals))


def collect_stats(
    trace_paths: list[Path],
    repo_root: Path,
    config: ReportConfig,
) -> dict[tuple[str, str], EventStats]:
    stats: dict[tuple[str, str], EventStats] = {}

    for trace_path in trace_paths:
        events = read_trace_events(trace_path, repo_root)
        collect_trace_stats(
            stats,
            events,
            trace_path.as_posix(),
            repo_root,
            report_config=config,
            exclusive_config=config,
        )

    return stats


def collect_trace_stats(
    stats: dict[tuple[str, str], EventStats],
    events: list[TraceEvent],
    trace_path: str,
    repo_root: Path,
    *,
    report_config: ReportConfig,
    exclusive_config: ReportConfig,
) -> None:
    for event in events:
        identity = report_event_identity(event, report_config.spec, repo_root)
        if identity is None:
            continue
        add_event_stats(
            stats,
            identity,
            event.inclusive_us,
            event_exclusive_us(
                event,
                exclusive_config.spec,
                exclusive_config.exclusive_scope,
            ),
            trace_path,
            event.root_tu,
        )


def add_event_stats(
    stats: dict[tuple[str, str], EventStats],
    identity: tuple[str, str],
    inclusive_us: int,
    exclusive_us: int,
    trace_path: str,
    root_tu: str,
) -> None:
    event_name, event_key = identity
    event_stats = stats.setdefault(
        identity, EventStats(event_name=event_name, event_key=event_key)
    )
    merge_event_stats(
        event_stats,
        EventStats(
            event_name=event_name,
            event_key=event_key,
            event_count=1,
            total_inclusive_us=inclusive_us,
            total_exclusive_us=exclusive_us,
            max_inclusive_us=inclusive_us,
            max_exclusive_us=exclusive_us,
            trace_paths={trace_path},
            root_tus={root_tu},
        ),
    )


def selected_total_us(stats: EventStats, timing: str) -> int:
    if timing == "inclusive":
        return stats.total_inclusive_us
    if timing == "exclusive":
        return stats.total_exclusive_us
    raise ValueError(f"unknown timing: {timing}")


def average_us(total_us: int, count: int) -> float:
    if count == 0:
        return 0.0
    return total_us / count


def selected_avg_us(stats: EventStats, timing: str) -> float:
    return average_us(selected_total_us(stats, timing), stats.event_count)


def selected_avg_per_root_tu_us(stats: EventStats, timing: str) -> float:
    return average_us(selected_total_us(stats, timing), len(stats.root_tus))


def selected_max_us(stats: EventStats, timing: str) -> int:
    if timing == "inclusive":
        return stats.max_inclusive_us
    if timing == "exclusive":
        return stats.max_exclusive_us
    raise ValueError(f"unknown timing: {timing}")


def selected_metric_us(stats: EventStats, timing: str, sort_by: str) -> float:
    if sort_by == "total":
        return float(selected_total_us(stats, timing))
    if sort_by == "avg":
        return selected_avg_us(stats, timing)
    if sort_by == "avg-root-tu":
        return selected_avg_per_root_tu_us(stats, timing)
    if sort_by == "max":
        return float(selected_max_us(stats, timing))
    raise ValueError(f"unknown sort: {sort_by}")


def trace_paths_by_relative_root(trace_dir: Path) -> dict[Path, Path]:
    return {path.relative_to(trace_dir): path for path in iter_trace_paths(trace_dir)}


def comparison_input(name: str, trace_dir: Path, repo_root: Path) -> ComparisonInput:
    return ComparisonInput(
        name=name,
        trace_paths=trace_paths_by_relative_root(trace_dir),
        repo_root=repo_root,
    )


def comparable_child_identities(
    events: list[TraceEvent],
    repo_root: Path,
    config: ReportConfig,
) -> set[tuple[str, str]]:
    if config.exclusive_scope == "all":
        return {general_event_identity(event, repo_root) for event in events}
    if config.exclusive_scope == "same-filter":
        return {
            identity
            for event in events
            if (identity := filter_event_identity(event, config.spec, repo_root))
            is not None
        }
    raise ValueError(f"unknown exclusive scope: {config.exclusive_scope}")


def comparable_report_filter(
    spec: FilterSpec,
    repo_root: Path,
    comparable_report_ids: set[tuple[str, str]],
) -> FilterSpec:
    def matches(event: TraceEvent) -> bool:
        identity = report_event_identity(event, spec, repo_root)
        return identity is not None and identity in comparable_report_ids

    return replace(spec, matches=matches)


def comparable_child_filter(
    spec: FilterSpec,
    repo_root: Path,
    exclusive_scope: str,
    comparable_child_ids: set[tuple[str, str]],
) -> FilterSpec:
    def child_identity(event: TraceEvent) -> tuple[str, str] | None:
        if exclusive_scope == "all":
            return general_event_identity(event, repo_root)
        if exclusive_scope == "same-filter":
            return filter_event_identity(event, spec, repo_root)
        raise ValueError(f"unknown exclusive scope: {exclusive_scope}")

    def matches(event: TraceEvent) -> bool:
        identity = child_identity(event)
        return identity is not None and identity in comparable_child_ids

    return replace(spec, matches=matches)


def merge_event_stats(target: EventStats, source: EventStats) -> None:
    target.event_count += source.event_count
    target.total_inclusive_us += source.total_inclusive_us
    target.total_exclusive_us += source.total_exclusive_us
    target.max_inclusive_us = max(target.max_inclusive_us, source.max_inclusive_us)
    target.max_exclusive_us = max(target.max_exclusive_us, source.max_exclusive_us)
    target.trace_paths.update(source.trace_paths)
    target.root_tus.update(source.root_tus)


def merge_comparison_side_stats(
    comparison_stats: dict[tuple[str, str], ComparisonStats],
    side_name: str,
    side_stats: dict[tuple[str, str], EventStats],
) -> None:
    for identity, source_stats in side_stats.items():
        event_name, event_key = identity
        comparison = comparison_stats.setdefault(
            identity,
            ComparisonStats(
                event_name=event_name,
                event_key=event_key,
                baseline=EventStats(event_name, event_key),
                current=EventStats(event_name, event_key),
            ),
        )
        target_stats = getattr(comparison, side_name)
        merge_event_stats(target_stats, source_stats)
        comparison.matched_trace_paths.update(source_stats.trace_paths)


def read_comparison_side(
    name: str,
    trace_path: Path,
    repo_root: Path,
    config: ReportConfig,
) -> ComparisonSide:
    events = read_trace_events(trace_path, repo_root)
    report_ids = {
        identity
        for event in events
        if (identity := report_event_identity(event, config.spec, repo_root))
        is not None
    }
    child_ids = comparable_child_identities(events, repo_root, config)
    return ComparisonSide(
        name=name,
        repo_root=repo_root,
        events=events,
        report_ids=report_ids,
        child_ids=child_ids,
    )


def collect_comparison_stats(
    baseline_trace_dir: Path,
    current_trace_dir: Path,
    baseline_repo_root: Path,
    current_repo_root: Path,
    config: ReportConfig,
) -> tuple[dict[tuple[str, str], ComparisonStats], int]:
    comparison_inputs = (
        comparison_input("baseline", baseline_trace_dir, baseline_repo_root),
        comparison_input("current", current_trace_dir, current_repo_root),
    )
    matched_rel_paths = sorted(
        set.intersection(
            *(
                set(comparison_input.trace_paths)
                for comparison_input in comparison_inputs
            )
        )
    )
    comparison_stats: dict[tuple[str, str], ComparisonStats] = {}

    for rel_path in matched_rel_paths:
        sides = tuple(
            read_comparison_side(
                comparison_input.name,
                comparison_input.trace_paths[rel_path],
                comparison_input.repo_root,
                config,
            )
            for comparison_input in comparison_inputs
        )

        comparable_report_ids = set.intersection(*(side.report_ids for side in sides))
        if not comparable_report_ids:
            continue

        comparable_child_ids = set.intersection(*(side.child_ids for side in sides))
        rel_path_str = rel_path.as_posix()

        for side in sides:
            report_config = replace(
                config,
                spec=comparable_report_filter(
                    config.spec, side.repo_root, comparable_report_ids
                ),
            )
            exclusive_config = replace(
                config,
                spec=comparable_child_filter(
                    config.spec,
                    side.repo_root,
                    config.exclusive_scope,
                    comparable_child_ids,
                ),
                exclusive_scope="same-filter",
            )
            side_stats: dict[tuple[str, str], EventStats] = {}
            collect_trace_stats(
                side_stats,
                side.events,
                rel_path_str,
                side.repo_root,
                report_config=report_config,
                exclusive_config=exclusive_config,
            )
            merge_comparison_side_stats(comparison_stats, side.name, side_stats)

    return comparison_stats, len(matched_rel_paths)


def sorted_rows(
    stats: dict[tuple[str, str], EventStats], config: ReportConfig
) -> list[EventStats]:
    if config.sort_by not in ("total", "avg", "avg-root-tu", "max"):
        raise ValueError(f"unknown sort: {config.sort_by}")

    # Python has heapq.nsmallest rather than a C++-style partial_sort. The
    # selected metric is negated here so "smallest" means "largest selected
    # time", while the string tie-breakers keep their natural ascending order.
    return heapq.nsmallest(
        config.top_n,
        stats.values(),
        key=lambda item: (
            -selected_metric_us(item, config.timing, config.sort_by),
            item.event_name,
            item.event_key,
        ),
    )


def seconds(us: float | int) -> str:
    return f"{us / 1_000_000.0:.6f}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "report"


def default_output_path(
    output_dir: Path,
    config: ReportConfig,
) -> Path:
    pieces = ["top", str(config.top_n), config.spec.label, config.timing]
    if config.timing == "exclusive":
        pieces.append(config.exclusive_scope)
    pieces.append(f"by-{config.sort_by}")
    if config.tag:
        pieces.append(slugify(config.tag))
    return output_dir / ("-".join(slugify(piece) for piece in pieces) + ".csv")


def comparison_output_path(
    output_dir: Path,
    config: ReportConfig,
    direction: str,
) -> Path:
    pieces = ["top", str(config.top_n), config.spec.label, config.timing]
    if config.timing == "exclusive":
        pieces.append(config.exclusive_scope)
    pieces.extend([f"by-{config.sort_by}", direction])
    if config.tag:
        pieces.append(slugify(config.tag))
    return output_dir / ("-".join(slugify(piece) for piece in pieces) + ".csv")


def event_stats_csv_row(rank: int, row: EventStats, timing: str) -> dict[str, object]:
    root_tu_count = len(row.root_tus)
    return {
        "rank": rank,
        "event_name": row.event_name,
        "event_key": row.event_key,
        "selected_total_s": seconds(selected_total_us(row, timing)),
        "selected_avg_per_event_s": seconds(selected_avg_us(row, timing)),
        "selected_avg_per_root_tu_s": seconds(selected_avg_per_root_tu_us(row, timing)),
        "selected_max_s": seconds(selected_max_us(row, timing)),
        "total_inclusive_s": seconds(row.total_inclusive_us),
        "avg_inclusive_per_event_s": seconds(
            average_us(row.total_inclusive_us, row.event_count)
        ),
        "avg_inclusive_per_root_tu_s": seconds(
            average_us(row.total_inclusive_us, root_tu_count)
        ),
        "max_inclusive_s": seconds(row.max_inclusive_us),
        "total_exclusive_s": seconds(row.total_exclusive_us),
        "avg_exclusive_per_event_s": seconds(
            average_us(row.total_exclusive_us, row.event_count)
        ),
        "avg_exclusive_per_root_tu_s": seconds(
            average_us(row.total_exclusive_us, root_tu_count)
        ),
        "max_exclusive_s": seconds(row.max_exclusive_us),
        "event_count": row.event_count,
        "trace_count": len(row.trace_paths),
        "root_tu_count": root_tu_count,
    }


def write_csv(
    output_csv: Path,
    rows: list[EventStats],
    timing: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "event_name",
                "event_key",
                "selected_total_s",
                "selected_avg_per_event_s",
                "selected_avg_per_root_tu_s",
                "selected_max_s",
                "total_inclusive_s",
                "avg_inclusive_per_event_s",
                "avg_inclusive_per_root_tu_s",
                "max_inclusive_s",
                "total_exclusive_s",
                "avg_exclusive_per_event_s",
                "avg_exclusive_per_root_tu_s",
                "max_exclusive_s",
                "event_count",
                "trace_count",
                "root_tu_count",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(event_stats_csv_row(rank, row, timing))


def write_report_csv(
    output_dir: Path,
    stats: dict[tuple[str, str], EventStats],
    config: ReportConfig,
) -> Path:
    output_csv = default_output_path(output_dir, config)
    write_csv(
        output_csv,
        sorted_rows(stats, config),
        config.timing,
    )
    return output_csv


def report_side(
    name: str,
    trace_dir: Path,
    repo_root: Path,
    output_dir: Path,
) -> ReportSide:
    trace_paths = iter_trace_paths(trace_dir)
    if not trace_paths:
        raise SystemExit(f"no JSON traces found under {trace_dir}")
    return ReportSide(
        name=name,
        trace_paths=trace_paths,
        repo_root=repo_root,
        output_dir=output_dir,
    )


def write_side_report(
    side: ReportSide,
    config: ReportConfig,
    filter_name: str,
) -> Path:
    stats = collect_stats(side.trace_paths, side.repo_root, config)
    if not stats:
        raise SystemExit(
            f"no {side.name} events matched filter '{filter_name}' "
            f"in {len(side.trace_paths)} trace(s)"
        )
    return write_report_csv(side.output_dir, stats, config)


def comparison_rows(
    stats: dict[tuple[str, str], ComparisonStats],
    config: ReportConfig,
    direction: str,
) -> list[ComparisonRow]:
    if direction == "worse":
        multiplier = 1
    elif direction == "better":
        multiplier = -1
    else:
        raise ValueError(f"unknown comparison direction: {direction}")

    rows: list[ComparisonRow] = []
    for comparison in stats.values():
        baseline_metric = selected_metric_us(
            comparison.baseline, config.timing, config.sort_by
        )
        current_metric = selected_metric_us(
            comparison.current, config.timing, config.sort_by
        )
        delta = current_metric - baseline_metric
        magnitude = multiplier * delta
        if magnitude <= config.threshold_us:
            continue
        rows.append(
            ComparisonRow(
                comparison,
                baseline_metric,
                current_metric,
                magnitude,
            )
        )

    # Python has heapq.nsmallest rather than a C++-style partial_sort. The
    # change magnitude is negated here so "smallest" means "largest requested
    # change", while the string tie-breakers keep their natural ascending order.
    return heapq.nsmallest(
        config.top_n,
        rows,
        key=lambda row: (-row.magnitude_us, row.stats.event_name, row.stats.event_key),
    )


def write_comparison_csv(
    output_csv: Path,
    rows: list[ComparisonRow],
    timing: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "event_name",
                "event_key",
                "baseline_selected_s",
                "current_selected_s",
                "delta_s",
                "magnitude_s",
                "baseline_total_inclusive_s",
                "current_total_inclusive_s",
                "baseline_total_exclusive_s",
                "current_total_exclusive_s",
                "baseline_event_count",
                "current_event_count",
                "matched_trace_count",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            stats = row.stats
            writer.writerow(
                {
                    "rank": rank,
                    "event_name": stats.event_name,
                    "event_key": stats.event_key,
                    "baseline_selected_s": seconds(row.baseline_metric_us),
                    "current_selected_s": seconds(row.current_metric_us),
                    "delta_s": seconds(row.delta_us),
                    "magnitude_s": seconds(row.magnitude_us),
                    "baseline_total_inclusive_s": seconds(
                        stats.baseline.total_inclusive_us
                    ),
                    "current_total_inclusive_s": seconds(
                        stats.current.total_inclusive_us
                    ),
                    "baseline_total_exclusive_s": seconds(
                        stats.baseline.total_exclusive_us
                    ),
                    "current_total_exclusive_s": seconds(
                        stats.current.total_exclusive_us
                    ),
                    "baseline_event_count": stats.baseline.event_count,
                    "current_event_count": stats.current.event_count,
                    "matched_trace_count": len(stats.matched_trace_paths),
                }
            )


def print_filters() -> None:
    for name, spec in sorted(builtin_filters().items()):
        print(f"{name:32} {spec.description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a top-N CSV report for events in NVCC --fdevice-time-trace JSON files."
        )
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        help="directory containing device-time-trace JSON files",
    )
    parser.add_argument(
        "-f",
        "--filter",
        default="file-processing",
        help=(
            "canonical event filter name or case-insensitive regex over event "
            "name/detail (default: file-processing); use --list-filters to see "
            "built-in filters"
        ),
    )
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument(
        "-i",
        "--inclusive",
        action="store_const",
        const="inclusive",
        dest="timing",
        help="rank by inclusive event time",
    )
    timing.add_argument(
        "-e",
        "--exclusive",
        action="store_const",
        const="exclusive",
        dest="timing",
        help="rank by exclusive event time",
    )
    parser.set_defaults(timing="inclusive")
    parser.add_argument("-n", "--top", type=int, default=15, help="number of rows")
    parser.add_argument(
        "--sort",
        choices=("total", "avg", "avg-root-tu", "max"),
        default="total",
        help=(
            "sort selected timing by total contribution, average event cost, "
            "average per root TU, or max event cost"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="output directory (default: <trace-dir>/event_reports)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help=(
            "optional baseline trace directory; writes baseline/current reports "
            "and worse/better comparison CSVs under --output-dir"
        ),
    )
    parser.add_argument(
        "--baseline-repo-root",
        type=Path,
        help=(
            "repository root that produced --baseline-dir traces (default: --repo-root)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "comparison-only minimum selected-metric change, in seconds, "
            "required for worse/better rows (default: 0)"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="exact output CSV path; overrides generated file name inside --output-dir",
    )
    parser.add_argument(
        "--exclusive-scope",
        choices=("auto", "all", "same-filter"),
        default="auto",
        help=(
            "exclusive timing scope; auto uses same-filter for file-processing "
            "and all nested events for other filters"
        ),
    )
    parser.add_argument(
        "--repo-root", default=Path(__file__).resolve().parents[2], type=Path
    )
    parser.add_argument(
        "--tag",
        help="optional suffix for the generated output filename",
    )
    parser.add_argument(
        "--list-filters",
        action="store_true",
        help="print built-in filters and exit",
    )
    args = parser.parse_args()

    if args.list_filters:
        print_filters()
        return

    if args.trace_dir is None:
        parser.error("trace_dir is required unless --list-filters is used")
    if args.top <= 0:
        parser.error("--top must be positive")
    if args.threshold < 0:
        parser.error("--threshold must be non-negative")
    if args.baseline_dir is None and args.threshold != 0:
        parser.error("--threshold can only be used together with --baseline-dir")
    if args.baseline_dir is not None and args.output_csv is not None:
        parser.error("--output-csv cannot be used together with --baseline-dir")

    trace_dir = args.trace_dir.resolve(strict=False)
    baseline_dir = (
        args.baseline_dir.resolve(strict=False) if args.baseline_dir else None
    )
    repo_root = args.repo_root.resolve(strict=False)
    baseline_repo_root = (
        args.baseline_repo_root.resolve(strict=False)
        if args.baseline_repo_root
        else repo_root
    )
    spec = resolve_filter(args.filter)
    exclusive_scope = (
        spec.default_exclusive_scope
        if args.exclusive_scope == "auto"
        else args.exclusive_scope
    )
    config = ReportConfig(
        spec=spec,
        timing=args.timing,
        exclusive_scope=exclusive_scope,
        sort_by=args.sort,
        top_n=args.top,
        tag=args.tag,
        threshold_us=args.threshold * 1_000_000.0,
    )
    output_dir = (
        args.output_dir.resolve(strict=False)
        if args.output_dir
        else trace_dir / "event_reports"
    )
    output_csv = (
        args.output_csv.resolve(strict=False)
        if args.output_csv
        else default_output_path(output_dir, config)
    )

    if baseline_dir is not None:
        comparison_output_dir = output_dir / "comparison"
        report_sides = (
            report_side(
                "baseline",
                baseline_dir,
                baseline_repo_root,
                output_dir / "baseline",
            ),
            report_side(
                "current",
                trace_dir,
                repo_root,
                output_dir / "current",
            ),
        )
        report_csvs = {
            side.name: write_side_report(side, config, args.filter)
            for side in report_sides
        }

        comparison_stats, matched_trace_count = collect_comparison_stats(
            baseline_dir,
            trace_dir,
            baseline_repo_root,
            repo_root,
            config,
        )

        wrote = []
        for direction in ("worse", "better"):
            comparison_csv = comparison_output_path(
                comparison_output_dir,
                config,
                direction,
            )
            rows = comparison_rows(comparison_stats, config, direction)
            write_comparison_csv(comparison_csv, rows, config.timing)
            wrote.append((comparison_csv, len(rows)))

        trace_counts = ", ".join(
            f"{len(side.trace_paths)} {side.name} trace(s)" for side in report_sides
        )
        print(
            "wrote baseline/current reports and comparison reports "
            f"from {trace_counts}, "
            f"{matched_trace_count} matched trace file(s):"
        )
        for side in report_sides:
            print(f"  {side.name}: {report_csvs[side.name]}")
        for path, row_count in wrote:
            print(f"  comparison ({row_count} row(s)): {path}")
        return

    side = report_side("current", trace_dir, repo_root, output_dir)
    stats = collect_stats(side.trace_paths, side.repo_root, config)
    if not stats:
        raise SystemExit(
            f"no events matched filter '{args.filter}' in {len(side.trace_paths)} trace(s)"
        )

    rows = sorted_rows(stats, config)
    write_csv(output_csv, rows, config.timing)
    print(
        f"wrote {len(rows)} row(s) from {len(side.trace_paths)} trace(s) to {output_csv}"
    )


if __name__ == "__main__":
    main()
