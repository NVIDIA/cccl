#!/usr/bin/env python3

import argparse
import csv
import heapq
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

DEFAULT_SCOPE_FILTER = r"(^|[^A-Za-z0-9_:])(?:::)?(?:cuda|thrust|cub|cccl)::"
SYMBOL_SCOPE_EVENT_NAMES = {
    "Scanning Function Body",
    "Instantiating Template Class",
    "Instantiating Template Function",
    "Generating Function IR",
    "OptFunction",
}
ITANIUM_CV_QUALIFIERS = frozenset("KOVR")


@dataclass(frozen=True)
class FilterSpec:
    label: str
    description: str
    matches: Callable[["TraceEvent"], bool]
    default_exclusive_scope: str = "all"


@dataclass(frozen=True)
class ReportConfig:
    slice_id: str
    title: str
    spec: FilterSpec
    timing: str
    exclusive_scope: str
    sort_by: str
    top_n: int
    tag: str | None
    threshold_us: float = 0.0
    scope_filter: re.Pattern[str] | None = None


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
    baseline_impact_us: float
    current_impact_us: float
    baseline_metric_us: float
    current_metric_us: float
    impact_magnitude_us: float

    @property
    def impact_delta_us(self) -> float:
        return self.current_impact_us - self.baseline_impact_us

    @property
    def selected_delta_us(self) -> float:
        return self.current_metric_us - self.baseline_metric_us

    @property
    def selected_magnitude_us(self) -> float:
        return abs(self.selected_delta_us)


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


@dataclass(frozen=True)
class SliceRequest:
    config: ReportConfig
    filter_name: str
    children: tuple["SliceRequest", ...] = ()


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


def strip_angle_arguments(symbol: str) -> str:
    stripped: list[str] = []
    depth = 0
    for char in symbol:
        if char == "<":
            depth += 1
            continue
        if char == ">" and depth:
            depth -= 1
            continue
        if depth == 0:
            stripped.append(char)
    return "".join(stripped)


def symbol_name_prefix(symbol: str) -> str:
    before_parameters = strip_angle_arguments(symbol).split("(", 1)[0].strip()
    if not before_parameters:
        return symbol
    if "::operator" in before_parameters:
        operator_scope = before_parameters.rfind("::operator")
        prefix_start = before_parameters.rfind(" ", 0, operator_scope)
        return before_parameters[prefix_start + 1 :]
    return before_parameters.rsplit(None, 1)[-1]


def itanium_nested_scope_candidates(symbol: str) -> list[str]:
    if not symbol.startswith("_Z"):
        return []

    candidates: list[str] = []
    for nested_marker in (i for i, char in enumerate(symbol) if char == "N"):
        index = nested_marker + 1
        while index < len(symbol) and symbol[index] in ITANIUM_CV_QUALIFIERS:
            index += 1

        scopes: list[str] = []
        while index < len(symbol) and symbol[index].isdigit():
            length_start = index
            while index < len(symbol) and symbol[index].isdigit():
                index += 1
            try:
                component_length = int(symbol[length_start:index])
            except ValueError:
                break

            component = symbol[index : index + component_length]
            if len(component) != component_length:
                break

            scopes.append(component)
            candidates.append("::".join(scopes) + "::")
            index += component_length

    return candidates


def symbol_scope_candidates(event: TraceEvent) -> list[str]:
    if event.name not in SYMBOL_SCOPE_EVENT_NAMES or not event.detail:
        return []

    candidates = [symbol_name_prefix(event.detail)]
    if " [" in event.detail:
        _, bracketed_symbol = event.detail.split(" [", 1)
        candidates.append(symbol_name_prefix(bracketed_symbol.rstrip("]")))

    for candidate in list(candidates):
        candidates.extend(itanium_nested_scope_candidates(candidate))

    return candidates


def matches_scope_filter(
    event: TraceEvent, scope_filter: re.Pattern[str] | None
) -> bool:
    if scope_filter is None or event.name not in SYMBOL_SCOPE_EVENT_NAMES:
        return True
    return any(
        scope_filter.search(candidate) for candidate in symbol_scope_candidates(event)
    )


def matches_report_config(event: TraceEvent, config: ReportConfig) -> bool:
    return config.spec.matches(event) and matches_scope_filter(
        event, config.scope_filter
    )


def report_event_identity(
    event: TraceEvent, config: ReportConfig, repo_root: Path
) -> tuple[str, str] | None:
    if not matches_report_config(event, config):
        return None

    if config.spec.label == "file-processing":
        event_key = normalize_project_file(event.detail, repo_root)
        if event_key is None:
            return None
    else:
        event_key = event.key(repo_root)

    return (event.name, event_key)


def filter_event_identity(
    event: TraceEvent, config: ReportConfig, repo_root: Path
) -> tuple[str, str] | None:
    if not matches_report_config(event, config):
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


def exclusive_child_events(event: TraceEvent, config: ReportConfig) -> list[TraceEvent]:
    if config.exclusive_scope == "all":
        return event.children
    if config.exclusive_scope == "same-filter":
        return [
            child for child in event.children if matches_report_config(child, config)
        ]
    raise ValueError(f"unknown exclusive scope: {config.exclusive_scope}")


def event_exclusive_us(event: TraceEvent, config: ReportConfig) -> int:
    child_intervals = [
        (child.start_us, child.end_us)
        for child in exclusive_child_events(event, config)
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
        identity = report_event_identity(event, report_config, repo_root)
        if identity is None:
            continue
        add_event_stats(
            stats,
            identity,
            event.inclusive_us,
            event_exclusive_us(event, exclusive_config),
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
            if (identity := filter_event_identity(event, config, repo_root)) is not None
        }
    raise ValueError(f"unknown exclusive scope: {config.exclusive_scope}")


def comparable_report_filter(
    config: ReportConfig,
    repo_root: Path,
    comparable_report_ids: set[tuple[str, str]],
) -> FilterSpec:
    def matches(event: TraceEvent) -> bool:
        identity = report_event_identity(event, config, repo_root)
        return identity is not None and identity in comparable_report_ids

    return replace(config.spec, matches=matches)


def comparable_child_filter(
    config: ReportConfig,
    repo_root: Path,
    comparable_child_ids: set[tuple[str, str]],
) -> FilterSpec:
    def child_identity(event: TraceEvent) -> tuple[str, str] | None:
        if config.exclusive_scope == "all":
            return general_event_identity(event, repo_root)
        if config.exclusive_scope == "same-filter":
            return filter_event_identity(event, config, repo_root)
        raise ValueError(f"unknown exclusive scope: {config.exclusive_scope}")

    def matches(event: TraceEvent) -> bool:
        identity = child_identity(event)
        return identity is not None and identity in comparable_child_ids

    return replace(config.spec, matches=matches)


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
        if (identity := report_event_identity(event, config, repo_root)) is not None
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
                    config, side.repo_root, comparable_report_ids
                ),
            )
            exclusive_config = replace(
                config,
                spec=comparable_child_filter(
                    config, side.repo_root, comparable_child_ids
                ),
                exclusive_scope="same-filter",
                scope_filter=(
                    config.scope_filter
                    if config.exclusive_scope == "same-filter"
                    else None
                ),
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


def comparison_row_dict(
    rank: int, row: ComparisonRow, timing: str
) -> dict[str, object]:
    stats = row.stats
    return {
        "rank": rank,
        "event_name": stats.event_name,
        "event_key": stats.event_key,
        "baseline_impact_s": seconds(row.baseline_impact_us),
        "current_impact_s": seconds(row.current_impact_us),
        "impact_delta_s": seconds(row.impact_delta_us),
        "impact_magnitude_s": seconds(row.impact_magnitude_us),
        "baseline_selected_s": seconds(row.baseline_metric_us),
        "current_selected_s": seconds(row.current_metric_us),
        "selected_delta_s": seconds(row.selected_delta_us),
        "selected_magnitude_s": seconds(row.selected_magnitude_us),
        "baseline_total_inclusive_s": seconds(stats.baseline.total_inclusive_us),
        "current_total_inclusive_s": seconds(stats.current.total_inclusive_us),
        "baseline_total_exclusive_s": seconds(stats.baseline.total_exclusive_us),
        "current_total_exclusive_s": seconds(stats.current.total_exclusive_us),
        "baseline_event_count": stats.baseline.event_count,
        "current_event_count": stats.current.event_count,
        "matched_trace_count": len(stats.matched_trace_paths),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


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
) -> tuple[Path, int]:
    stats = collect_stats(side.trace_paths, side.repo_root, config)
    rows = sorted_rows(stats, config) if stats else []
    output_csv = default_output_path(side.output_dir, config)
    write_csv(output_csv, rows, config.timing)
    return output_csv, len(rows)


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
        baseline_impact = float(selected_total_us(comparison.baseline, config.timing))
        current_impact = float(selected_total_us(comparison.current, config.timing))
        baseline_metric = selected_metric_us(
            comparison.baseline, config.timing, config.sort_by
        )
        current_metric = selected_metric_us(
            comparison.current, config.timing, config.sort_by
        )
        delta = current_impact - baseline_impact
        magnitude = multiplier * delta
        if magnitude <= config.threshold_us:
            continue
        rows.append(
            ComparisonRow(
                comparison,
                baseline_impact,
                current_impact,
                baseline_metric,
                current_metric,
                magnitude,
            )
        )

    # Python has heapq.nsmallest rather than a C++-style partial_sort. The
    # total-impact change is negated here so "smallest" means "largest
    # requested aggregate movement across matched traces", while the string
    # tie-breakers keep their natural ascending order.
    return heapq.nsmallest(
        config.top_n,
        rows,
        key=lambda row: (
            -row.impact_magnitude_us,
            row.stats.event_name,
            row.stats.event_key,
        ),
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
                "baseline_impact_s",
                "current_impact_s",
                "impact_delta_s",
                "impact_magnitude_s",
                "baseline_selected_s",
                "current_selected_s",
                "selected_delta_s",
                "selected_magnitude_s",
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
            writer.writerow(comparison_row_dict(rank, row, timing))


def print_filters() -> None:
    for name, spec in sorted(builtin_filters().items()):
        print(f"{name:32} {spec.description}")


def compile_scope_filter(
    pattern: str, parser: argparse.ArgumentParser
) -> re.Pattern[str] | None:
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as e:
        parser.error(f"invalid --scope-filter regex: {e}")


def report_config(
    *,
    slice_id: str,
    title: str,
    filter_name: str,
    timing: str,
    exclusive_scope: str,
    sort_by: str,
    top_n: int,
    tag: str | None,
    threshold_s: float,
    scope_filter: re.Pattern[str] | None,
) -> ReportConfig:
    spec = resolve_filter(filter_name)
    resolved_exclusive_scope = (
        spec.default_exclusive_scope if exclusive_scope == "auto" else exclusive_scope
    )
    return ReportConfig(
        slice_id=slice_id,
        title=title,
        spec=spec,
        timing=timing,
        exclusive_scope=resolved_exclusive_scope,
        sort_by=sort_by,
        top_n=top_n,
        tag=tag,
        threshold_us=threshold_s * 1_000_000.0,
        scope_filter=scope_filter,
    )


def single_slice_request(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> SliceRequest:
    config = report_config(
        slice_id=slugify(args.tag or args.filter),
        title=args.tag or resolve_filter(args.filter).description,
        filter_name=args.filter,
        timing=args.timing,
        exclusive_scope=args.exclusive_scope,
        sort_by=args.sort,
        top_n=args.top,
        tag=args.tag,
        threshold_s=args.threshold,
        scope_filter=compile_scope_filter(args.scope_filter, parser),
    )
    return SliceRequest(config=config, filter_name=args.filter)


def require_slice_field(slice_data: dict[str, Any], field_name: str, path: str) -> Any:
    if field_name not in slice_data:
        raise ValueError(f"{path}: missing required field '{field_name}'")
    return slice_data[field_name]


def validate_slice_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value):
        raise ValueError(f"{path}: id must match ^[a-z0-9][a-z0-9_.-]*$")
    return value


def slice_request_from_json(
    slice_data: dict[str, Any],
    parser: argparse.ArgumentParser,
    *,
    path: str,
    seen_ids: set[str],
) -> SliceRequest:
    if not isinstance(slice_data, dict):
        raise ValueError(f"{path}: slice entry must be an object")

    slice_id = validate_slice_id(require_slice_field(slice_data, "id", path), path)
    if slice_id in seen_ids:
        raise ValueError(f"{path}: duplicate slice id '{slice_id}'")
    seen_ids.add(slice_id)

    title = require_slice_field(slice_data, "title", path)
    filter_name = require_slice_field(slice_data, "filter", path)
    timing = require_slice_field(slice_data, "timing", path)
    sort_by = require_slice_field(slice_data, "sort", path)
    top_n = require_slice_field(slice_data, "top", path)
    threshold = require_slice_field(slice_data, "threshold", path)
    exclusive_scope = slice_data.get("exclusive_scope", "auto")
    scope_filter_pattern = slice_data.get("scope_filter", DEFAULT_SCOPE_FILTER)

    if not isinstance(title, str) or not title:
        raise ValueError(f"{path}: title must be a non-empty string")
    if not isinstance(filter_name, str) or not filter_name:
        raise ValueError(f"{path}: filter must be a non-empty string")
    if timing not in ("inclusive", "exclusive"):
        raise ValueError(f"{path}: timing must be 'inclusive' or 'exclusive'")
    if sort_by not in ("total", "avg", "avg-root-tu", "max"):
        raise ValueError(f"{path}: unsupported sort '{sort_by}'")
    if exclusive_scope not in ("auto", "all", "same-filter"):
        raise ValueError(f"{path}: unsupported exclusive_scope '{exclusive_scope}'")
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError(f"{path}: top must be a positive integer")
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError(f"{path}: threshold must be a non-negative number")
    if not isinstance(scope_filter_pattern, str):
        raise ValueError(f"{path}: scope_filter must be a string")

    children_data = slice_data.get("children", [])
    if not isinstance(children_data, list):
        raise ValueError(f"{path}: children must be a list")

    config = report_config(
        slice_id=slice_id,
        title=title,
        filter_name=filter_name,
        timing=timing,
        exclusive_scope=exclusive_scope,
        sort_by=sort_by,
        top_n=top_n,
        tag=None,
        threshold_s=float(threshold),
        scope_filter=compile_scope_filter(scope_filter_pattern, parser),
    )
    children = tuple(
        slice_request_from_json(
            child,
            parser,
            path=f"{path}.children[{index}]",
            seen_ids=seen_ids,
        )
        for index, child in enumerate(children_data)
    )
    return SliceRequest(config=config, filter_name=filter_name, children=children)


def read_slice_requests(
    slices_path: Path, parser: argparse.ArgumentParser
) -> list[SliceRequest]:
    try:
        with slices_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as e:
        parser.error(f"failed to read --slices file: {e}")
    except json.JSONDecodeError as e:
        parser.error(f"failed to parse --slices JSON: {e}")

    slices_data = payload.get("slices") if isinstance(payload, dict) else payload
    if not isinstance(slices_data, list) or not slices_data:
        parser.error(
            "--slices JSON must be a non-empty list or an object with a non-empty 'slices' list"
        )

    seen_ids: set[str] = set()
    try:
        return [
            slice_request_from_json(
                slice_data,
                parser,
                path=f"slices[{index}]",
                seen_ids=seen_ids,
            )
            for index, slice_data in enumerate(slices_data)
        ]
    except ValueError as e:
        parser.error(str(e))


def run_slice_report(
    request: SliceRequest,
    *,
    trace_dir: Path,
    baseline_dir: Path | None,
    repo_root: Path,
    baseline_repo_root: Path,
    output_dir: Path,
    output_csv: Path | None,
    allow_empty: bool,
) -> dict[str, Any]:
    config = request.config
    slice_output_dir = output_dir
    manifest: dict[str, Any] = {
        "id": config.slice_id,
        "title": config.title,
        "filter": request.filter_name,
        "filter_label": config.spec.label,
        "timing": config.timing,
        "exclusive_scope": config.exclusive_scope,
        "sort": config.sort_by,
        "top": config.top_n,
        "threshold_s": config.threshold_us / 1_000_000.0,
        "output_dir": slice_output_dir.as_posix(),
        "children": [],
    }

    if baseline_dir is not None:
        comparison_output_dir = slice_output_dir / "comparison"
        report_sides = (
            report_side(
                "baseline",
                baseline_dir,
                baseline_repo_root,
                slice_output_dir / "baseline",
            ),
            report_side("current", trace_dir, repo_root, slice_output_dir / "current"),
        )
        report_csvs: dict[str, tuple[Path, int]] = {}
        for side in report_sides:
            report_csvs[side.name] = write_side_report(side, config)

        comparison_stats, matched_trace_count = collect_comparison_stats(
            baseline_dir,
            trace_dir,
            baseline_repo_root,
            repo_root,
            config,
        )
        warnings: list[str] = []
        for side_name, (_, row_count) in report_csvs.items():
            if row_count == 0:
                warnings.append(
                    f"{side_name} report matched no events for this slice; "
                    f"check filter '{request.filter_name}', scope filtering, "
                    "and trace format"
                )
        if matched_trace_count == 0:
            warnings.append(
                "baseline and current trace directories have no matching trace files"
            )
        elif not comparison_stats:
            warnings.append(
                "baseline and current traces have no comparable event keys for this slice"
            )

        comparison_manifest: dict[str, Any] = {
            "matched_trace_count": matched_trace_count,
        }
        wrote: list[tuple[Path, int]] = []
        for direction in ("worse", "better"):
            rows = comparison_rows(comparison_stats, config, direction)
            comparison_csv = comparison_output_path(
                comparison_output_dir,
                config,
                direction,
            )
            write_comparison_csv(comparison_csv, rows, config.timing)
            row_dicts = [
                comparison_row_dict(rank, row, config.timing)
                for rank, row in enumerate(rows, start=1)
            ]
            comparison_manifest[direction] = {
                "csv": comparison_csv.as_posix(),
                "row_count": len(rows),
                "rows": row_dicts,
            }
            wrote.append((comparison_csv, len(rows)))

        manifest["reports"] = {
            side: {"csv": path.as_posix(), "row_count": row_count}
            for side, (path, row_count) in report_csvs.items()
        }
        manifest["comparison"] = comparison_manifest
        if warnings:
            manifest["warnings"] = warnings
        trace_counts = ", ".join(
            f"{len(side.trace_paths)} {side.name} trace(s)" for side in report_sides
        )
        print(
            f"wrote slice '{config.slice_id}' baseline/current reports and "
            f"comparison reports from {trace_counts}, "
            f"{matched_trace_count} matched trace file(s):"
        )
        for side_name, (path, _) in report_csvs.items():
            print(f"  {side_name}: {path}")
        for path, row_count in wrote:
            print(f"  comparison ({row_count} row(s)): {path}")
        for warning in warnings:
            print(f"  warning: {warning}")
    else:
        side = report_side("current", trace_dir, repo_root, slice_output_dir)
        stats = collect_stats(side.trace_paths, side.repo_root, config)
        if not stats and not allow_empty:
            raise SystemExit(
                f"no events matched filter '{request.filter_name}' "
                f"in {len(side.trace_paths)} trace(s)"
            )

        rows = sorted_rows(stats, config) if stats else []
        report_csv = (
            output_csv
            if output_csv is not None
            else default_output_path(slice_output_dir, config)
        )
        write_csv(report_csv, rows, config.timing)
        manifest["reports"] = {
            "current": {"csv": report_csv.as_posix(), "row_count": len(rows)}
        }
        if not rows:
            manifest["warnings"] = [
                f"current report matched no events for filter '{request.filter_name}'"
            ]
        print(
            f"wrote slice '{config.slice_id}' {len(rows)} row(s) "
            f"from {len(side.trace_paths)} trace(s) to {report_csv}"
        )

    manifest["children"] = [
        run_slice_report(
            child,
            trace_dir=trace_dir,
            baseline_dir=baseline_dir,
            repo_root=repo_root,
            baseline_repo_root=baseline_repo_root,
            output_dir=output_dir / child.config.slice_id,
            output_csv=None,
            allow_empty=allow_empty,
        )
        for child in request.children
    ]
    return manifest


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
            "comparison-only minimum total-impact change, in seconds, "
            "required for worse/better rows (default: 0)"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="exact output CSV path; overrides generated file name inside --output-dir",
    )
    parser.add_argument(
        "--slices",
        type=Path,
        help=(
            "JSON file describing multiple report slices; writes each slice under "
            "--output-dir/<slice-id> and emits --output-dir/summary.json"
        ),
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
        "--scope-filter",
        default=DEFAULT_SCOPE_FILTER,
        help=(
            "case-sensitive regex for symbol-scope reports; applies to demangled "
            "symbols and decoded Itanium-mangled namespace prefixes for symbol-like "
            "events only; pass an empty string to disable (default: CCCL top-level "
            "namespaces)"
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
    if args.slices is not None and args.output_csv is not None:
        parser.error("--output-csv cannot be used together with --slices")

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
    output_dir = (
        args.output_dir.resolve(strict=False)
        if args.output_dir
        else trace_dir / "event_reports"
    )
    output_csv = args.output_csv.resolve(strict=False) if args.output_csv else None
    multi_slice = args.slices is not None
    requests = (
        read_slice_requests(args.slices.resolve(strict=False), parser)
        if args.slices is not None
        else [single_slice_request(args, parser)]
    )

    manifest = {
        "schema_version": 1,
        "mode": "comparison" if baseline_dir is not None else "single",
        "trace_dir": trace_dir.as_posix(),
        "baseline_dir": baseline_dir.as_posix() if baseline_dir else None,
        "repo_root": repo_root.as_posix(),
        "baseline_repo_root": baseline_repo_root.as_posix(),
        "slices": [],
    }

    for request in requests:
        slice_output_dir = (
            output_dir / request.config.slice_id if multi_slice else output_dir
        )
        manifest["slices"].append(
            run_slice_report(
                request,
                trace_dir=trace_dir,
                baseline_dir=baseline_dir,
                repo_root=repo_root,
                baseline_repo_root=baseline_repo_root,
                output_dir=slice_output_dir,
                output_csv=output_csv,
                allow_empty=multi_slice,
            )
        )

    summary_json = output_dir / "summary.json"
    write_json(summary_json, manifest)
    print(f"wrote summary manifest: {summary_json}")


if __name__ == "__main__":
    main()
