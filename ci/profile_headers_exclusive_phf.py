#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HeaderEvent:
    start_us: int
    end_us: int
    header: str | None
    child_intervals: list[tuple[int, int]] = field(default_factory=list)

    @property
    def inclusive_us(self) -> int:
        return self.end_us - self.start_us

    @property
    def exclusive_us(self) -> int:
        return max(
            0, self.inclusive_us - merged_interval_duration(self.child_intervals)
        )


@dataclass
class HeaderStats:
    public_tus: set[str] = field(default_factory=set)
    event_count: int = 0
    inclusive_us: int = 0
    exclusive_us: int = 0


def merged_interval_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0

    total = 0
    merged_start, merged_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start > merged_end:
            total += merged_end - merged_start
            merged_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)

    total += merged_end - merged_start
    return total


def generated_header_from_tu(tu: str) -> str:
    marker = "/headers/"
    if marker not in tu:
        return tu

    rel = tu.split(marker, 1)[1]
    parts = rel.split("/", 1)
    if len(parts) != 2:
        return tu

    header = parts[1]
    for suffix in (".cu", ".cpp", ".cxx", ".cc", ".c"):
        if header.endswith(suffix):
            return header[: -len(suffix)]

    return header


def normalize_header(path: str, repo_root: Path) -> str | None:
    if not path:
        return None

    path = path.strip()
    if not path:
        return None

    header_path = Path(path)
    if not header_path.is_absolute():
        return None

    try:
        rel = header_path.resolve(strict=False).relative_to(repo_root)
    except ValueError:
        return None

    rel_path = rel.as_posix()
    for prefix in ("libcudacxx/include/", "cudax/include/", "c/parallel/include/"):
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix) :]
            break

    if rel_path.startswith("cub/cub/"):
        rel_path = "cub/" + rel_path[len("cub/cub/") :]
    elif rel_path.startswith("thrust/thrust/"):
        rel_path = "thrust/" + rel_path[len("thrust/thrust/") :]

    if rel_path.startswith("build/"):
        return None
    if not rel_path.startswith(("cub/", "thrust/", "cuda/", "cccl/")):
        return None

    return rel_path


def trace_root_tu(trace: dict, trace_path: Path) -> str:
    input_files = trace.get("otherData", {}).get("inputFiles", [])
    if input_files:
        return generated_header_from_tu(input_files[0])
    return trace_path.as_posix()


def direct_header_events(trace: dict, repo_root: Path) -> list[HeaderEvent]:
    grouped: dict[tuple[int, int], list[HeaderEvent]] = defaultdict(list)

    for event in trace.get("traceEvents", []):
        if event.get("name") != "Processing Header File":
            continue
        if event.get("ph") not in (None, "X"):
            continue
        if "ts" not in event or "dur" not in event:
            continue

        detail = event.get("args", {}).get("detail", "")
        if not detail:
            continue

        start_us = int(event["ts"])
        dur_us = int(event["dur"])
        grouped[(int(event.get("pid", 0)), int(event.get("tid", 0)))].append(
            HeaderEvent(
                start_us=start_us,
                end_us=start_us + dur_us,
                header=normalize_header(detail, repo_root),
            )
        )

    all_events = []
    for events in grouped.values():
        stack: list[HeaderEvent] = []
        for event in sorted(events, key=lambda e: (e.start_us, -e.end_us)):
            while stack and stack[-1].end_us <= event.start_us:
                stack.pop()

            if (
                stack
                and stack[-1].start_us <= event.start_us
                and event.end_us <= stack[-1].end_us
            ):
                if (
                    stack[-1].start_us != event.start_us
                    or stack[-1].end_us != event.end_us
                ):
                    stack[-1].child_intervals.append((event.start_us, event.end_us))

            stack.append(event)
            if event.header is not None:
                all_events.append(event)

    return all_events


def collect_stats(trace_paths: list[Path], repo_root: Path) -> dict[str, HeaderStats]:
    stats: dict[str, HeaderStats] = defaultdict(HeaderStats)

    for trace_path in trace_paths:
        with trace_path.open(encoding="utf-8") as f:
            trace = json.load(f)

        public_tu = trace_root_tu(trace, trace_path)
        for event in direct_header_events(trace, repo_root):
            header_stats = stats[event.header]
            header_stats.public_tus.add(public_tu)
            header_stats.event_count += 1
            header_stats.inclusive_us += event.inclusive_us
            header_stats.exclusive_us += event.exclusive_us

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute exclusive header-processing metrics from NVCC device-time-trace JSON files."
    )
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--repo-root", default=Path(__file__).resolve().parents[1], type=Path
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve(strict=False)
    trace_paths = sorted(args.trace_dir.rglob("*.json"))
    if not trace_paths:
        raise SystemExit(
            f"no device-time-trace JSON files found under {args.trace_dir}"
        )

    stats = collect_stats(trace_paths, repo_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "header_path",
                "weighted_exclusive_process_time_s",
                "public_tu_count",
                "avg_exclusive_process_time_s",
                "total_exclusive_process_time_s",
                "total_inclusive_process_time_s",
                "event_count",
            ],
        )
        writer.writeheader()

        rows = sorted(stats.items(), key=lambda item: (-item[1].exclusive_us, item[0]))
        for header, header_stats in rows:
            public_tu_count = len(header_stats.public_tus)
            avg_exclusive_us = (
                header_stats.exclusive_us / public_tu_count if public_tu_count else 0
            )
            writer.writerow(
                {
                    "header_path": header,
                    "weighted_exclusive_process_time_s": f"{header_stats.exclusive_us / 1_000_000.0:.6f}",
                    "public_tu_count": public_tu_count,
                    "avg_exclusive_process_time_s": f"{avg_exclusive_us / 1_000_000.0:.6f}",
                    "total_exclusive_process_time_s": f"{header_stats.exclusive_us / 1_000_000.0:.6f}",
                    "total_inclusive_process_time_s": f"{header_stats.inclusive_us / 1_000_000.0:.6f}",
                    "event_count": header_stats.event_count,
                }
            )


if __name__ == "__main__":
    main()
