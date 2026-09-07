#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

DETAIL_EVENT_NAMES = {
    "Code Generation Function",
    "CodeGen Function",
    "ExecuteCompiler",
    "Frontend",
    "Instantiating Template Class",
    "Instantiating Template Function",
    "InstantiateClass",
    "InstantiateFunction",
    "OptFunction",
    "ParseClass",
    "PerformPendingInstantiations",
    "Processing Header File",
    "RunPass",
    "Scanning Function Body",
    "Source",
}

DETAIL_PREFIXES_TO_STRIP = (
    "libcudacxx/include/",
    "cudax/include/",
    "c/parallel/include/",
)

DETAIL_PREFIXES_TO_COLLAPSE = (
    ("cub/cub/", "cub/"),
    ("thrust/thrust/", "thrust/"),
)


def normalize_detail(detail: str, repo_root: Path) -> str:
    detail_path = Path(detail)
    if detail_path.is_absolute():
        try:
            rel = detail_path.resolve(strict=False).relative_to(repo_root)
            detail = rel.as_posix()
        except ValueError:
            pass

    for prefix in DETAIL_PREFIXES_TO_STRIP:
        if detail.startswith(prefix):
            detail = detail[len(prefix) :]
            break

    for prefix, replacement in DETAIL_PREFIXES_TO_COLLAPSE:
        if detail.startswith(prefix):
            detail = replacement + detail[len(prefix) :]
            break

    return detail


def display_detail(detail: str, repo_root: Path, max_detail_len: int | None) -> str:
    detail = normalize_detail(detail, repo_root)
    if (
        max_detail_len is not None
        and max_detail_len > 0
        and len(detail) > max_detail_len
    ):
        return detail[: max_detail_len - 1] + "..."
    return detail


def rewrite_event_name(
    event: dict, repo_root: Path, max_detail_len: int | None
) -> bool:
    name = event.get("name")
    if name not in DETAIL_EVENT_NAMES:
        return False

    args = event.get("args")
    if not isinstance(args, dict):
        return False

    detail = args.get("detail")
    if not detail:
        return False

    args.setdefault("original_name", name)
    event["name"] = f"{name}: {display_detail(str(detail), repo_root, max_detail_len)}"
    return True


def prepare_trace(
    input_path: Path, output_path: Path, repo_root: Path, max_detail_len: int | None
) -> int:
    with input_path.open(encoding="utf-8") as f:
        trace = json.load(f)

    rewritten = 0
    for event in trace.get("traceEvents", []):
        if rewrite_event_name(event, repo_root, max_detail_len):
            rewritten += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, separators=(",", ":"))

    return rewritten


def iter_input_traces(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.json"))


def output_path_for(input_trace: Path, input_root: Path, output_path: Path) -> Path:
    if input_root.is_file():
        if output_path.is_dir() or not output_path.suffix:
            return output_path / f"{input_trace.stem}.perfetto.json"
        return output_path

    rel = input_trace.relative_to(input_root)
    return output_path / rel.parent / f"{rel.stem}.perfetto.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NVCC device-time-trace JSON files for Perfetto by promoting args.detail into event names."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input trace JSON file or directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output trace JSON file or directory"
    )
    parser.add_argument(
        "--repo-root", default=Path(__file__).resolve().parents[2], type=Path
    )
    parser.add_argument(
        "--max-detail-len",
        default=0,
        type=int,
        help="Truncate promoted detail text to this many characters; 0 keeps full details",
    )
    args = parser.parse_args()

    input_path = args.input.resolve(strict=False)
    output_path = args.output.resolve(strict=False)
    repo_root = args.repo_root.resolve(strict=False)
    max_detail_len = args.max_detail_len if args.max_detail_len > 0 else None

    traces = iter_input_traces(input_path)
    if not traces:
        raise SystemExit(f"no JSON traces found under {args.input}")

    total_rewritten = 0
    for trace_path in traces:
        total_rewritten += prepare_trace(
            trace_path,
            output_path_for(trace_path, input_path, output_path),
            repo_root,
            max_detail_len,
        )

    print(f"prepared {len(traces)} trace(s); renamed {total_rewritten} event(s)")


if __name__ == "__main__":
    main()
