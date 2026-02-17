#!/usr/bin/env python
"""Generate a manifest for the web benchmark report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def normalize_axis_name(name: str) -> str:
    if "{" in name:
        return name.split("{")[0]
    return name


def read_json(path: Path) -> dict | None:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        print(f"Warning: {path} is not valid JSON, skipping")
    except OSError as exc:
        print(f"Warning: failed to read {path}: {exc}")
    return None


def extract_axes(root: dict) -> list[str]:
    axes: list[str] = []
    for benchmark in root.get("benchmarks", []):
        for axis in benchmark.get("axes", []):
            name = normalize_axis_name(axis.get("name", ""))
            if name and name not in axes:
                axes.append(name)
    return axes


def extract_device(root: dict) -> dict | None:
    devices = root.get("devices", [])
    if not devices:
        return None
    device = devices[0]
    return {
        "id": device.get("id"),
        "name": device.get("name"),
        "sm_version": device.get("sm_version"),
    }


def build_manifest(results_dir: Path) -> dict:
    entries = []
    for py_path in sorted(results_dir.rglob("*_py.json")):
        name = py_path.name
        base = name[: -len("_py.json")]
        cpp_path = py_path.with_name(f"{base}_cpp.json")
        if not cpp_path.exists():
            continue

        rel_dir = py_path.parent.relative_to(results_dir)
        benchmark_id = str(rel_dir / base)

        py_root = read_json(py_path)
        cpp_root = read_json(cpp_path)
        if py_root is None or cpp_root is None:
            continue

        axes = extract_axes(py_root)
        for axis in extract_axes(cpp_root):
            if axis not in axes:
                axes.append(axis)

        device = extract_device(py_root) or extract_device(cpp_root)

        entries.append(
            {
                "id": benchmark_id,
                "label": benchmark_id,
                "category": str(rel_dir) if str(rel_dir) != "." else "",
                "name": base,
                "py_path": str(Path(str(rel_dir)) / f"{base}_py.json"),
                "cpp_path": str(Path(str(rel_dir)) / f"{base}_cpp.json"),
                "axes": axes,
                "device": device,
            }
        )

    entries.sort(key=lambda item: item["id"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results_base": "../../results",
        "benchmarks": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing nvbench JSON results",
    )
    parser.add_argument(
        "--output",
        default="results/manifest.json",
        help="Output manifest path",
    )
    parser.add_argument(
        "--public-output",
        default="analysis/web-report/public/manifest.json",
        help="Optional manifest copy for Vite dev public dir",
    )
    parser.add_argument(
        "--results-base",
        default=".",
        help="Base path used by the web app to fetch results",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    public_output_path = Path(args.public_output)

    manifest = build_manifest(results_dir)
    manifest["results_base"] = args.results_base

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    if public_output_path != output_path:
        public_output_path.parent.mkdir(parents=True, exist_ok=True)
        with public_output_path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

    print("Wrote manifest with " f"{len(manifest['benchmarks'])} benchmarks")


if __name__ == "__main__":
    main()
