#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path

from ci.compile_time.collect_traces import parse_input


def command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_configuration(roots: list[tuple[str, Path]]) -> dict:
    keys = {
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CUDA_FLAGS",
        "CMAKE_CXX_FLAGS",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_STANDARD",
        "CMAKE_CUDA_STANDARD",
        "TORCH_CUDA_ARCH_LIST",
        "CUDA_NVCC_FLAGS",
        "RAPIDS_CUDA_ARCHITECTURES",
    }
    configurations = {}
    for name, root in roots:
        for path in sorted(root.rglob("CMakeCache.txt")):
            values = {}
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                key, separator, value = line.partition("=")
                key = key.partition(":")[0]
                if separator and (
                    key in keys
                    or key.startswith(("CMAKE_CUDA_FLAGS_", "CMAKE_CXX_FLAGS_"))
                ):
                    values[key] = value
            configurations[f"{name}/{path.relative_to(root)}"] = values
    return configurations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record provenance for original compiler traces."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--cccl-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--build-exit-code", type=int, required=True)
    parser.add_argument("--collection-exit-code", type=int, required=True)
    parser.add_argument("--source", type=parse_input, action="append", default=[])
    parser.add_argument("--build-tree", type=parse_input, action="append", default=[])
    args = parser.parse_args()
    trace_count = sum(1 for path in args.trace_dir.rglob("*.json") if path.is_file())
    complete = (
        args.build_exit_code == args.collection_exit_code == 0 and trace_count > 0
    )
    data = {
        "schema_version": 1,
        "status": "complete" if complete else "partial" if trace_count else "empty",
        "project": args.project,
        "cccl_sha": command_output(
            ["git", "-C", str(args.cccl_root), "rev-parse", "HEAD"]
        ),
        "repo_root": str(args.repo_root),
        "started_at": args.started_at,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "build_exit_code": args.build_exit_code,
        "collection_exit_code": args.collection_exit_code,
        "trace_count": trace_count,
        "parallel_level": os.environ.get("PARALLEL_LEVEL"),
        "build_configuration": build_configuration(args.build_tree),
        "sources": {
            name: {
                "path": str(path),
                "sha": command_output(["git", "-C", str(path), "rev-parse", "HEAD"]),
            }
            for name, path in args.source
        },
        "toolchain": {
            "nvcc": command_output(["nvcc", "--version"]),
            "cxx": command_output(["c++", "--version"]),
            "platform": platform.platform(),
        },
        "workflow": {
            key.lower(): os.environ.get(key)
            for key in (
                "GITHUB_REPOSITORY",
                "GITHUB_SHA",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Recorded {data['status']} dataset with {trace_count} original trace(s): {args.output}"
    )
    if not trace_count:
        raise SystemExit("No compiler traces were collected")


if __name__ == "__main__":
    main()
