#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

"""Build one instrumented object, then discard its outputs before measurement."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def instrumented_targets(build_dir: Path) -> list[str]:
    commands = json.loads((build_dir / "compile_commands.json").read_text())
    targets = []
    for entry in commands:
        args = entry.get("arguments") or shlex.split(entry["command"])
        if not any(arg.lstrip("-").startswith("fdevice-time-trace") for arg in args):
            continue
        output = entry.get("output") or args[args.index("-o") + 1]
        object_path = Path(entry["directory"]) / output
        targets.append(object_path.relative_to(build_dir).as_posix())
    return targets


def warmup(build_dir: Path, peer_build_dir: Path | None = None) -> None:
    targets = set(instrumented_targets(build_dir))
    if peer_build_dir is not None:
        targets.intersection_update(instrumented_targets(peer_build_dir))
    if not targets:
        raise SystemExit(
            f"no instrumented compile command shared with the peer found in {build_dir}"
        )
    target = min(targets)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", target, "-j1"], check=True
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "clean"], check=True
    )
    # NVCC trace files are side effects rather than Ninja outputs.
    # CMake creates the trace directories at configure time. Preserve
    # those directories so the measured NVCC invocations can open files.
    for trace in (build_dir / "compile_time" / "raw_traces").rglob("*.json"):
        trace.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--peer-build-dir", type=Path)
    args = parser.parse_args()
    warmup(
        args.build_dir.resolve(),
        args.peer_build_dir.resolve() if args.peer_build_dir else None,
    )


if __name__ == "__main__":
    main()
