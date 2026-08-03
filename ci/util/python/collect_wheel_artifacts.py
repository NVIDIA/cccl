#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect wheel artifacts without silently overwriting duplicate filenames."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

from wheel_release_expectations import (
    WheelExpectation,
    load_release_expectations,
    wheel_compatibility_tags,
)

_RELEASE_ARTIFACT = re.compile(r"wheel-cccl-(?:linux|windows)-[^/]+-py[^/]+")


def _logical_contents(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in sorted(archive.namelist())}


def _artifact_name(source: Path, wheel: Path) -> str | None:
    relative_parts = wheel.relative_to(source).parts
    return relative_parts[0] if len(relative_parts) > 1 else None


def _is_release_artifact(artifact_name: str | None) -> bool:
    return bool(artifact_name and _RELEASE_ARTIFACT.fullmatch(artifact_name))


def _is_universal_wheel(name: str) -> bool:
    return name.endswith("-py3-none-any.whl")


def _validate_expected_artifacts(
    source: Path,
    wheels: list[Path],
    expectations: tuple[WheelExpectation, ...],
) -> None:
    expected_names = {expectation.artifact_name for expectation in expectations}
    actual_names = {
        artifact_name
        for wheel in wheels
        if (artifact_name := _artifact_name(source, wheel)) is not None
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            "Release artifacts do not match the generated workflow "
            f"(missing={missing}, unexpected={unexpected})"
        )

    compute_wheels: dict[str, Path] = {}
    for expectation in expectations:
        artifact_wheels = [
            wheel
            for wheel in wheels
            if _artifact_name(source, wheel) == expectation.artifact_name
        ]
        headers_wheels = [
            wheel for wheel in artifact_wheels if wheel.name.startswith("cccl_headers-")
        ]
        compute_candidates = [
            wheel for wheel in artifact_wheels if wheel.name.startswith("cuda_compute-")
        ]
        meta_wheels = [
            wheel for wheel in artifact_wheels if wheel.name.startswith("cuda_cccl-")
        ]
        recognized = {*headers_wheels, *compute_candidates, *meta_wheels}
        if (
            len(headers_wheels) != 1
            or len(compute_candidates) != 1
            or len(meta_wheels) != 1
        ):
            raise RuntimeError(
                f"Expected one wheel for each coordinated distribution in "
                f"{expectation.artifact_name}"
            )
        if set(artifact_wheels) != recognized:
            unexpected = sorted(
                wheel.name for wheel in set(artifact_wheels) - recognized
            )
            raise RuntimeError(
                f"Unexpected wheels in {expectation.artifact_name}: {unexpected}"
            )
        for universal_wheel in (*headers_wheels, *meta_wheels):
            if not _is_universal_wheel(universal_wheel.name):
                raise RuntimeError(
                    f"Expected a universal wheel in {expectation.artifact_name}: "
                    f"{universal_wheel.name}"
                )

        compute_wheel = compute_candidates[0]
        tags = wheel_compatibility_tags(compute_wheel)
        if not expectation.matches_compute_tags(tags):
            raise RuntimeError(
                f"{compute_wheel.name} does not match producer artifact "
                f"{expectation.artifact_name}"
            )
        if previous := compute_wheels.get(compute_wheel.name):
            raise RuntimeError(
                f"Producer artifacts {previous.parent} and {compute_wheel.parent} "
                f"contain the same cuda-compute wheel {compute_wheel.name}"
            )
        compute_wheels[compute_wheel.name] = compute_wheel


def collect_wheels(
    source: Path,
    destination: Path,
    canonical_artifact: str,
    workflow_file: Path | None = None,
) -> None:
    discovered_wheels = sorted(source.rglob("*.whl"))
    wheels = [
        wheel
        for wheel in discovered_wheels
        if _is_release_artifact(_artifact_name(source, wheel))
    ]
    if not wheels:
        raise RuntimeError(f"No release wheel artifacts found below {source}")

    if workflow_file is not None:
        expectations = load_release_expectations(workflow_file)
        if canonical_artifact not in {
            expectation.artifact_name for expectation in expectations
        }:
            raise RuntimeError(
                f"Canonical artifact {canonical_artifact} is not a producer in "
                f"{workflow_file}"
            )
        _validate_expected_artifacts(source, wheels, expectations)

    for wheel in sorted(set(discovered_wheels) - set(wheels)):
        print(f"Ignored wheel outside a release artifact: {wheel}")

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"Destination must be empty: {destination}")

    by_name: dict[str, list[Path]] = {}
    for wheel in wheels:
        by_name.setdefault(wheel.name, []).append(wheel)

    for name, candidates in sorted(by_name.items()):
        candidates.sort()
        canonical_candidates = [
            path
            for path in candidates
            if _artifact_name(source, path) == canonical_artifact
        ]
        if _is_universal_wheel(name):
            if not canonical_candidates:
                raise RuntimeError(
                    f"Canonical artifact {canonical_artifact} does not contain {name}"
                )
            selected = canonical_candidates[0]
        else:
            selected = candidates[0]

        if len(candidates) > 1:
            selected_contents = _logical_contents(selected)
            for duplicate in candidates:
                if duplicate == selected:
                    continue
                if _logical_contents(duplicate) != selected_contents:
                    locations = "\n  ".join(str(path) for path in candidates)
                    raise RuntimeError(
                        f"Conflicting wheels share the filename {name}:\n  {locations}"
                    )
        shutil.copy2(selected, destination / name)
        if len(candidates) > 1:
            print(
                f"Selected {selected} after verifying {len(candidates)} "
                f"logically identical copies of {name}"
            )
        else:
            print(f"Selected {selected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--workflow",
        required=True,
        type=Path,
        help="generated workflow.json that defines the complete release wheel matrix",
    )
    parser.add_argument(
        "--canonical-artifact",
        default="wheel-cccl-linux-amd64-py3.10",
        help="required source artifact for universal wheels",
    )
    args = parser.parse_args()

    try:
        collect_wheels(
            args.source,
            args.destination,
            args.canonical_artifact,
            args.workflow,
        )
    except (json.JSONDecodeError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
