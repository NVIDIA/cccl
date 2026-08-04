#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate the coordinated cccl-headers, cuda-compute, and cuda-cccl wheels."""

from __future__ import annotations

import argparse
import email
import json
import re
import sys
import zipfile
from pathlib import Path

from wheel_release_expectations import (
    WheelExpectation,
    load_release_expectations,
    wheel_compatibility_tags,
)


def _wheels(wheelhouse: Path, prefix: str) -> list[Path]:
    matches = sorted(wheelhouse.glob(f"{prefix}-*.whl"))
    if not matches:
        raise RuntimeError(f"Expected at least one {prefix} wheel in {wheelhouse}")
    return matches


def _one_wheel(wheelhouse: Path, prefix: str) -> Path:
    matches = _wheels(wheelhouse, prefix)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {prefix} wheel in {wheelhouse}, found {len(matches)}"
        )
    return matches[0]


def _metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected one METADATA file in {wheel}")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
    return metadata, names


def _payload(names: list[str]) -> set[str]:
    return {
        name for name in names if ".dist-info/" not in name and not name.endswith("/")
    }


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _validate_identity(metadata, wheel: Path, expected_name: str) -> str:
    name = metadata["Name"]
    if name is None or _normalized_distribution_name(name) != expected_name:
        raise RuntimeError(
            f"Expected {wheel.name} to identify as {expected_name}, found {name!r}"
        )

    version = metadata["Version"]
    if version is None:
        raise RuntimeError(f"Expected a Version in {wheel.name}")
    return version


def _has_exact_requirement(requirements: list[str], name: str, version: str) -> bool:
    expected = f"{name}=={version}".lower()
    for requirement in requirements:
        requirement_text, separator, marker = requirement.partition(";")
        if separator and marker.strip():
            continue
        if requirement_text.replace(" ", "").lower() == expected:
            return True
    return False


def _requirement_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        requirement_text, separator, marker = requirement.partition(";")
        if separator and marker.strip():
            continue
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement_text)
        if match:
            names.add(_normalized_distribution_name(match.group(1)))
    return names


def validate(wheelhouse: Path, workflow_file: Path | None = None) -> None:
    expectations: tuple[WheelExpectation, ...] = ()
    if workflow_file is not None:
        expectations = load_release_expectations(workflow_file)

    headers_wheel = _one_wheel(wheelhouse, "cccl_headers")
    compute_wheels = _wheels(wheelhouse, "cuda_compute")
    meta_wheel = _one_wheel(wheelhouse, "cuda_cccl")

    coordinated_wheels = {headers_wheel, meta_wheel, *compute_wheels}
    unexpected_wheels = set(wheelhouse.glob("*.whl")) - coordinated_wheels
    if unexpected_wheels:
        raise RuntimeError(
            "Unexpected wheels in coordinated release: "
            f"{sorted(wheel.name for wheel in unexpected_wheels)}"
        )

    headers_metadata, headers_names = _metadata(headers_wheel)
    meta_metadata, meta_names = _metadata(meta_wheel)
    compute_artifacts = [(wheel, *_metadata(wheel)) for wheel in compute_wheels]

    versions = {
        _validate_identity(headers_metadata, headers_wheel, "cccl-headers"),
        _validate_identity(meta_metadata, meta_wheel, "cuda-cccl"),
        *(
            _validate_identity(metadata, wheel, "cuda-compute")
            for wheel, metadata, _ in compute_artifacts
        ),
    }
    if len(versions) != 1:
        raise RuntimeError(f"Wheel versions are not coordinated: {sorted(versions)}")
    version = versions.pop()

    meta_requirements = meta_metadata.get_all("Requires-Dist", [])
    if not _has_exact_requirement(meta_requirements, "cuda-compute", version):
        raise RuntimeError(
            "cuda-cccl does not exactly require the cuda-compute version"
        )

    headers_payload = _payload(headers_names)
    meta_payload = _payload(meta_names)
    if meta_payload:
        raise RuntimeError(f"cuda-cccl metapackage owns payload files: {meta_payload}")
    if any(not name.startswith("cuda/cccl/") for name in headers_payload):
        raise RuntimeError("cccl-headers owns files outside cuda/cccl")

    compute_tags: dict[tuple[str, str, str], Path] = {}
    observed_expectations: set[WheelExpectation] = set()
    for compute_wheel, compute_metadata, compute_names in compute_artifacts:
        tags = wheel_compatibility_tags(compute_wheel)
        if tags == ("py3", "none", "any"):
            raise RuntimeError(
                f"Expected a platform cuda-compute wheel: {compute_wheel.name}"
            )
        if previous_wheel := compute_tags.get(tags):
            raise RuntimeError(
                "Duplicate cuda-compute compatibility tags "
                f"{'-'.join(tags)} in {previous_wheel.name} and "
                f"{compute_wheel.name}"
            )
        compute_tags[tags] = compute_wheel

        if expectations:
            matching_expectations = [
                expectation
                for expectation in expectations
                if expectation.matches_compute_tags(tags)
            ]
            if len(matching_expectations) != 1:
                raise RuntimeError(
                    f"{compute_wheel.name} does not match exactly one producer in "
                    f"{workflow_file}"
                )
            observed_expectations.add(matching_expectations[0])

        compute_requirements = compute_metadata.get_all("Requires-Dist", [])
        required_compute_dependencies = {
            "cccl-headers",
            "cuda-core",
            "cuda-pathfinder",
            "numpy",
            "typing-extensions",
        }
        missing_compute_dependencies = (
            required_compute_dependencies - _requirement_names(compute_requirements)
        )
        if missing_compute_dependencies:
            raise RuntimeError(
                f"{compute_wheel.name} is missing runtime dependencies: "
                f"{sorted(missing_compute_dependencies)}"
            )
        if not _has_exact_requirement(compute_requirements, "cccl-headers", version):
            raise RuntimeError(
                f"{compute_wheel.name} does not exactly require "
                "the cccl-headers version"
            )

        compute_payload = _payload(compute_names)
        if headers_payload & compute_payload:
            raise RuntimeError(
                f"cccl-headers and {compute_wheel.name} own overlapping files"
            )
        if any(not name.startswith("cuda/compute/") for name in compute_payload):
            raise RuntimeError(f"{compute_wheel.name} owns files outside cuda/compute")
        if "cuda/compute/__init__.py" not in compute_payload:
            raise RuntimeError(f"{compute_wheel.name} is missing cuda.compute")

    expected_headers = {
        "cuda/cccl/__init__.py",
        "cuda/cccl/headers/include/cub/version.cuh",
        "cuda/cccl/headers/include/thrust/version.h",
        "cuda/cccl/headers/include/cuda/version",
        "cuda/cccl/headers/include/cuda/experimental/coop.cuh",
    }
    missing_headers = expected_headers - headers_payload
    if missing_headers:
        raise RuntimeError(
            f"cccl-headers is missing headers: {sorted(missing_headers)}"
        )

    for universal_wheel in (headers_wheel, meta_wheel):
        if wheel_compatibility_tags(universal_wheel) != ("py3", "none", "any"):
            raise RuntimeError(f"Expected a universal wheel: {universal_wheel.name}")

    if expectations and observed_expectations != set(expectations):
        missing = sorted(
            expectation.artifact_name
            for expectation in set(expectations) - observed_expectations
        )
        raise RuntimeError(
            f"Missing cuda-compute wheels required by the generated workflow: {missing}"
        )

    print(
        "Validated coordinated CCCL Python wheel set at version "
        f"{version} with {len(compute_wheels)} cuda-compute wheel(s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument(
        "--workflow",
        type=Path,
        help="generated workflow.json that defines the complete release wheel matrix",
    )
    args = parser.parse_args()
    try:
        validate(args.wheelhouse, args.workflow)
    except (json.JSONDecodeError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
