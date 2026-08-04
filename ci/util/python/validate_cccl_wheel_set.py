#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate the coordinated cccl-headers, cuda-compute, and cuda-cccl wheels."""

from __future__ import annotations

import argparse
import email
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path


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


_SPECIFIER_PART = re.compile(r"\s*(===|==|!=|<=|>=|~=|<|>)\s*([^\s,()]+)\s*")
_CURRENT_EXTRA_MARKER = re.compile(
    r"\s*(?:python_version\s*<\s*(?:\"3\.11\"|'3\.11')\s+and\s+)?"
    r"extra\s*==\s*(?:\"[A-Za-z0-9][A-Za-z0-9._-]*\"|"
    r"'[A-Za-z0-9][A-Za-z0-9._-]*')\s*"
)


def _normalized_specifier(specifier: str) -> str | None:
    specifier = specifier.strip()
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1]
    if not specifier:
        return ""
    matches = [_SPECIFIER_PART.fullmatch(part) for part in specifier.split(",")]
    if any(match is None for match in matches):
        return None
    return ",".join(
        sorted(
            f"{match.group(1)}{match.group(2)}"
            for match in matches
            if match is not None
        )
    )


def _parse_unconditional_requirement(requirement: str) -> tuple[str, str] | None:
    requirement_text, separator, _marker = requirement.partition(";")
    if separator or "@" in requirement_text or "[" in requirement_text:
        return None

    match = re.fullmatch(
        r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*", requirement_text
    )
    if match is None:
        return None
    specifier = _normalized_specifier(match.group(2))
    if specifier is None:
        return None
    return (
        _normalized_distribution_name(match.group(1)),
        specifier,
    )


def _is_extra_guarded_requirement(requirement: str) -> bool:
    requirement_text, separator, marker = requirement.partition(";")
    if not separator or not marker.strip() or "@" in requirement_text:
        return False
    # These are the two marker shapes emitted by the current project extras.
    # Fail closed if that metadata contract changes instead of maintaining a
    # second, partial PEP 508 parser in release tooling.
    return _CURRENT_EXTRA_MARKER.fullmatch(marker) is not None


def _validate_unconditional_requirements(
    wheel: Path,
    requirements: list[str],
    expected_requirements: tuple[tuple[str, str], ...],
) -> None:
    expected_entries = []
    for name, specifier in expected_requirements:
        normalized_specifier = _normalized_specifier(specifier)
        if normalized_specifier is None:
            raise AssertionError(f"Invalid expected dependency: {name}{specifier}")
        expected_entries.append(
            (_normalized_distribution_name(name), normalized_specifier)
        )
    expected = Counter(expected_entries)
    # Optional dependencies emitted by the build backend are guarded by their
    # extra. Treat every other entry as part of the base contract so malformed
    # markers and semicolons inside direct-reference URLs cannot bypass it.
    unconditional = [
        requirement
        for requirement in requirements
        if not _is_extra_guarded_requirement(requirement)
    ]
    parsed = [
        _parse_unconditional_requirement(requirement) for requirement in unconditional
    ]
    actual = Counter(parsed)
    if actual == expected:
        return

    missing = [
        f"{name}{specifier}" for name, specifier in (expected - actual).elements()
    ]
    unexpected_counts = actual - expected
    unexpected = []
    for requirement, parsed_requirement in zip(unconditional, parsed):
        if unexpected_counts[parsed_requirement]:
            unexpected.append(requirement)
            unexpected_counts[parsed_requirement] -= 1

    raise RuntimeError(
        f"{wheel.name} has invalid unconditional runtime dependencies; "
        f"missing or altered: {missing}; unexpected or duplicate: {unexpected}"
    )


def _wheel_compatibility_tags(wheel: Path) -> tuple[str, str, str]:
    components = wheel.name.removesuffix(".whl").rsplit("-", 3)
    if len(components) != 4 or not all(components[-3:]):
        raise RuntimeError(f"Unable to parse wheel compatibility tags: {wheel.name}")
    return components[-3], components[-2], components[-1]


def _is_release_platform_tag(platform_tag: str) -> bool:
    if platform_tag in {"win_amd64", "win_arm64"}:
        return True
    manylinux = re.compile(
        r"manylinux(?:1|2010|2014|_[0-9]+_[0-9]+)_(?:x86_64|aarch64)"
    )
    return all(manylinux.fullmatch(tag) is not None for tag in platform_tag.split("."))


def validate(wheelhouse: Path, require_release_tags: bool = False) -> None:
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

    _validate_unconditional_requirements(
        headers_wheel,
        headers_metadata.get_all("Requires-Dist", []),
        (("cuda-pathfinder", ">=1.2.3"),),
    )
    _validate_unconditional_requirements(
        meta_wheel,
        meta_metadata.get_all("Requires-Dist", []),
        (("cuda-compute", f"=={version}"),),
    )

    headers_payload = _payload(headers_names)
    meta_payload = _payload(meta_names)
    if meta_payload:
        raise RuntimeError(f"cuda-cccl metapackage owns payload files: {meta_payload}")
    if any(not name.startswith("cuda/cccl/") for name in headers_payload):
        raise RuntimeError("cccl-headers owns files outside cuda/cccl")

    compute_tags: dict[tuple[str, str, str], Path] = {}
    for compute_wheel, compute_metadata, compute_names in compute_artifacts:
        tags = _wheel_compatibility_tags(compute_wheel)
        if tags == ("py3", "none", "any"):
            raise RuntimeError(
                f"Expected a platform cuda-compute wheel: {compute_wheel.name}"
            )
        if require_release_tags and not _is_release_platform_tag(tags[2]):
            raise RuntimeError(
                f"Expected a repaired release cuda-compute wheel: {compute_wheel.name}"
            )
        if previous_wheel := compute_tags.get(tags):
            raise RuntimeError(
                "Duplicate cuda-compute compatibility tags "
                f"{'-'.join(tags)} in {previous_wheel.name} and "
                f"{compute_wheel.name}"
            )
        compute_tags[tags] = compute_wheel

        compute_requirements = compute_metadata.get_all("Requires-Dist", [])
        # Keep this runtime contract synchronized with [project].dependencies
        # in python/cuda_compute/pyproject.toml. The exact cccl-headers pin is
        # checked separately below because its version is dynamic.
        required_compute_dependencies = (
            ("numpy", ""),
            ("cuda-pathfinder", ">=1.2.3"),
            ("cuda-core", ""),
            ("typing-extensions", ""),
        )
        _validate_unconditional_requirements(
            compute_wheel,
            compute_requirements,
            (*required_compute_dependencies, ("cccl-headers", f"=={version}")),
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
        if _wheel_compatibility_tags(universal_wheel) != ("py3", "none", "any"):
            raise RuntimeError(f"Expected a universal wheel: {universal_wheel.name}")

    print(
        "Validated coordinated CCCL Python wheel set at version "
        f"{version} with {len(compute_wheels)} cuda-compute wheel(s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument(
        "--require-release-tags",
        action="store_true",
        help="reject raw linux and other non-release platform tags",
    )
    args = parser.parse_args()
    try:
        validate(args.wheelhouse, args.require_release_tags)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
