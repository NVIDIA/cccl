#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate a component release and stage only its wheels."""

from __future__ import annotations

import argparse
import email
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

_DISTRIBUTIONS = frozenset({"cuda-compute", "cuda-cccl"})
_RELEASE_ARTIFACT = re.compile(r"wheel-cccl-(?:linux|windows)-[^/]+-py[^/]+")
_INTEGER = r"(?:0|[1-9][0-9]*)"
_CANONICAL_VERSION = re.compile(
    rf"{_INTEGER}(?:\.{_INTEGER})*"
    rf"(?:(?:a|b|rc){_INTEGER})?"
    rf"(?:\.post{_INTEGER})?"
    rf"(?:\.dev{_INTEGER})?"
)


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def release_tag(distribution: str, version: str) -> str:
    if distribution not in _DISTRIBUTIONS:
        raise RuntimeError(f"Unsupported distribution: {distribution!r}")
    if len(version) > 128 or _CANONICAL_VERSION.fullmatch(version) is None:
        raise RuntimeError(
            f"Version must use canonical public PEP 440 form: {version!r}"
        )
    return f"{distribution}-{version}"


def _metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected one METADATA file in {wheel}")
        return email.message_from_bytes(archive.read(metadata_names[0]))


def _identity(wheel: Path) -> tuple[str, str]:
    metadata = _metadata(wheel)
    name = metadata["Name"]
    version = metadata["Version"]
    if name is None or version is None:
        raise RuntimeError(f"Expected Name and Version metadata in {wheel}")
    return _normalize_distribution(name), version


def _logical_contents(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
        }


def _release_artifacts(source: Path) -> list[Path]:
    if not source.is_dir():
        raise RuntimeError(f"Wheel artifact directory does not exist: {source}")
    return [
        path
        for path in sorted(source.iterdir())
        if path.is_dir() and _RELEASE_ARTIFACT.fullmatch(path.name) is not None
    ]


def _selected_wheels(source: Path, distribution: str, version: str) -> list[Path]:
    release_tag(distribution, version)
    selected: list[Path] = []
    found_versions: set[str] = set()
    artifacts = _release_artifacts(source)
    if not artifacts:
        raise RuntimeError(f"No release wheel artifacts found below {source}")
    for artifact in artifacts:
        artifact_wheels: list[Path] = []
        for wheel in sorted(artifact.rglob("*.whl")):
            wheel_distribution, wheel_version = _identity(wheel)
            if wheel_distribution != distribution:
                continue
            found_versions.add(wheel_version)
            if wheel_version == version:
                artifact_wheels.append(wheel)
        if len(artifact_wheels) != 1:
            raise RuntimeError(
                f"Expected one {distribution} {version} wheel in {artifact.name}, "
                f"found {len(artifact_wheels)}"
            )
        selected.extend(artifact_wheels)
    if found_versions != {version}:
        raise RuntimeError(
            f"Expected only {distribution} {version} artifacts, "
            f"found versions {sorted(found_versions)}"
        )
    return selected


def stage_wheels(
    source: Path,
    destination: Path,
    distribution: str,
    version: str,
) -> None:
    selected = _selected_wheels(source, distribution, version)
    if not selected:
        raise RuntimeError(f"No {distribution} {version} wheels found below {source}")
    if source.resolve() == destination.resolve():
        raise RuntimeError("Source and destination must be different directories")

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"Destination must be empty: {destination}")

    by_name: dict[str, list[Path]] = {}
    for wheel in selected:
        by_name.setdefault(wheel.name, []).append(wheel)

    for name, candidates in sorted(by_name.items()):
        selected_wheel = candidates[0]
        selected_contents = _logical_contents(selected_wheel)
        for duplicate in candidates[1:]:
            if _logical_contents(duplicate) != selected_contents:
                locations = "\n  ".join(str(path) for path in candidates)
                raise RuntimeError(
                    f"Conflicting wheels share the filename {name}:\n  {locations}"
                )
        shutil.copy2(selected_wheel, destination / name)
        print(f"Staged {selected_wheel}")

    staged = sorted(destination.glob("*.whl"))
    if distribution == "cuda-cccl" and (
        len(staged) != 1 or not staged[0].name.endswith("-py3-none-any.whl")
    ):
        raise RuntimeError("Expected one universal cuda-cccl wheel")


def exact_dependency_version(
    wheelhouse: Path,
    distribution: str,
    dependency: str,
) -> str:
    if distribution not in _DISTRIBUTIONS or dependency not in _DISTRIBUTIONS:
        raise RuntimeError("Unsupported distribution or dependency")
    dependency = _normalize_distribution(dependency)
    wheels = sorted(wheelhouse.glob("*.whl"))
    matching_wheels = [wheel for wheel in wheels if _identity(wheel)[0] == distribution]
    if len(matching_wheels) != 1:
        raise RuntimeError(
            f"Expected one staged {distribution} wheel, found {len(matching_wheels)}"
        )

    requirements = _metadata(matching_wheels[0]).get_all("Requires-Dist", [])
    exact_versions: list[str] = []
    for requirement in requirements:
        requirement_text, separator, _marker = requirement.partition(";")
        name_match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement_text)
        if (
            name_match is None
            or _normalize_distribution(name_match.group(1)) != dependency
        ):
            continue
        if separator:
            continue
        exact_match = re.fullmatch(
            r"\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*(?:\(\s*)?==\s*"
            r"([^\s,()]+)\s*\)?\s*",
            requirement_text,
        )
        if exact_match is None:
            raise RuntimeError(
                f"Expected an exact {dependency} dependency, found {requirement!r}"
            )
        exact_versions.append(exact_match.group(1))

    if len(exact_versions) != 1:
        raise RuntimeError(
            f"Expected one exact unconditional {dependency} dependency, "
            f"found {exact_versions}"
        )
    release_tag(dependency, exact_versions[0])
    return exact_versions[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tag_parser = subparsers.add_parser("tag", help="print the release tag")
    tag_parser.add_argument("distribution", choices=sorted(_DISTRIBUTIONS))
    tag_parser.add_argument("version")

    stage_parser = subparsers.add_parser("stage", help="stage one distribution")
    stage_parser.add_argument("source", type=Path)
    stage_parser.add_argument("destination", type=Path)
    stage_parser.add_argument("distribution", choices=sorted(_DISTRIBUTIONS))
    stage_parser.add_argument("version")

    dependency_parser = subparsers.add_parser(
        "dependency-version", help="print one exact unconditional dependency version"
    )
    dependency_parser.add_argument("wheelhouse", type=Path)
    dependency_parser.add_argument("distribution", choices=sorted(_DISTRIBUTIONS))
    dependency_parser.add_argument("dependency", choices=sorted(_DISTRIBUTIONS))

    args = parser.parse_args()
    try:
        if args.command == "tag":
            print(release_tag(args.distribution, args.version))
        elif args.command == "stage":
            stage_wheels(
                args.source,
                args.destination,
                args.distribution,
                args.version,
            )
        else:
            print(
                exact_dependency_version(
                    args.wheelhouse,
                    args.distribution,
                    args.dependency,
                )
            )
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
