#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect wheel artifacts without silently overwriting duplicate filenames."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

_RELEASE_ARTIFACT = re.compile(r"wheel-cccl-(?:linux|windows)-[^/]+-py[^/]+")


def _logical_contents(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        contents = {}
        for name in sorted(archive.namelist()):
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            contents[name] = digest.hexdigest()
        return contents


def _artifact_name(source: Path, wheel: Path) -> str | None:
    relative_parts = wheel.relative_to(source).parts
    return relative_parts[0] if len(relative_parts) > 1 else None


def _is_release_artifact(artifact_name: str | None) -> bool:
    return bool(artifact_name and _RELEASE_ARTIFACT.fullmatch(artifact_name))


def _is_universal_wheel(name: str) -> bool:
    return name.endswith("-py3-none-any.whl")


def collect_wheels(
    source: Path,
    destination: Path,
    canonical_artifact: str,
) -> None:
    discovered_wheels = sorted(source.rglob("*.whl"))
    wheels = [
        wheel
        for wheel in discovered_wheels
        if _is_release_artifact(_artifact_name(source, wheel))
    ]
    if not wheels:
        raise RuntimeError(f"No release wheel artifacts found below {source}")

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
        )
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
