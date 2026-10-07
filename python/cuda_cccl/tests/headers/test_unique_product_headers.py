# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GCC's ``#pragma once`` treats byte-identical headers as the same file.

Including a second copy is skipped, which drops the header and breaks sccache.
Every shipped CCCL header must have distinct contents.
"""

import hashlib
from collections import defaultdict
from pathlib import Path

# Directories installed by cccl_generate_install_rules.
_PRODUCT_HEADER_DIRS = (
    ("cub", "cub"),
    ("thrust", "thrust"),
    ("libcudacxx", "include", "cuda"),
    ("libcudacxx", "include", "nv"),
    ("cudax", "include", "cuda"),
)
_HEADER_SUFFIXES = {".h", ".cuh", ".hpp", ".hh", ".hxx", ".inl", ""}
_IGNORED_NAMES = {"CMakeLists.txt"}


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        version = parent / "thrust" / "thrust" / "version.h"
        cub_version = parent / "cub" / "cub" / "version.cuh"
        if version.is_file() and cub_version.is_file():
            return parent
    raise RuntimeError("Could not locate the CCCL repository root")


def _is_product_header(path: Path) -> bool:
    return (
        path.is_file()
        and path.name not in _IGNORED_NAMES
        and path.suffix in _HEADER_SUFFIXES
    )


def _iter_product_headers(repo: Path) -> list[Path]:
    headers: list[Path] = []
    seen: set[Path] = set()
    for parts in _PRODUCT_HEADER_DIRS:
        root = repo.joinpath(*parts)
        assert root.is_dir(), f"missing product header directory: {root}"
        for path in root.rglob("*"):
            if not _is_product_header(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            headers.append(path)
    return headers


def _digest(path: Path) -> bytes:
    hasher = hashlib.sha256()
    with path.open("rb") as header:
        for chunk in iter(lambda: header.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.digest()


def _byte_identical_groups(repo: Path, headers: list[Path]) -> list[list[str]]:
    grouped: dict[bytes, list[str]] = defaultdict(list)
    for header in headers:
        grouped[_digest(header)].append(header.relative_to(repo).as_posix())
    return [sorted(paths) for paths in grouped.values() if len(paths) > 1]


def test_product_headers_are_not_byte_identical():
    repo = _repo_root()
    headers = _iter_product_headers(repo)
    assert headers, "no product headers found"

    duplicates = _byte_identical_groups(repo, headers)
    assert not duplicates, "byte-identical product headers:\n" + "\n".join(
        "\n".join(f"  {path}" for path in group) + "\n" for group in duplicates
    )
