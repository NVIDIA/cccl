# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from cuda.coop._headers._identity import (
    HeaderIdentityError,
    include_dirs_identity,
)


def _include_root(path: Path, content: bytes) -> Path:
    path.mkdir()
    (path / "primitive.cuh").write_bytes(content)
    return path


def test_identity_preserves_include_order_and_provenance(tmp_path: Path) -> None:
    first = _include_root(tmp_path / "first", b"same header")
    second = _include_root(tmp_path / "second", b"same header")

    forward = include_dirs_identity((str(first), str(second)))
    reverse = include_dirs_identity((str(second), str(first)))
    first_only = include_dirs_identity((str(first),))
    second_only = include_dirs_identity((str(second),))

    assert tuple(root.path for root in forward.roots) == (
        str(first.resolve()),
        str(second.resolve()),
    )
    assert all(root.method == "recursive-content" for root in forward.roots)
    assert forward.digest != reverse.digest
    assert first_only.roots[0].digest == second_only.roots[0].digest
    assert first_only.digest != second_only.digest


def test_identity_changes_with_header_content(tmp_path: Path) -> None:
    include = _include_root(tmp_path / "include", b"before")
    before = include_dirs_identity((str(include),))

    (include / "primitive.cuh").write_bytes(b"after")
    after = include_dirs_identity((str(include),))

    assert before.roots[0].digest != after.roots[0].digest
    assert before.digest != after.digest


def test_recursive_identity_frames_file_contents_unambiguously(
    tmp_path: Path,
) -> None:
    combined = tmp_path / "combined"
    combined.mkdir()
    (combined / "a").write_bytes(b"A\0" + b"1:b\0file\0B")

    split = tmp_path / "split"
    split.mkdir()
    (split / "a").write_bytes(b"A")
    (split / "b").write_bytes(b"B")

    combined_identity = include_dirs_identity((str(combined),))
    split_identity = include_dirs_identity((str(split),))

    assert combined_identity.roots[0].method == "recursive-content"
    assert split_identity.roots[0].method == "recursive-content"
    assert combined_identity.roots[0].digest != split_identity.roots[0].digest


def test_identity_rejects_missing_include_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(HeaderIdentityError, match="not a directory"):
        include_dirs_identity((str(missing),))
