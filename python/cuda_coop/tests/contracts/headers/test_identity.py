# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

import pytest

from cuda.coop._headers import _identity
from cuda.coop._headers._identity import include_dirs_identity


def _record_hash(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return f"sha256={encoded.rstrip(b'=').decode('ascii')}"


@pytest.mark.parametrize("mutation", ["content", "addition", "deletion"])
def test_installed_record_never_replaces_live_header_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    site_packages = tmp_path / "site-packages"
    include = site_packages / "cuda" / "cccl" / "include"
    include.mkdir(parents=True)
    header = include / "header.cuh"
    original_content = b"first"
    header.write_bytes(original_content)

    dist_info = site_packages / "cuda_cccl-1.0.dist-info"
    dist_info.mkdir()
    record = dist_info / "RECORD"
    relative_header = header.relative_to(site_packages).as_posix()
    record.write_text(
        f"{relative_header},{_record_hash(original_content)},"
        f"{len(original_content)}\n{dist_info.name}/RECORD,,\n",
        encoding="utf-8",
    )

    original_stat = header.stat()
    original = include_dirs_identity((str(include),))
    if mutation == "content":
        header.write_bytes(b"other")
        os.utime(
            header,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
    elif mutation == "addition":
        (include / "added.cuh").write_bytes(b"added")
    else:
        header.unlink()
    mutated = include_dirs_identity((str(include),))

    assert original.roots[0].method == "recursive-content"
    assert mutated.roots[0].method == "recursive-content"
    assert original.digest != mutated.digest


def test_missing_git_executable_falls_back_to_recursive_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    include = repository / "include"
    include.mkdir()
    (include / "header.cuh").write_bytes(b"header")

    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(_identity.subprocess, "run", missing_git)

    identity = include_dirs_identity((str(include),))

    assert identity.roots[0].method == "recursive-content"
    assert identity.recursive_walks == 1
