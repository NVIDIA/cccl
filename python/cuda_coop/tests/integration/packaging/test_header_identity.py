# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cuda.coop._headers import _identity as header_identity


def _record_hash(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return f"sha256={encoded.rstrip(b'=').decode('ascii')}"


def _write_record_distribution(
    site_packages: Path,
    *,
    name: str,
    version: str,
    files: tuple[Path, ...],
    editable: bool = False,
) -> Path:
    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Name: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    rows = []
    for path in files:
        content = path.read_bytes()
        rows.append(
            f"{path.relative_to(site_packages).as_posix()},"
            f"{_record_hash(content)},{len(content)}"
        )
    rows.append(f"{dist_info.name}/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if editable:
        (dist_info / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "file:///source",
                    "dir_info": {"editable": True},
                }
            ),
            encoding="utf-8",
        )
    return dist_info


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _git_status(repo: Path) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout


def _symlink_to_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip(
                "Windows symlink creation requires Developer Mode or "
                "SeCreateSymbolicLinkPrivilege"
            )
        raise


def test_multi_owner_record_identity_avoids_recursive_walk(monkeypatch, tmp_path):
    site_packages = tmp_path / "site-packages"
    include = site_packages / "nvidia" / "cu13" / "include"
    include.mkdir(parents=True)
    runtime = include / "cuda_runtime.h"
    extensionless = include / "nv"
    runtime.write_bytes(b"runtime")
    extensionless.write_bytes(b"extensionless")
    _write_record_distribution(
        site_packages,
        name="nvidia-runtime",
        version="13.0",
        files=(runtime,),
    )
    _write_record_distribution(
        site_packages,
        name="nvidia-nvcc",
        version="13.0",
        files=(extensionless,),
    )

    def forbidden(_root):
        raise AssertionError("complete wheel RECORDs must avoid recursive walks")

    monkeypatch.setattr(
        header_identity,
        "_recursive_include_root_identity",
        forbidden,
    )
    first = header_identity.include_dirs_identity((str(include),))
    second = header_identity.include_dirs_identity((str(include),))

    assert first.digest == second.digest
    assert first.recursive_walks == second.recursive_walks == 0
    assert [root.method for root in first.roots] == ["pep376-record"]
    assert first.duration_ns >= sum(root.duration_ns for root in first.roots)


def test_installed_wheel_identity_is_metadata_only_and_bounded():
    try:
        distribution = importlib.metadata.distribution("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("cuda-coop is not installed from a wheel")
    include = Path(
        str(distribution.locate_file("cuda/coop/_headers/include"))
    ).resolve()
    if not (include / "cub/version.cuh").is_file():
        pytest.skip("the installed cuda-coop distribution has no header bundle")

    identity = header_identity.include_dirs_identity((str(include),))
    if any(root.method != "pep376-record" for root in identity.roots):
        pytest.skip("cuda-coop is not installed from an immutable wheel")

    assert identity.recursive_walks == 0
    # This generous bound catches accidental recursive-content regression
    # without turning ordinary shared-runner jitter into a timing failure.
    assert identity.duration_ns < 250_000_000


def test_record_owner_version_hash_and_order_invalidate_identity(tmp_path):
    site_packages = tmp_path / "site-packages"
    include_a = site_packages / "cuda" / "coop" / "_headers" / "include"
    include_b = site_packages / "nvidia" / "cu13" / "include"
    include_a.mkdir(parents=True)
    include_b.mkdir(parents=True)
    header_a = include_a / "cub.cuh"
    header_b = include_b / "cuda_runtime.h"
    header_a.write_bytes(b"a")
    header_b.write_bytes(b"b")
    owner_a = _write_record_distribution(
        site_packages,
        name="cuda-coop",
        version="3.5",
        files=(header_a,),
    )
    _write_record_distribution(
        site_packages,
        name="nvidia-runtime",
        version="13.0",
        files=(header_b,),
    )

    original = header_identity.include_dirs_identity((str(include_a), str(include_b)))
    reversed_roots = header_identity.include_dirs_identity(
        (str(include_b), str(include_a))
    )
    owner_a.rename(site_packages / "cuda_coop-3.6.dist-info")
    new_version = header_identity.include_dirs_identity(
        (str(include_a), str(include_b))
    )

    assert original.digest != reversed_roots.digest
    assert original.digest != new_version.digest


def test_unhashed_or_editable_owner_falls_back_to_content(tmp_path):
    site_packages = tmp_path / "site-packages"
    include = site_packages / "cuda" / "coop" / "_headers" / "include"
    include.mkdir(parents=True)
    header = include / "extensionless"
    header.write_bytes(b"first")
    dist_info = _write_record_distribution(
        site_packages,
        name="cuda-coop",
        version="3.5",
        files=(header,),
    )
    record = dist_info / "RECORD"
    record.write_text(
        f"{header.relative_to(site_packages).as_posix()},,\n",
        encoding="utf-8",
    )

    unhashed = header_identity.include_dirs_identity((str(include),))
    header.write_bytes(b"other")
    mutated = header_identity.include_dirs_identity((str(include),))
    assert unhashed.roots[0].method == "recursive-content"
    assert unhashed.digest != mutated.digest

    record.write_text(
        f"{header.relative_to(site_packages).as_posix()},"
        f"{_record_hash(header.read_bytes())},{header.stat().st_size}\n",
        encoding="utf-8",
    )
    (dist_info / "direct_url.json").write_text(
        '{"url":"file:///source","dir_info":{"editable":true}}',
        encoding="utf-8",
    )
    editable = header_identity.include_dirs_identity((str(include),))
    assert editable.roots[0].method == "recursive-content"


def test_recursive_identity_hashes_extensionless_content_and_symlink_referent(
    tmp_path,
):
    include = tmp_path / "include"
    external = tmp_path / "external"
    include.mkdir()
    external.mkdir()
    extensionless = include / "header"
    referent = external / "target.h"
    extensionless.write_bytes(b"abcd")
    referent.write_bytes(b"first")
    _symlink_to_or_skip(include / "linked.h", referent)

    initial_stat = extensionless.stat()
    first = header_identity.include_dirs_identity((str(include),))
    extensionless.write_bytes(b"wxyz")
    os.utime(
        extensionless,
        ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
    )
    content_changed = header_identity.include_dirs_identity((str(include),))
    referent.write_bytes(b"other")
    referent_changed = header_identity.include_dirs_identity((str(include),))

    assert first.roots[0].method == "recursive-content"
    assert first.digest != content_changed.digest
    assert content_changed.digest != referent_changed.digest


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_clean_git_tree_is_metadata_only_and_flags_force_fallback(
    monkeypatch,
    tmp_path,
):
    repo = tmp_path / "repo"
    include = repo / "include"
    include.mkdir(parents=True)
    header = include / "extensionless"
    header.write_bytes(b"tracked")
    (repo / ".gitignore").write_text("include/ignored\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "headers")

    original_recursive = header_identity._recursive_include_root_identity

    def forbidden(_root):
        raise AssertionError("clean Git roots must avoid recursive walks")

    monkeypatch.setattr(
        header_identity,
        "_recursive_include_root_identity",
        forbidden,
    )
    clean = header_identity.include_dirs_identity((str(include),))
    assert clean.roots[0].method == "git-tree"
    assert clean.recursive_walks == 0

    monkeypatch.setattr(
        header_identity,
        "_recursive_include_root_identity",
        original_recursive,
    )
    header.write_bytes(b"changed")
    _git(repo, "add", "include/extensionless")
    _git(repo, "commit", "-m", "change header")
    committed = header_identity.include_dirs_identity((str(include),))
    assert committed.roots[0].method == "git-tree"
    assert committed.digest != clean.digest

    _git(repo, "update-index", "--assume-unchanged", "include/extensionless")
    assumed = header_identity.include_dirs_identity((str(include),))
    assert assumed.roots[0].method == "recursive-content"
    _git(repo, "update-index", "--no-assume-unchanged", "include/extensionless")

    _git(repo, "update-index", "--skip-worktree", "include/extensionless")
    skipped = header_identity.include_dirs_identity((str(include),))
    assert skipped.roots[0].method == "recursive-content"
    _git(repo, "update-index", "--no-skip-worktree", "include/extensionless")

    ignored = include / "ignored"
    ignored.write_bytes(b"one")
    first_ignored = header_identity.include_dirs_identity((str(include),))
    ignored.write_bytes(b"two")
    second_ignored = header_identity.include_dirs_identity((str(include),))
    assert first_ignored.roots[0].method == "recursive-content"
    assert first_ignored.digest != second_ignored.digest


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
@pytest.mark.parametrize("referent_location", ["external", "tracked-outside-root"])
def test_clean_git_symlink_falls_back_and_tracks_referent(
    tmp_path,
    referent_location,
):
    repo = tmp_path / "repo"
    include = repo / "include"
    safe_include = repo / "safe-include"
    include.mkdir(parents=True)
    safe_include.mkdir()
    (safe_include / "direct.h").write_bytes(b"tracked")
    if referent_location == "external":
        referent = tmp_path / "external" / "target.h"
    else:
        referent = repo / "shared" / "target.h"
    referent.parent.mkdir()
    referent.write_bytes(b"first")
    _symlink_to_or_skip(include / "linked.h", referent)

    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "headers")

    assert _git_status(repo) == b""
    include_dirs = (str(include), str(safe_include))
    first = header_identity.include_dirs_identity(include_dirs)

    referent.write_bytes(b"other")
    if referent_location == "tracked-outside-root":
        _git(repo, "add", "shared/target.h")
        _git(repo, "commit", "-m", "change shared header")

    assert _git_status(repo) == b""
    changed = header_identity.include_dirs_identity(include_dirs)

    assert [root.method for root in first.roots] == [
        "recursive-content",
        "git-tree",
    ]
    assert first.recursive_walks == 1
    assert [root.method for root in changed.roots] == [
        "recursive-content",
        "git-tree",
    ]
    assert changed.recursive_walks == 1
    assert first.digest != changed.digest
