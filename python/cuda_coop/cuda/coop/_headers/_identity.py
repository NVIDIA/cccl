# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provenance-based identities for ordered compiler include roots."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class HeaderIdentityError(RuntimeError):
    """Raised when an include root cannot be fingerprinted safely."""


@dataclass(frozen=True)
class IncludeRootIdentity:
    """Identity and provenance for one resolved include root."""

    path: str
    method: str
    digest: str
    duration_ns: int


@dataclass(frozen=True)
class IncludeDirsIdentity:
    """Ordered identity of every CCCL and CUDA include root."""

    roots: tuple[IncludeRootIdentity, ...]
    digest: str
    recursive_walks: int
    duration_ns: int


def _find_git_root(root: Path) -> Path | None:
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "--literal-pathspecs", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def _git_include_root_identities(
    repo: Path,
    roots: tuple[Path, ...],
) -> dict[Path, str]:
    relatives = tuple(root.relative_to(repo) for root in roots)
    pathspecs = tuple(
        "." if not relative.parts else relative.as_posix() for relative in relatives
    )
    status = _run_git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--ignore-submodules=none",
        "--",
        *pathspecs,
    )
    if status.returncode != 0 or status.stdout:
        return {}

    index_entries = _run_git(repo, "ls-files", "-v", "-z", "--", *pathspecs)
    if index_entries.returncode != 0 or not index_entries.stdout:
        return {}
    for entry in index_entries.stdout.split(b"\0"):
        if entry and entry[:1] != b"H":
            # `S` marks skip-worktree; lowercase marks assume-unchanged.
            return {}

    staged_entries = _run_git(repo, "ls-files", "--stage", "-z", "--", *pathspecs)
    if staged_entries.returncode != 0 or not staged_entries.stdout:
        return {}
    symlinked_roots: set[Path] = set()
    for entry in staged_entries.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, separator, path = entry.partition(b"\t")
        fields = metadata.split()
        if separator != b"\t" or len(fields) != 3 or fields[2] != b"0":
            return {}
        if fields[0] != b"120000":
            continue
        tracked_path = path.decode("utf-8", errors="surrogateescape")
        for root, relative in zip(roots, relatives):
            relative_path = relative.as_posix()
            if (
                not relative.parts
                or tracked_path == relative_path
                or tracked_path.startswith(f"{relative_path}/")
            ):
                symlinked_roots.add(root)

    result: dict[Path, str] = {}
    non_root_relatives = tuple(relative for relative in relatives if relative.parts)
    tree_oids: dict[str, bytes] = {}
    if non_root_relatives:
        tree = _run_git(
            repo,
            "ls-tree",
            "-d",
            "-z",
            "HEAD",
            "--",
            *(relative.as_posix() for relative in non_root_relatives),
        )
        if tree.returncode != 0:
            return {}
        for record in tree.stdout.split(b"\0"):
            if not record:
                continue
            metadata, separator, path = record.partition(b"\t")
            fields = metadata.split()
            if separator != b"\t" or len(fields) != 3 or fields[1] != b"tree":
                return {}
            tree_oids[path.decode("utf-8", errors="surrogateescape")] = fields[2]

    repository_tree: bytes | None = None
    if any(not relative.parts for relative in relatives):
        tree = _run_git(repo, "rev-parse", "--verify", "HEAD^{tree}")
        if tree.returncode != 0:
            return {}
        repository_tree = tree.stdout.strip()

    for root, relative in zip(roots, relatives):
        tree_oid = (
            repository_tree
            if not relative.parts
            else tree_oids.get(relative.as_posix())
        )
        if (
            tree_oid is None
            or len(tree_oid) not in (40, 64)
            or re.fullmatch(rb"[0-9a-f]+", tree_oid) is None
        ):
            return {}
        if root in symlinked_roots:
            continue
        digest = hashlib.sha256()
        digest.update(b"git-tree-v1\0")
        digest.update(tree_oid)
        result[root] = digest.hexdigest()
    return result


def _dist_info_is_editable(dist_info: Path) -> bool:
    try:
        direct_url = (dist_info / "direct_url.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if not direct_url:
        return False
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return True
    dir_info = payload.get("dir_info")
    return isinstance(dir_info, dict) and dir_info.get("editable") is True


def _record_owner_candidates(root: Path) -> tuple[Path, ...]:
    for installed_root in root.parents:
        try:
            relative = root.relative_to(installed_root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        top_level = relative.parts[0].replace("-", "_")
        candidates = tuple(sorted(installed_root.glob(f"{top_level}*.dist-info")))
        if candidates:
            return candidates
    return ()


def _record_entries_for_dist_info(
    root: Path,
    dist_info: Path,
) -> tuple[bool, tuple[tuple[str, str, str], ...] | None]:
    if (
        not dist_info.is_dir()
        or dist_info.is_symlink()
        or _dist_info_is_editable(dist_info)
    ):
        return False, None
    installed_root = dist_info.parent.resolve()
    try:
        root_prefix = root.relative_to(installed_root)
    except ValueError:
        return False, None
    try:
        if (installed_root / root_prefix).resolve() != root:
            return False, None
        record = (dist_info / "RECORD").read_bytes()
    except OSError:
        return False, None
    if not record:
        return False, None

    prefix = root_prefix.as_posix().rstrip("/")
    prefix_with_separator = f"{prefix}/"
    entries: list[tuple[str, str, str]] = []
    try:
        rows = csv.reader(record.decode("utf-8").splitlines())
    except UnicodeError:
        return False, None
    try:
        for row in rows:
            record_path = row[0].replace("\\", "/") if row else ""
            if record_path == prefix:
                relative = ""
            elif record_path.startswith(prefix_with_separator):
                relative = record_path[len(prefix_with_separator) :]
            else:
                continue
            if len(row) != 3:
                return True, None
            relative_parts = relative.split("/")
            if "__pycache__" in relative_parts or relative.endswith(".pyc"):
                continue
            record_hash = row[1]
            if (
                not relative
                or relative.startswith("/")
                or ".." in relative_parts
                or re.fullmatch(r"sha256=[A-Za-z0-9_-]{43}", record_hash) is None
            ):
                return True, None
            entries.append((relative, record_hash, row[2]))
    except csv.Error:
        return True, None
    if not entries:
        return False, None
    return True, tuple(sorted(entries))


def _wheel_record_include_root_identity(
    root: Path,
    dist_infos: Iterable[Path] | None = None,
) -> str | None:
    if dist_infos is None:
        dist_infos = _record_owner_candidates(root)
    owners: list[tuple[str, tuple[tuple[str, str, str], ...]]] = []
    for dist_info in dist_infos:
        matches, entries = _record_entries_for_dist_info(root, dist_info)
        if matches and entries is None:
            return None
        if entries is not None:
            owners.append((dist_info.name, entries))
    if not owners:
        return None

    paths: dict[str, tuple[str, str]] = {}
    for _owner, entries in owners:
        for relative, record_hash, size in entries:
            value = (record_hash, size)
            existing = paths.get(relative)
            if existing is not None:
                return None
            paths[relative] = value

    digest = hashlib.sha256()
    digest.update(b"pep376-record-v1\0")
    for owner, entries in sorted(owners):
        digest.update(owner.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        for relative, record_hash, size in entries:
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(record_hash.encode("ascii"))
            digest.update(b"\0")
            digest.update(size.encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _recursive_include_root_identity(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"recursive-content-v2\0")

    def update_path(relative: Path) -> None:
        encoded = relative.as_posix().encode("utf-8", errors="surrogateescape")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")

    def hash_file(path: Path) -> None:
        with path.open("rb") as header:
            while chunk := header.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")

    def visit(
        directory: Path,
        relative: Path,
        ancestors: frozenset[Path],
    ) -> None:
        resolved_directory = directory.resolve()
        if resolved_directory in ancestors:
            digest.update(b"directory-cycle\0")
            return
        ancestors = ancestors | {resolved_directory}
        with os.scandir(directory) as scanned:
            entries = sorted(scanned, key=lambda entry: os.fsencode(entry.name))
        for entry in entries:
            if entry.name in {".git", "__pycache__"}:
                continue
            entry_path = Path(entry.path)
            entry_relative = relative / entry.name
            update_path(entry_relative)
            if entry.is_symlink():
                digest.update(b"symlink\0")
                target = os.readlink(entry.path)
                digest.update(
                    target.encode("utf-8", errors="surrogateescape")
                    if isinstance(target, str)
                    else target
                )
                digest.update(b"\0")
                try:
                    target_path = entry_path.resolve(strict=True)
                except OSError:
                    digest.update(b"broken\0")
                    continue
                if target_path.is_dir():
                    digest.update(b"target-directory\0")
                    visit(target_path, entry_relative, ancestors)
                elif target_path.is_file():
                    digest.update(b"target-file\0")
                    hash_file(target_path)
                else:
                    raise HeaderIdentityError(
                        f"Unsupported special include path: {target_path}"
                    )
            elif entry.is_dir(follow_symlinks=False):
                digest.update(b"directory\0")
                visit(entry_path, entry_relative, ancestors)
            elif entry.is_file(follow_symlinks=False):
                digest.update(b"file\0")
                hash_file(entry_path)
            else:
                raise HeaderIdentityError(
                    f"Unsupported special include path: {entry_path}"
                )

    visit(root, Path(), frozenset())
    return digest.hexdigest()


def include_dirs_identity(include_dirs: Iterable[str]) -> IncludeDirsIdentity:
    started_ns = time.perf_counter_ns()
    resolved_roots = tuple(
        Path(include_dir).expanduser().resolve() for include_dir in include_dirs
    )
    for root in resolved_roots:
        if not root.is_dir():
            raise HeaderIdentityError(
                f"Provider include path is not a directory: {root}"
            )
    if not resolved_roots:
        digest = hashlib.sha256(b"ordered-include-roots-v2\0none").hexdigest()
        return IncludeDirsIdentity(
            roots=(),
            digest=digest,
            recursive_walks=0,
            duration_ns=max(0, time.perf_counter_ns() - started_ns),
        )

    record_identities: dict[Path, str] = {}
    root_durations_ns: dict[Path, int] = {}
    for root in resolved_roots:
        root_started_ns = time.perf_counter_ns()
        record_identity = _wheel_record_include_root_identity(root)
        root_durations_ns[root] = max(
            0,
            time.perf_counter_ns() - root_started_ns,
        )
        if record_identity is not None:
            record_identities[root] = record_identity

    roots_by_repo: dict[Path, list[Path]] = {}
    for root in resolved_roots:
        if root in record_identities:
            continue
        if (repo := _find_git_root(root)) is not None:
            roots_by_repo.setdefault(repo, []).append(root)
    git_identities: dict[Path, str] = {}
    for repo, roots in roots_by_repo.items():
        git_started_ns = time.perf_counter_ns()
        git_identities.update(_git_include_root_identities(repo, tuple(roots)))
        git_duration_ns = max(0, time.perf_counter_ns() - git_started_ns)
        per_root_duration_ns = git_duration_ns // len(roots)
        for root in roots:
            root_durations_ns[root] += per_root_duration_ns

    identities: list[IncludeRootIdentity] = []
    recursive_walks = 0
    for root in resolved_roots:
        method = "pep376-record"
        root_digest = record_identities.get(root)
        if root_digest is None:
            method = "git-tree"
            root_digest = git_identities.get(root)
        if root_digest is None:
            method = "recursive-content"
            recursive_walks += 1
            recursive_started_ns = time.perf_counter_ns()
            try:
                root_digest = _recursive_include_root_identity(root)
            except OSError as exc:
                raise HeaderIdentityError(
                    f"Failed fingerprinting provider include path: {root}"
                ) from exc
            root_durations_ns[root] += max(
                0,
                time.perf_counter_ns() - recursive_started_ns,
            )
        identities.append(
            IncludeRootIdentity(
                path=str(root),
                method=method,
                digest=root_digest,
                duration_ns=root_durations_ns[root],
            )
        )

    digest = hashlib.sha256()
    digest.update(b"ordered-include-roots-v2\0")
    for index, identity in enumerate(identities):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(identity.path.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(identity.method.encode("ascii"))
        digest.update(b"\0")
        digest.update(identity.digest.encode("ascii"))
        digest.update(b"\0")
    return IncludeDirsIdentity(
        roots=tuple(identities),
        digest=digest.hexdigest(),
        recursive_walks=recursive_walks,
        duration_ns=max(0, time.perf_counter_ns() - started_ns),
    )


__all__ = [
    "HeaderIdentityError",
    "IncludeDirsIdentity",
    "IncludeRootIdentity",
    "include_dirs_identity",
]
