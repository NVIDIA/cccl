# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Provider artifact cache paths, atomic I/O, and metadata validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from ._bundle_contract import (
    BUNDLE_METADATA_VERSION,
    BundleCacheIdentity,
    StorageLayout,
    _validate_storage_layout,
)

CACHE_DIR_ENV = "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR"

_SOURCE_CACHE: dict[str, _CachedBundle] = {}
_MANAGED_BUNDLE_PATHS: set[str] = set()
_COMPILE_COUNTER = 0
_STATE_LOCK = threading.RLock()
_ARTIFACT_LOCKS: dict[str, threading.RLock] = {}


@dataclass(frozen=True)
class _CachedBundle:
    path: str
    layouts_by_expression: dict[str, StorageLayout]
    producer_compiler: str | None = None
    producer_compiler_version: str | None = None
    producer_toolkit_version: str | None = None


def _local_artifact_lock(path: str) -> threading.RLock:
    real_path = os.path.realpath(path)
    with _STATE_LOCK:
        return _ARTIFACT_LOCKS.setdefault(real_path, threading.RLock())


@contextmanager
def artifact_lock(path: str, *, scope: str):
    """Serialize one cache artifact across threads and processes."""

    lock_path = f"{path}.lock"
    local_lock = _local_artifact_lock(path)
    with local_lock:
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(lock_path, flags, 0o600)
            lock_stat = os.fstat(fd)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise OSError("provider artifact lock is not a regular file")
            getuid = getattr(os, "getuid", None)
            if getuid is not None and lock_stat.st_uid != getuid():
                raise OSError("provider artifact lock is not owned by this user")
            if os.name == "nt":
                if lock_stat.st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            if "fd" in locals():
                os.close(fd)
            raise DSLRuntimeError(
                f"Failed locking {scope} provider cache artifact.",
                cause=exc,
            ) from exc
        try:
            yield
        finally:
            try:
                if os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def memory_cached_bundle(cache_key: str) -> _CachedBundle | None:
    with _STATE_LOCK:
        cached = _SOURCE_CACHE.get(cache_key)
    if cached is not None and os.path.exists(cached.path):
        return cached
    return None


def store_memory_bundle(cache_key: str, cached: _CachedBundle) -> None:
    with _STATE_LOCK:
        _SOURCE_CACHE[cache_key] = cached


def record_compilation() -> None:
    global _COMPILE_COUNTER
    with _STATE_LOCK:
        _COMPILE_COUNTER += 1


def reset_compile_state() -> None:
    global _COMPILE_COUNTER
    with _STATE_LOCK:
        _COMPILE_COUNTER = 0
        _SOURCE_CACHE.clear()


def get_compile_counter() -> int:
    with _STATE_LOCK:
        return _COMPILE_COUNTER


def add_managed_bundle_path(path: str) -> None:
    with _STATE_LOCK:
        _MANAGED_BUNDLE_PATHS.add(os.path.realpath(path))


def managed_bundle_paths() -> frozenset[str]:
    with _STATE_LOCK:
        return frozenset(_MANAGED_BUNDLE_PATHS)


def _cache_dir_name() -> str:
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return "cuda_coop_cutlass_provider"
    return f"cuda_coop_cutlass_provider_{getuid()}"


_CACHE_DIR = os.path.join(tempfile.gettempdir(), _cache_dir_name())


def configured_cache_dir() -> str:
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if cache_dir:
        return os.path.abspath(os.path.expanduser(cache_dir))
    return _CACHE_DIR


def ensure_cache_dir(scope: str) -> str:
    cache_dir = configured_cache_dir()
    try:
        try:
            os.mkdir(cache_dir, mode=0o700)
        except FileExistsError:
            pass
        cache_stat = os.lstat(cache_dir)
        if stat.S_ISLNK(cache_stat.st_mode) or not stat.S_ISDIR(cache_stat.st_mode):
            raise DSLRuntimeError(
                f"{scope} provider cache path is not a real directory."
            )
        getuid = getattr(os, "getuid", None)
        if getuid is not None and cache_stat.st_uid != getuid():
            raise DSLRuntimeError(
                f"{scope} provider cache directory is not owned by this user."
            )
        if stat.S_IMODE(cache_stat.st_mode) != 0o700:
            os.chmod(cache_dir, 0o700)
    except OSError as exc:
        raise DSLRuntimeError(
            f"Failed preparing {scope} provider cache directory.",
            cause=exc,
        ) from exc
    return cache_dir


def write_binary_atomic(path: str, blob: bytes | bytearray, *, scope: str) -> None:
    cache_dir = ensure_cache_dir(scope)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=".bundle-",
            delete=False,
        ) as f:
            temp_path = f.name
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise DSLRuntimeError(
            f"Failed writing {scope} provider cache artifact.",
            cause=exc,
        ) from exc


def write_text_atomic(path: str, text: str, *, scope: str) -> None:
    write_binary_atomic(path, text.encode("utf-8"), scope=scope)


def _layout_metadata_path(output_path: str) -> str:
    return f"{output_path}.layouts.json"


def _optional_metadata_string(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("invalid provider producer metadata")


def _load_bundle_metadata(
    output_path: str,
    expressions: tuple[str, ...],
    cache_identity: BundleCacheIdentity,
) -> _CachedBundle | None:
    metadata_path = _layout_metadata_path(output_path)
    try:
        metadata_stat = os.lstat(metadata_path)
        if not stat.S_ISREG(metadata_stat.st_mode):
            return None
        with open(metadata_path, encoding="utf-8") as metadata_file:
            payload = json.load(metadata_file)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != BUNDLE_METADATA_VERSION
            or payload.get("cache_key") != cache_identity.cache_key
        ):
            return None

        artifact = payload.get("artifact")
        if not isinstance(artifact, dict):
            return None
        artifact_size = artifact.get("size")
        artifact_sha256 = artifact.get("sha256")
        if (
            not isinstance(artifact_size, int)
            or isinstance(artifact_size, bool)
            or artifact_size < 0
            or not isinstance(artifact_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
        ):
            return None
        output_stat = os.lstat(output_path)
        if (
            not stat.S_ISREG(output_stat.st_mode)
            or output_stat.st_size != artifact_size
        ):
            return None
        artifact_digest = hashlib.sha256()
        with open(output_path, "rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                artifact_digest.update(chunk)
        if artifact_digest.hexdigest() != artifact_sha256:
            return None

        serialized_layouts = payload.get("layouts")
        if not isinstance(serialized_layouts, dict) or set(serialized_layouts) != set(
            expressions
        ):
            return None
        layouts = {}
        for expression in expressions:
            serialized_layout = serialized_layouts[expression]
            if not isinstance(serialized_layout, list) or len(serialized_layout) != 2:
                return None
            layouts[expression] = _validate_storage_layout(
                serialized_layout[0],
                serialized_layout[1],
                description=expression,
            )

        producer = payload.get("producer")
        if not isinstance(producer, dict):
            return None
        return _CachedBundle(
            path=output_path,
            layouts_by_expression=layouts,
            producer_compiler=_optional_metadata_string(producer.get("compiler")),
            producer_compiler_version=_optional_metadata_string(
                producer.get("compiler_version")
            ),
            producer_toolkit_version=_optional_metadata_string(
                producer.get("toolkit_version")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None


def _write_bundle_metadata(
    output_path: str,
    artifact_blob: bytes | bytearray,
    cached: _CachedBundle,
    cache_identity: BundleCacheIdentity,
    *,
    scope: str,
) -> None:
    payload = {
        "version": BUNDLE_METADATA_VERSION,
        "cache_key": cache_identity.cache_key,
        "artifact": {
            "size": len(artifact_blob),
            "sha256": hashlib.sha256(artifact_blob).hexdigest(),
        },
        "producer": {
            "compiler": cached.producer_compiler,
            "compiler_version": cached.producer_compiler_version,
            "toolkit_version": cached.producer_toolkit_version,
        },
        "layouts": {
            expression: [layout.size_in_bytes, layout.alignment]
            for expression, layout in cached.layouts_by_expression.items()
        },
    }
    write_text_atomic(
        _layout_metadata_path(output_path),
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        scope=scope,
    )
