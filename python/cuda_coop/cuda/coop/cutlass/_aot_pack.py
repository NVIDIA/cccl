# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable exact-bundle AOT packs for the CUTLASS cooperative provider."""

from __future__ import annotations

import ctypes
import errno
import functools
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

PACK_FORMAT = "cuda.coop.cutlass.aot-pack"
PACK_SCHEMA_VERSION = 1
PROVIDER_ABI_VERSION = 1
PACK_PATH_ENV = "CUDA_COOP_CUTLASS_AOT_PACK_PATH"
PACK_MODE_ENV = "CUDA_COOP_CUTLASS_AOT_MODE"
MANIFEST_NAME = "manifest.json"
_CAPTURE_PROCESS_BOUNDARY_ERROR = (
    "AOT capture contexts cannot cross a process boundary; "
    "enter capture after the child process starts."
)
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"([0-9]+)\.([0-9]+)(?:\.[0-9]+)?")
_COMPUTE_ARCH = re.compile(r"compute_([0-9]+[af]?)")
_SM_ARCH = re.compile(r"sm_([0-9]+[af]?)")
_VALID_MODES = frozenset({"auto", "required", "off"})


def _provider_contract():
    """Load compiler identity types only when an AOT operation needs them."""

    from ._compiler import _bundle_contract

    return _bundle_contract


class PackError(RuntimeError):
    """Base error for CUTLASS cooperative AOT packs."""


class PackIntegrityError(PackError):
    """A pack is malformed, corrupt, or unsupported."""


class PackMissError(PackError):
    """A required pack has no compatible exact-bundle entry."""


class CaptureError(PackError):
    """A provider resolution cannot be captured or published."""


def _require_supported_platform() -> None:
    if sys.platform != "linux":
        raise PackError("CUTLASS provider AOT packs currently require Linux.")


@dataclass(frozen=True)
class EntryInfo:
    """Portable information about one exact provider bundle."""

    entry_id: str
    source_sha256: str
    artifact_sha256: str
    artifact_size: int
    provider_abi_version: int
    bundle_format: str
    compute_arch: str
    sm_arch: str
    compiler_options: tuple[str, ...]
    layout_expressions: tuple[str, ...]
    symbols: tuple[str, ...]
    producer_compiler: str
    producer_version: tuple[int, int]
    producer_toolkit_version: str | None


@dataclass(frozen=True)
class PackInfo:
    """Validated information about one relocatable AOT pack."""

    path: Path
    name: str | None
    schema_version: int
    provider_abi_version: int
    writer_version: str
    entries: tuple[EntryInfo, ...]

    @property
    def artifact_bytes(self) -> int:
        return sum(entry.artifact_size for entry in self.entries)


@dataclass(frozen=True)
class CaptureResult:
    """Result published by a successful capture context."""

    path: Path
    name: str | None
    observations: int
    entries: tuple[EntryInfo, ...]

    @property
    def artifact_bytes(self) -> int:
        return sum(entry.artifact_size for entry in self.entries)


@dataclass(frozen=True)
class _CudaVersion:
    major: int
    minor: int


@dataclass(frozen=True)
class _ManifestIdentity:
    provider_abi_version: int
    source_sha256: str
    bundle_format: str
    compute_arch: str
    sm_arch: str
    compiler_options: tuple[str, ...]
    layout_expressions: tuple[str, ...]


@dataclass(frozen=True)
class _ManifestLayout:
    expression: str
    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class _ManifestEntry:
    entry_id: str
    identity: _ManifestIdentity
    artifact_sha256: str
    artifact_size: int
    producer_compiler: str
    producer_version: _CudaVersion
    producer_toolkit_version: str | None
    layouts: tuple[_ManifestLayout, ...]
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class _Manifest:
    name: str | None
    writer_version: str
    entries: tuple[_ManifestEntry, ...]


@dataclass(frozen=True)
class _LoadedPack:
    root: Path
    manifest: _Manifest
    entries_by_id: Mapping[str, _ManifestEntry]
    artifact_paths_by_digest: Mapping[str, str]

    @property
    def info(self) -> PackInfo:
        return _pack_info(self.root, self.manifest)


@dataclass(frozen=True)
class _Selection:
    mode: Literal["auto", "required", "off"]
    pack: _LoadedPack | None


@dataclass(frozen=True)
class _MaterializedArtifact:
    digest: str
    descriptor: int
    path: str
    size: int


class _DuplicateJsonKey(ValueError):
    pass


_ACTIVE_CAPTURE: ContextVar[Capture | None] = ContextVar(
    "cuda_coop_cutlass_active_aot_capture",
    default=None,
)
_ACTIVE_SELECTION: ContextVar[_Selection | None] = ContextVar(
    "cuda_coop_cutlass_active_aot_selection",
    default=None,
)
_PACK_CACHE: dict[str, _LoadedPack] = {}
_PACK_CACHE_LOCK = threading.RLock()
_MATERIALIZED_ARTIFACTS: dict[str, _MaterializedArtifact] = {}
_MATERIALIZED_ARTIFACTS_BY_PATH: dict[str, _MaterializedArtifact] = {}
_MATERIALIZED_ARTIFACTS_LOCK = threading.RLock()


def _reset_locks_after_fork() -> None:
    global _MATERIALIZED_ARTIFACTS_LOCK, _PACK_CACHE_LOCK
    _PACK_CACHE_LOCK = threading.RLock()
    _MATERIALIZED_ARTIFACTS_LOCK = threading.RLock()
    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        return

    _ACTIVE_CAPTURE.set(None)
    sink = capture._sink
    observers_context = capture._provider_observers_context
    if sink is not None and observers_context is not None:
        observers_context.set(
            tuple(
                observer
                for observer in observers_context.get()
                if getattr(observer, "__self__", None) is not sink
            )
        )
    capture._capture_token = None
    capture._observer_context = None
    capture._provider_observers_context = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_locks_after_fork)


def _secure_nofollow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", None)
    if flag is None:
        raise PackError("Secure AOT pack reads require O_NOFOLLOW support.")
    return flag


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Any,
    keys: frozenset[str],
    *,
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackIntegrityError(f"{description} must be a JSON object.")
    actual = frozenset(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unexpected = sorted(actual - keys)
        raise PackIntegrityError(
            f"{description} has invalid fields: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return value


def _require_string(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PackIntegrityError(f"{description} must be a non-empty string.")
    return value


def _require_optional_string(value: Any, *, description: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, description=description)


def _require_integer(
    value: Any,
    *,
    description: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PackIntegrityError(
            f"{description} must be an integer greater than or equal to {minimum}."
        )
    return value


def _require_digest(value: Any, *, description: str) -> str:
    digest = _require_string(value, description=description)
    if _HEX_DIGEST.fullmatch(digest) is None:
        raise PackIntegrityError(
            f"{description} must be a lowercase SHA-256 hexadecimal digest."
        )
    return digest


def _require_string_tuple(value: Any, *, description: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PackIntegrityError(f"{description} must be a JSON array.")
    return tuple(
        _require_string(item, description=f"{description} item") for item in value
    )


def _identity_payload(identity: _ManifestIdentity) -> dict[str, Any]:
    return {
        "bundle_format": identity.bundle_format,
        "compiler_options": list(identity.compiler_options),
        "compute_arch": identity.compute_arch,
        "layout_expressions": list(identity.layout_expressions),
        "provider_abi_version": identity.provider_abi_version,
        "sm_arch": identity.sm_arch,
        "source_sha256": identity.source_sha256,
    }


def _entry_id(
    identity: _ManifestIdentity,
    symbols: tuple[str, ...],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "identity": _identity_payload(identity),
                "symbols": list(symbols),
            }
        )
    ).hexdigest()


def _layout_payload(layout: _ManifestLayout) -> dict[str, Any]:
    return {
        "alignment": layout.alignment,
        "expression": layout.expression,
        "size_in_bytes": layout.size_in_bytes,
    }


def _entry_payload(entry: _ManifestEntry) -> dict[str, Any]:
    return {
        "artifact_sha256": entry.artifact_sha256,
        "artifact_size": entry.artifact_size,
        "entry_id": entry.entry_id,
        "identity": _identity_payload(entry.identity),
        "layouts": [_layout_payload(layout) for layout in entry.layouts],
        "producer": {
            "compiler": entry.producer_compiler,
            "toolkit_version": entry.producer_toolkit_version,
            "version": {
                "major": entry.producer_version.major,
                "minor": entry.producer_version.minor,
            },
        },
        "symbols": list(entry.symbols),
    }


def _manifest_payload(manifest: _Manifest) -> dict[str, Any]:
    return {
        "entries": [_entry_payload(entry) for entry in manifest.entries],
        "format": PACK_FORMAT,
        "name": manifest.name,
        "provider_abi_version": PROVIDER_ABI_VERSION,
        "schema_version": PACK_SCHEMA_VERSION,
        "writer": {
            "package": "cuda-coop",
            "version": manifest.writer_version,
        },
    }


def _parse_identity(value: Any, *, entry_index: int) -> _ManifestIdentity:
    payload = _require_exact_keys(
        value,
        frozenset(
            {
                "bundle_format",
                "compiler_options",
                "compute_arch",
                "layout_expressions",
                "provider_abi_version",
                "sm_arch",
                "source_sha256",
            }
        ),
        description=f"entry {entry_index} identity",
    )
    identity = _ManifestIdentity(
        provider_abi_version=_require_integer(
            payload["provider_abi_version"],
            description=f"entry {entry_index} provider ABI version",
            minimum=1,
        ),
        source_sha256=_require_digest(
            payload["source_sha256"],
            description=f"entry {entry_index} source digest",
        ),
        bundle_format=_require_string(
            payload["bundle_format"],
            description=f"entry {entry_index} bundle format",
        ),
        compute_arch=_require_string(
            payload["compute_arch"],
            description=f"entry {entry_index} compute architecture",
        ),
        sm_arch=_require_string(
            payload["sm_arch"],
            description=f"entry {entry_index} SM architecture",
        ),
        compiler_options=_require_string_tuple(
            payload["compiler_options"],
            description=f"entry {entry_index} compiler options",
        ),
        layout_expressions=_require_string_tuple(
            payload["layout_expressions"],
            description=f"entry {entry_index} layout expressions",
        ),
    )
    if identity.provider_abi_version != PROVIDER_ABI_VERSION:
        raise PackIntegrityError(
            f"entry {entry_index} uses unsupported provider ABI version "
            f"{identity.provider_abi_version}."
        )
    if identity.bundle_format != "ltoir":
        raise PackIntegrityError(f"entry {entry_index} must contain an LTO-IR bundle.")
    compute_match = _COMPUTE_ARCH.fullmatch(identity.compute_arch)
    sm_match = _SM_ARCH.fullmatch(identity.sm_arch)
    if (
        compute_match is None
        or sm_match is None
        or compute_match.group(1) != sm_match.group(1)
    ):
        raise PackIntegrityError(
            f"entry {entry_index} compute and SM architectures must be an "
            "exact matching compute_N/sm_N pair."
        )
    if tuple(sorted(set(identity.layout_expressions))) != identity.layout_expressions:
        raise PackIntegrityError(
            f"entry {entry_index} layout expressions must be unique and sorted."
        )
    return identity


def _parse_layout(
    value: Any, *, entry_index: int, layout_index: int
) -> _ManifestLayout:
    payload = _require_exact_keys(
        value,
        frozenset({"alignment", "expression", "size_in_bytes"}),
        description=f"entry {entry_index} layout {layout_index}",
    )
    layout = _ManifestLayout(
        expression=_require_string(
            payload["expression"],
            description=f"entry {entry_index} layout {layout_index} expression",
        ),
        size_in_bytes=_require_integer(
            payload["size_in_bytes"],
            description=f"entry {entry_index} layout {layout_index} size",
            minimum=1,
        ),
        alignment=_require_integer(
            payload["alignment"],
            description=f"entry {entry_index} layout {layout_index} alignment",
            minimum=1,
        ),
    )
    if (
        layout.alignment & (layout.alignment - 1)
        or layout.size_in_bytes % layout.alignment
    ):
        raise PackIntegrityError(
            f"entry {entry_index} layout {layout_index} has an invalid "
            "size or alignment."
        )
    return layout


def _parse_entry(value: Any, *, entry_index: int) -> _ManifestEntry:
    payload = _require_exact_keys(
        value,
        frozenset(
            {
                "artifact_sha256",
                "artifact_size",
                "entry_id",
                "identity",
                "layouts",
                "producer",
                "symbols",
            }
        ),
        description=f"entry {entry_index}",
    )
    identity = _parse_identity(payload["identity"], entry_index=entry_index)
    producer = _require_exact_keys(
        payload["producer"],
        frozenset({"compiler", "toolkit_version", "version"}),
        description=f"entry {entry_index} producer",
    )
    producer_version = _require_exact_keys(
        producer["version"],
        frozenset({"major", "minor"}),
        description=f"entry {entry_index} producer version",
    )
    layouts_value = payload["layouts"]
    if not isinstance(layouts_value, list):
        raise PackIntegrityError(f"entry {entry_index} layouts must be a JSON array.")
    layouts = tuple(
        _parse_layout(item, entry_index=entry_index, layout_index=layout_index)
        for layout_index, item in enumerate(layouts_value)
    )
    entry = _ManifestEntry(
        entry_id=_require_digest(
            payload["entry_id"],
            description=f"entry {entry_index} ID",
        ),
        identity=identity,
        artifact_sha256=_require_digest(
            payload["artifact_sha256"],
            description=f"entry {entry_index} artifact digest",
        ),
        artifact_size=_require_integer(
            payload["artifact_size"],
            description=f"entry {entry_index} artifact size",
            minimum=1,
        ),
        producer_compiler=_require_string(
            producer["compiler"],
            description=f"entry {entry_index} producer compiler",
        ),
        producer_version=_CudaVersion(
            major=_require_integer(
                producer_version["major"],
                description=f"entry {entry_index} producer major version",
            ),
            minor=_require_integer(
                producer_version["minor"],
                description=f"entry {entry_index} producer minor version",
            ),
        ),
        producer_toolkit_version=_require_optional_string(
            producer["toolkit_version"],
            description=f"entry {entry_index} producer toolkit version",
        ),
        layouts=layouts,
        symbols=_require_string_tuple(
            payload["symbols"],
            description=f"entry {entry_index} symbols",
        ),
    )
    if entry.entry_id != _entry_id(identity, entry.symbols):
        raise PackIntegrityError(
            f"entry {entry_index} ID does not match its exact bundle identity "
            "and symbol set."
        )
    if entry.producer_compiler != "nvrtc":
        raise PackIntegrityError(
            f"entry {entry_index} has unsupported producer compiler "
            f"{entry.producer_compiler!r}."
        )
    layout_expressions = tuple(layout.expression for layout in entry.layouts)
    if layout_expressions != identity.layout_expressions:
        raise PackIntegrityError(
            f"entry {entry_index} layout metadata does not match its identity."
        )
    if tuple(sorted(set(entry.symbols))) != entry.symbols:
        raise PackIntegrityError(
            f"entry {entry_index} symbols must be unique and sorted."
        )
    return entry


def _parse_manifest(data: bytes) -> _Manifest:
    try:
        payload = json.loads(
            data,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PackIntegrityError(
            "AOT pack manifest is not valid canonical JSON."
        ) from exc
    root = _require_exact_keys(
        payload,
        frozenset(
            {
                "entries",
                "format",
                "name",
                "provider_abi_version",
                "schema_version",
                "writer",
            }
        ),
        description="AOT pack manifest",
    )
    if root["format"] != PACK_FORMAT:
        raise PackIntegrityError("AOT pack manifest has an unsupported format.")
    if root["schema_version"] != PACK_SCHEMA_VERSION:
        raise PackIntegrityError(
            f"AOT pack manifest uses unsupported schema version "
            f"{root['schema_version']!r}."
        )
    if root["provider_abi_version"] != PROVIDER_ABI_VERSION:
        raise PackIntegrityError(
            f"AOT pack manifest uses unsupported provider ABI version "
            f"{root['provider_abi_version']!r}."
        )
    writer = _require_exact_keys(
        root["writer"],
        frozenset({"package", "version"}),
        description="AOT pack writer",
    )
    if writer["package"] != "cuda-coop":
        raise PackIntegrityError("AOT pack writer package must be 'cuda-coop'.")
    writer_version = _require_string(
        writer["version"],
        description="AOT pack writer version",
    )
    name = root["name"]
    if name is not None:
        name = _require_string(name, description="AOT pack name")
    entries_value = root["entries"]
    if not isinstance(entries_value, list) or not entries_value:
        raise PackIntegrityError("AOT pack manifest must contain at least one entry.")
    entries = tuple(
        _parse_entry(entry, entry_index=index)
        for index, entry in enumerate(entries_value)
    )
    entry_ids = tuple(entry.entry_id for entry in entries)
    if tuple(sorted(set(entry_ids))) != entry_ids:
        raise PackIntegrityError("AOT pack entries must have unique sorted IDs.")
    manifest = _Manifest(
        name=name,
        writer_version=writer_version,
        entries=entries,
    )
    if data != _canonical_json_bytes(_manifest_payload(manifest)):
        raise PackIntegrityError("AOT pack manifest is not canonically encoded.")
    return manifest


def _open_directory(
    path: str | os.PathLike[str],
    *,
    description: str,
    directory_descriptor: int | None = None,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | _secure_nofollow_flag()
    )
    try:
        descriptor = os.open(path, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise PackIntegrityError(f"{description} must be a real directory.") from exc
    try:
        result = os.fstat(descriptor)
        if not stat.S_ISDIR(result.st_mode):
            raise PackIntegrityError(f"{description} must be a real directory.")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_open_regular_file(
    descriptor: int,
    *,
    description: str,
    maximum_size: int | None,
) -> bytes:
    try:
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode):
            raise PackIntegrityError(f"{description} must be a regular file.")
        if maximum_size is not None and result.st_size > maximum_size:
            raise PackIntegrityError(f"{description} is unexpectedly large.")
        with os.fdopen(descriptor, "rb", closefd=True) as input_file:
            descriptor = -1
            if maximum_size is None:
                return input_file.read()
            data = input_file.read(result.st_size)
            if input_file.read(1):
                raise PackIntegrityError(f"{description} is unexpectedly large.")
            return data
    except PackIntegrityError:
        raise
    except OSError as exc:
        raise PackIntegrityError(f"{description} could not be read.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    description: str,
    maximum_size: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _secure_nofollow_flag()
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise PackIntegrityError(
            f"{description} is missing, unreadable, or not a regular file."
        ) from exc
    return _read_open_regular_file(
        descriptor,
        description=description,
        maximum_size=maximum_size,
    )


def _read_regular_file(
    path: Path,
    *,
    description: str,
    maximum_size: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _secure_nofollow_flag()
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PackIntegrityError(
            f"{description} is missing, unreadable, or not a regular file."
        ) from exc
    return _read_open_regular_file(
        descriptor,
        description=description,
        maximum_size=maximum_size,
    )


def _directory_inventory(
    descriptor: int,
    *,
    description: str,
) -> set[str]:
    try:
        return set(os.listdir(descriptor))
    except OSError as exc:
        raise PackIntegrityError(f"{description} is unreadable.") from exc


def _create_sealed_memfd(digest: str, artifact: bytes) -> _MaterializedArtifact:
    try:
        function = ctypes.CDLL(None, use_errno=True).memfd_create
    except AttributeError as exc:
        raise PackError(
            "Stable AOT artifact materialization requires Linux memfd_create."
        ) from exc
    function.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = function(
        f"cuda-coop-{digest}.ltoir".encode("ascii"),
        _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise PackError("Failed creating a stable AOT artifact.") from OSError(
            error_number,
            os.strerror(error_number),
        )

    try:
        view = memoryview(artifact)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write to AOT artifact memfd")
            view = view[written:]
        os.fsync(descriptor)
        try:
            import fcntl

            fcntl.fcntl(
                descriptor,
                _F_ADD_SEALS,
                _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE,
            )
        except (ImportError, OSError) as exc:
            raise PackError("Failed sealing a stable AOT artifact.") from exc
        os.lseek(descriptor, 0, os.SEEK_SET)
        path = f"/proc/self/fd/{descriptor}"
        if not os.path.isfile(path):
            raise PackError(
                "Stable AOT artifact materialization requires Linux procfs."
            )
        return _MaterializedArtifact(
            digest=digest,
            descriptor=descriptor,
            path=path,
            size=len(artifact),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _materialize_artifact(digest: str, artifact: bytes) -> str:
    with _MATERIALIZED_ARTIFACTS_LOCK:
        existing = _MATERIALIZED_ARTIFACTS.get(digest)
        if existing is not None:
            if existing.size != len(artifact):
                raise PackIntegrityError(
                    f"AOT artifact {digest} conflicts with its materialized bytes."
                )
            return existing.path
        materialized = _create_sealed_memfd(digest, artifact)
        _MATERIALIZED_ARTIFACTS[digest] = materialized
        _MATERIALIZED_ARTIFACTS_BY_PATH[materialized.path] = materialized
        return materialized.path


def _validate_pack_tree(
    root_descriptor: int,
    manifest: _Manifest,
    *,
    materialize_artifacts: bool,
) -> dict[str, str]:
    artifacts_descriptor = _open_directory(
        "artifacts",
        description="AOT pack artifacts directory",
        directory_descriptor=root_descriptor,
    )
    try:
        sources_descriptor = _open_directory(
            "sources",
            description="AOT pack sources directory",
            directory_descriptor=root_descriptor,
        )
    except BaseException:
        os.close(artifacts_descriptor)
        raise

    try:
        expected_root = {MANIFEST_NAME, "artifacts", "sources"}
        actual_root = _directory_inventory(
            root_descriptor,
            description="AOT pack",
        )
        if actual_root != expected_root:
            raise PackIntegrityError(
                "AOT pack contains unexpected or missing top-level entries."
            )

        expected_artifacts = {
            f"{entry.artifact_sha256}.ltoir" for entry in manifest.entries
        }
        expected_sources = {
            f"{entry.identity.source_sha256}.cu" for entry in manifest.entries
        }
        if (
            _directory_inventory(
                artifacts_descriptor,
                description="AOT pack artifacts directory",
            )
            != expected_artifacts
        ):
            raise PackIntegrityError("AOT pack artifact inventory is inconsistent.")
        if (
            _directory_inventory(
                sources_descriptor,
                description="AOT pack sources directory",
            )
            != expected_sources
        ):
            raise PackIntegrityError("AOT pack source inventory is inconsistent.")

        artifacts_to_materialize: dict[str, bytes] = {}
        verified_artifact_sizes: dict[str, int] = {}
        verified_sources: set[str] = set()
        for entry in manifest.entries:
            verified_size = verified_artifact_sizes.get(entry.artifact_sha256)
            if verified_size is None:
                artifact = _read_regular_file_at(
                    artifacts_descriptor,
                    f"{entry.artifact_sha256}.ltoir",
                    description=f"AOT pack artifact {entry.artifact_sha256}",
                    maximum_size=entry.artifact_size,
                )
                if len(artifact) != entry.artifact_size:
                    raise PackIntegrityError(
                        f"AOT pack artifact {entry.artifact_sha256} "
                        "has an invalid size."
                    )
                if hashlib.sha256(artifact).hexdigest() != entry.artifact_sha256:
                    raise PackIntegrityError(
                        f"AOT pack artifact {entry.artifact_sha256} "
                        "has an invalid digest."
                    )
                if materialize_artifacts:
                    artifacts_to_materialize[entry.artifact_sha256] = artifact
                verified_artifact_sizes[entry.artifact_sha256] = len(artifact)
            elif entry.artifact_size != verified_size:
                raise PackIntegrityError(
                    f"AOT pack artifact {entry.artifact_sha256} "
                    "has conflicting declared sizes."
                )

            source_sha256 = entry.identity.source_sha256
            if source_sha256 not in verified_sources:
                source = _read_regular_file_at(
                    sources_descriptor,
                    f"{source_sha256}.cu",
                    description=f"AOT pack source {source_sha256}",
                    maximum_size=MAX_SOURCE_BYTES,
                )
                if hashlib.sha256(source).hexdigest() != source_sha256:
                    raise PackIntegrityError(
                        f"AOT pack source {source_sha256} has an invalid digest."
                    )
                try:
                    source.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise PackIntegrityError(
                        f"AOT pack source {source_sha256} is not valid UTF-8."
                    ) from exc
                verified_sources.add(source_sha256)
        return {
            digest: _materialize_artifact(digest, artifact)
            for digest, artifact in artifacts_to_materialize.items()
        }
    finally:
        os.close(artifacts_descriptor)
        os.close(sources_descriptor)


def _normalize_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _load_pack(
    path: str | os.PathLike[str],
    *,
    materialize_artifacts: bool = False,
) -> _LoadedPack:
    _require_supported_platform()
    root = _normalize_path(path)
    root_descriptor = _open_directory(root, description="AOT pack")
    try:
        data = _read_regular_file_at(
            root_descriptor,
            MANIFEST_NAME,
            description="AOT pack manifest",
            maximum_size=MAX_MANIFEST_BYTES,
        )
        manifest = _parse_manifest(data)
        artifact_paths = _validate_pack_tree(
            root_descriptor,
            manifest,
            materialize_artifacts=materialize_artifacts,
        )
    finally:
        os.close(root_descriptor)
    return _LoadedPack(
        root=root,
        manifest=manifest,
        entries_by_id={entry.entry_id: entry for entry in manifest.entries},
        artifact_paths_by_digest=artifact_paths,
    )


def _load_pack_cached(path: str | os.PathLike[str]) -> _LoadedPack:
    """Load an environment-selected pack once and treat it as immutable."""

    normalized = _normalize_path(path)
    key = os.fspath(normalized)
    with _PACK_CACHE_LOCK:
        pack = _PACK_CACHE.get(key)
    if pack is not None:
        return pack
    pack = _load_pack(normalized, materialize_artifacts=True)
    with _PACK_CACHE_LOCK:
        return _PACK_CACHE.setdefault(key, pack)


def _entry_info(entry: _ManifestEntry) -> EntryInfo:
    identity = entry.identity
    return EntryInfo(
        entry_id=entry.entry_id,
        source_sha256=identity.source_sha256,
        artifact_sha256=entry.artifact_sha256,
        artifact_size=entry.artifact_size,
        provider_abi_version=identity.provider_abi_version,
        bundle_format=identity.bundle_format,
        compute_arch=identity.compute_arch,
        sm_arch=identity.sm_arch,
        compiler_options=identity.compiler_options,
        layout_expressions=identity.layout_expressions,
        symbols=entry.symbols,
        producer_compiler=entry.producer_compiler,
        producer_version=(entry.producer_version.major, entry.producer_version.minor),
        producer_toolkit_version=entry.producer_toolkit_version,
    )


def _pack_info(root: Path, manifest: _Manifest) -> PackInfo:
    return PackInfo(
        path=root,
        name=manifest.name,
        schema_version=PACK_SCHEMA_VERSION,
        provider_abi_version=PROVIDER_ABI_VERSION,
        writer_version=manifest.writer_version,
        entries=tuple(_entry_info(entry) for entry in manifest.entries),
    )


def inspect(pack: str | os.PathLike[str]) -> PackInfo:
    """Validate and describe one AOT pack.

    Selected packs are treated as immutable. Explicit ``use`` contexts validate
    at entry, while environment-selected packs validate once per process.
    Integrity validation is not authenticity or semantic LTO-IR validation:
    packs are trusted native-code inputs and must come from trusted provenance.
    """

    return _load_pack(pack).info


def _validate_mode(mode: str) -> Literal["auto", "required", "off"]:
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
    return mode  # type: ignore[return-value]


@contextmanager
def use(
    pack: str | os.PathLike[str],
    *,
    mode: Literal["auto", "required", "off"] = "auto",
) -> Iterator[PackInfo | None]:
    """Select one exact-bundle pack for provider resolution in this context.

    Validation copies artifact bytes into sealed process-lifetime
    materializations, so later pack mutation cannot alter linker input.
    """

    validated_mode = _validate_mode(mode)
    loaded = (
        None
        if validated_mode == "off"
        else _load_pack(pack, materialize_artifacts=True)
    )
    selection = _Selection(mode=validated_mode, pack=loaded)
    token = _ACTIVE_SELECTION.set(selection)
    try:
        if loaded is None:
            yield None
        else:
            from ._compiler import _bundle as _provider_bundle

            with _provider_bundle.activate_bundle_precompile_resolver(
                _precompile_resolver()
            ):
                yield loaded.info
    finally:
        _ACTIVE_SELECTION.reset(token)


def _environment_selection() -> _Selection:
    raw_mode = os.environ.get(PACK_MODE_ENV)
    path = os.environ.get(PACK_PATH_ENV)
    mode = _validate_mode("auto" if raw_mode is None else raw_mode.strip().lower())
    if mode == "off":
        return _Selection(mode="off", pack=None)
    if not path:
        if mode == "required":
            raise PackMissError(
                f"{PACK_MODE_ENV}=required needs a non-empty {PACK_PATH_ENV}."
            )
        return _Selection(mode="off", pack=None)
    if not os.path.isabs(path):
        raise PackError(f"{PACK_PATH_ENV} must be an absolute path, got {path!r}.")
    return _Selection(mode=mode, pack=_load_pack_cached(path))


def _current_selection() -> _Selection:
    selection = _ACTIVE_SELECTION.get()
    if selection is not None:
        return selection
    return _environment_selection()


def _precompile_resolver() -> Any:
    from ._compiler import _bundle as _provider_bundle

    return _provider_bundle._bundle_precompile_resolver(
        resolve_precompiled_bundle,
        route=_provider_contract().RESOLUTION_ROUTE_AOT_PACK,
        phase="pack_lookup",
    )


def _environment_precompile_resolver() -> Any | None:
    """Return the lazy environment resolver unless an explicit selection wins."""

    if _ACTIVE_SELECTION.get() is not None:
        return None
    return _precompile_resolver()


def _manifest_identity_from_request(request: Any) -> _ManifestIdentity:
    identity = request.identity
    return _ManifestIdentity(
        provider_abi_version=identity.provider_abi_version,
        source_sha256=identity.source_hash,
        bundle_format=identity.bundle_format,
        compute_arch=identity.bundle_arch,
        sm_arch=identity.bundle_sm_arch,
        compiler_options=tuple(identity.compiler_options),
        layout_expressions=tuple(identity.layout_expressions),
    )


def _assert_provider_constants() -> None:
    if _provider_contract().PROVIDER_BUNDLE_ABI_VERSION != PROVIDER_ABI_VERSION:
        raise PackIntegrityError(
            "AOT pack provider ABI version is out of sync with the provider."
        )


@functools.lru_cache(maxsize=1)
def _consumer_nvjitlink_version() -> _CudaVersion:
    try:
        import cuda.bindings.nvjitlink as cuda_nvjitlink
    except (ImportError, OSError, RuntimeError) as exc:
        raise PackMissError(
            "Unable to determine the consumer nvJitLink version."
        ) from exc
    try:
        major, minor = cuda_nvjitlink.version()
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        cuda_nvjitlink.nvJitLinkError,
    ) as exc:
        raise PackMissError(
            "Unable to determine the consumer nvJitLink version."
        ) from exc
    if (
        isinstance(major, bool)
        or not isinstance(major, int)
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or major < 0
        or minor < 0
    ):
        raise PackMissError("Consumer nvJitLink returned an invalid version.")
    return _CudaVersion(major=major, minor=minor)


def _compatible_nvjitlink(entry: _ManifestEntry) -> tuple[bool, str]:
    try:
        consumer = _consumer_nvjitlink_version()
    except PackMissError as exc:
        return False, str(exc)
    producer = entry.producer_version
    if consumer.major != producer.major:
        return (
            False,
            f"producer NVRTC {producer.major}.{producer.minor} and consumer "
            f"nvJitLink {consumer.major}.{consumer.minor} have different major "
            "versions",
        )
    if consumer.minor < producer.minor:
        return (
            False,
            f"consumer nvJitLink {consumer.major}.{consumer.minor} is older than "
            f"producer NVRTC {producer.major}.{producer.minor}",
        )
    return True, ""


def _required_miss(selection: _Selection, identity: _ManifestIdentity, reason: str):
    assert selection.pack is not None
    raise PackMissError(
        f"Required AOT pack {selection.pack.root} has no compatible exact bundle "
        f"for {identity.compute_arch}/{identity.sm_arch}: {reason}."
    )


def resolve_precompiled_bundle(request: Any) -> Any | None:
    """Foundation resolver dispatcher for explicit or environment-selected packs."""

    selection = _current_selection()
    if selection.mode == "off":
        return None
    assert selection.pack is not None
    _assert_provider_constants()

    identity = _manifest_identity_from_request(request)
    request_symbols = tuple(request.symbols)
    if tuple(sorted(set(request_symbols))) != request_symbols:
        raise PackError(
            "Provider resolution request symbols must be unique and sorted."
        )
    entry = selection.pack.entries_by_id.get(_entry_id(identity, request_symbols))
    if entry is None or entry.identity != identity or entry.symbols != request_symbols:
        if selection.mode == "required":
            _required_miss(selection, identity, "entry is absent")
        return None

    compatible, reason = _compatible_nvjitlink(entry)
    if not compatible:
        if selection.mode == "required":
            _required_miss(selection, identity, reason)
        return None

    try:
        artifact_path = selection.pack.artifact_paths_by_digest[entry.artifact_sha256]
    except KeyError as exc:
        raise PackIntegrityError(
            "Selected AOT pack has no stable materialization for its exact artifact."
        ) from exc
    return _provider_contract().BundleResolution(
        request=request,
        path=artifact_path,
        layouts_by_expression={
            layout.expression: _provider_contract().StorageLayout(
                size_in_bytes=layout.size_in_bytes,
                alignment=layout.alignment,
            )
            for layout in entry.layouts
        },
        route=_provider_contract().RESOLUTION_ROUTE_AOT_PACK,
        producer_compiler=entry.producer_compiler,
        producer_compiler_version=(
            f"{entry.producer_version.major}.{entry.producer_version.minor}"
        ),
        producer_toolkit_version=entry.producer_toolkit_version,
        phase_timings_ns={},
    )


def _parse_producer_version(value: str | None) -> _CudaVersion:
    if value is None:
        raise CaptureError(
            "Cannot capture a provider artifact without producer compiler "
            "version metadata. Clear the legacy provider disk cache and compile "
            "the bundle again."
        )
    match = _VERSION.fullmatch(value)
    if match is None:
        raise CaptureError(
            f"Cannot capture invalid producer compiler version {value!r}."
        )
    return _CudaVersion(major=int(match.group(1)), minor=int(match.group(2)))


def _write_new_file(path: Path, data: bytes, *, description: str) -> None:
    try:
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        try:
            existing = _read_regular_file(path, description=description)
        except PackIntegrityError as exc:
            raise CaptureError(
                f"{description} conflicts with captured content."
            ) from exc
        if existing != data:
            raise CaptureError(f"{description} conflicts with captured content.")
    except OSError as exc:
        raise CaptureError(f"Failed writing {description}.") from exc


def _read_materialized_artifact(path: str) -> bytes:
    with _MATERIALIZED_ARTIFACTS_LOCK:
        materialized = _MATERIALIZED_ARTIFACTS_BY_PATH.get(path)
    if materialized is None:
        raise CaptureError(
            "Resolved AOT artifact is not a process-owned stable materialization."
        )
    chunks = []
    offset = 0
    try:
        while offset < materialized.size:
            chunk = os.pread(
                materialized.descriptor,
                materialized.size - offset,
                offset,
            )
            if not chunk:
                raise OSError(errno.EIO, "short read from AOT artifact memfd")
            chunks.append(chunk)
            offset += len(chunk)
    except OSError as exc:
        raise CaptureError("Resolved AOT artifact could not be read.") from exc
    return b"".join(chunks)


def _read_resolution_artifact(resolution: Any) -> bytes:
    if resolution.route == _provider_contract().RESOLUTION_ROUTE_AOT_PACK:
        artifact = _read_materialized_artifact(resolution.path)
    else:
        try:
            artifact = _read_regular_file(
                Path(resolution.path),
                description="Resolved provider artifact",
            )
        except PackIntegrityError as exc:
            raise CaptureError(
                "Resolved provider artifact must be a readable regular file."
            ) from exc
    if not artifact:
        raise CaptureError("Resolved provider artifact is empty.")
    return artifact


def _capture_resolution_routes() -> frozenset[str]:
    return frozenset(
        {
            _provider_contract().RESOLUTION_ROUTE_AOT_PACK,
            _provider_contract().RESOLUTION_ROUTE_DISK,
            _provider_contract().RESOLUTION_ROUTE_MEMORY,
            _provider_contract().RESOLUTION_ROUTE_NVRTC,
        }
    )


def _require_trusted_capture_resolution(
    resolution: Any,
) -> None:
    if resolution.route not in _capture_resolution_routes():
        raise CaptureError(
            "AOT capture only accepts provider-owned NVRTC, memory-cache, "
            "disk-cache, or AOT-pack resolutions. Provider artifacts are "
            "trusted native-code inputs."
        )
    try:
        path = resolution.path
    except AttributeError as exc:
        raise CaptureError("Provider resolution is missing its artifact path.") from exc
    if not isinstance(path, str) or not path:
        raise CaptureError("Provider resolution has an invalid artifact path.")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _warn_parent_fsync_failure(output: Path, error: OSError) -> None:
    try:
        warnings.warn(
            f"AOT pack {output} was published atomically, but synchronizing its "
            f"parent directory failed: {error}",
            RuntimeWarning,
            stacklevel=3,
        )
    except Exception:
        # renameat2 is the publication commit point. Warning policy must not
        # turn an already-visible successful publication into a false failure.
        pass


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise CaptureError(
            "Atomic create-only AOT pack publication requires Linux renameat2."
        ) from exc
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise CaptureError(f"AOT pack output already exists: {destination}")
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise CaptureError(
            "Atomic create-only AOT pack publication is unavailable on this filesystem."
        )
    raise CaptureError(
        f"Failed atomically publishing AOT pack to {destination}."
    ) from OSError(error_number, os.strerror(error_number))


class _CaptureSink:
    def __init__(self, staging: Path):
        self.staging = staging
        self.artifacts = staging / "artifacts"
        self.sources = staging / "sources"
        self.entries: dict[str, _ManifestEntry] = {}
        self.observations = 0
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()

    def record(self, resolution: Any) -> None:
        if os.getpid() != self._owner_pid:
            raise CaptureError(_CAPTURE_PROCESS_BOUNDARY_ERROR)

        _require_trusted_capture_resolution(resolution)
        identity = _manifest_identity_from_request(resolution.request)
        if identity.bundle_format != "ltoir":
            raise CaptureError("AOT capture only supports LTO-IR provider bundles.")
        if (
            resolution.route == _provider_contract().RESOLUTION_ROUTE_DISK
            and resolution.producer_compiler_version is None
        ):
            raise CaptureError(
                "Cannot capture a legacy provider disk-cache artifact without "
                "producer compiler version metadata. Clear the legacy provider "
                "disk cache and compile the bundle again."
            )
        if resolution.producer_compiler != "nvrtc":
            raise CaptureError(
                "AOT capture requires an NVRTC-produced LTO-IR artifact."
            )
        producer_version = _parse_producer_version(resolution.producer_compiler_version)

        source = resolution.request.source.encode("utf-8")
        if len(source) > MAX_SOURCE_BYTES:
            raise CaptureError("Provider source is unexpectedly large.")
        if hashlib.sha256(source).hexdigest() != identity.source_sha256:
            raise CaptureError("Provider source does not match its bundle identity.")
        artifact = _read_resolution_artifact(resolution)
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()

        if set(resolution.layouts_by_expression) != set(identity.layout_expressions):
            raise CaptureError(
                "Resolved provider layouts do not match the exact bundle identity."
            )
        layouts = tuple(
            _ManifestLayout(
                expression=expression,
                size_in_bytes=resolution.layouts_by_expression[
                    expression
                ].size_in_bytes,
                alignment=resolution.layouts_by_expression[expression].alignment,
            )
            for expression in identity.layout_expressions
        )
        symbols = tuple(resolution.request.symbols)
        if tuple(sorted(set(symbols))) != symbols:
            raise CaptureError("Resolved provider symbols must be unique and sorted.")
        entry = _ManifestEntry(
            entry_id=_entry_id(identity, symbols),
            identity=identity,
            artifact_sha256=artifact_sha256,
            artifact_size=len(artifact),
            producer_compiler="nvrtc",
            producer_version=producer_version,
            producer_toolkit_version=resolution.producer_toolkit_version,
            layouts=layouts,
            symbols=symbols,
        )

        with self._lock:
            existing = self.entries.get(entry.entry_id)
            if existing is not None and existing != entry:
                raise CaptureError(
                    f"Conflicting provider artifacts share AOT entry {entry.entry_id}."
                )
            _write_new_file(
                self.sources / f"{identity.source_sha256}.cu",
                source,
                description=f"AOT source {identity.source_sha256}",
            )
            _write_new_file(
                self.artifacts / f"{artifact_sha256}.ltoir",
                artifact,
                description=f"AOT artifact {artifact_sha256}",
            )
            self.entries[entry.entry_id] = entry
            self.observations += 1


class Capture:
    """Context manager that captures resolved exact provider bundles."""

    def __init__(
        self,
        output: str | os.PathLike[str],
        *,
        name: str | None,
    ):
        if name is not None:
            _require_string(name, description="AOT pack name")
        self.output = _normalize_path(output)
        self.name = name
        self._staging: Path | None = None
        self._sink: _CaptureSink | None = None
        self._capture_token = None
        self._observer_context = None
        self._provider_observers_context = None
        self._entered = False
        self._result: CaptureResult | None = None
        self._owner_pid: int | None = None

    @property
    def result(self) -> CaptureResult:
        if self._result is None:
            raise CaptureError("AOT capture has not published a result.")
        return self._result

    def __enter__(self) -> Capture:
        _require_supported_platform()
        if self._entered:
            raise CaptureError("AOT capture contexts cannot be reused.")
        if _ACTIVE_CAPTURE.get() is not None:
            raise CaptureError("Nested AOT capture contexts are not supported.")
        self._entered = True
        self._owner_pid = os.getpid()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists() or self.output.is_symlink():
            raise CaptureError(f"AOT pack output already exists: {self.output}")

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output.name}.staging-",
                dir=self.output.parent,
            )
        )
        sink = _CaptureSink(staging)
        try:
            sink.artifacts.mkdir()
            sink.sources.mkdir()
            self._staging = staging
            self._sink = sink
            self._capture_token = _ACTIVE_CAPTURE.set(self)
            from ._compiler import _bundle as _provider_bundle

            _assert_provider_constants()
            self._provider_observers_context = (
                _provider_bundle._ACTIVE_POST_RESOLUTION_OBSERVERS
            )
            self._observer_context = (
                _provider_bundle.activate_bundle_resolution_observer(sink.record)
            )
            self._observer_context.__enter__()
        except BaseException:
            if self._capture_token is not None:
                _ACTIVE_CAPTURE.reset(self._capture_token)
                self._capture_token = None
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        assert self._staging is not None
        assert self._sink is not None
        assert self._owner_pid is not None
        crossed_process_boundary = os.getpid() != self._owner_pid
        try:
            if self._observer_context is not None:
                self._observer_context.__exit__(
                    exception_type,
                    exception,
                    traceback,
                )
        finally:
            self._observer_context = None
            self._provider_observers_context = None
            if self._capture_token is not None:
                _ACTIVE_CAPTURE.reset(self._capture_token)
                self._capture_token = None

        if crossed_process_boundary:
            raise CaptureError(_CAPTURE_PROCESS_BOUNDARY_ERROR)

        if exception_type is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            return False

        try:
            if not self._sink.entries:
                raise CaptureError("AOT capture observed no provider bundles.")
            manifest = _Manifest(
                name=self.name,
                writer_version=_writer_version(),
                entries=tuple(
                    self._sink.entries[entry_id]
                    for entry_id in sorted(self._sink.entries)
                ),
            )
            _write_new_file(
                self._staging / MANIFEST_NAME,
                _canonical_json_bytes(_manifest_payload(manifest)),
                description="AOT pack manifest",
            )
            _fsync_directory(self._sink.artifacts)
            _fsync_directory(self._sink.sources)
            _fsync_directory(self._staging)
            loaded = _load_pack(self._staging)
            info = _pack_info(self.output, loaded.manifest)
            result = CaptureResult(
                path=info.path,
                name=info.name,
                observations=self._sink.observations,
                entries=info.entries,
            )
            _rename_noreplace(self._staging, self.output)
            try:
                _fsync_directory(self.output.parent)
            except OSError as exc:
                _warn_parent_fsync_failure(self.output, exc)
            self._result = result
        except BaseException:
            shutil.rmtree(self._staging, ignore_errors=True)
            raise
        return False


def capture(
    output: str | os.PathLike[str],
    *,
    name: str | None = None,
) -> Capture:
    """Capture all provider-owned exact bundles resolved in the context.

    Captured LTO-IR is a trusted native-code input. The capture observer accepts
    only artifacts produced or re-exported through internal provider routes.
    """

    return Capture(output, name=name)


def _writer_version() -> str:
    try:
        return importlib.metadata.version("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


__all__ = [
    "Capture",
    "CaptureError",
    "CaptureResult",
    "EntryInfo",
    "PackError",
    "PackInfo",
    "PackIntegrityError",
    "PackMissError",
    "capture",
    "inspect",
    "use",
]
