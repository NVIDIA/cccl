# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read and publish relocatable packs of exact provider LTO-IR bundles."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from ._compiler._layout import _validate_storage_layout
from ._compiler._types import ScratchLayout


class PackError(RuntimeError):
    """A provider pack could not be captured or consumed."""


class PackIntegrityError(PackError):
    """The pack format, metadata, or artifact bytes failed validation."""


class PackMissError(PackError):
    """A required pack has no compatible entry for a provider bundle."""


class CaptureError(PackError):
    """The selected workload could not publish its provider bundles."""


@dataclass(frozen=True)
class EntryInfo:
    """Describe one exact bundle and its compiled scratch layouts.

    Each layout contains the query expression, size in bytes, and alignment.
    The producer version contains the NVRTC major and minor version.
    """

    key: str
    arch: str
    producer_version: tuple[int, int]
    artifact_sha256: str
    artifact_size: int
    layouts: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class PackInfo:
    """Summarize the validated entries in a provider pack."""

    schema_version: int
    name: str | None
    entries: tuple[EntryInfo, ...]

    @property
    def artifact_bytes(self) -> int:
        """Total LTO-IR bytes across all entries."""
        return sum(entry.artifact_size for entry in self.entries)


PACK_PATH_ENV = "CUDA_COOP_CUTLASS_AOT_PACK_PATH"
PACK_MODE_ENV = "CUDA_COOP_CUTLASS_AOT_MODE"
_Entries = dict[str, tuple[EntryInfo, bytes]]
_SELECTION: ContextVar[tuple[str, _Entries] | None] = ContextVar(
    "cuda_coop_cutlass_aot_selection", default=None
)
_CAPTURE: ContextVar[tuple[int, _Entries] | None] = ContextVar(
    "cuda_coop_cutlass_aot_capture", default=None
)
_SCHEMA = 1
_DIGEST = re.compile(r"[0-9a-f]{64}")
_LOGGER = logging.getLogger(__name__)


def _checksum(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise PackIntegrityError("Duplicate field in the AOT pack manifest")
    return result


def _read_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PackIntegrityError(f"Expected a regular AOT pack file: {path}")
    return path.read_bytes()


def _decode_entry(data: dict) -> EntryInfo:
    key = data["key"]
    arch = data["arch"]
    digest = data["artifact_sha256"]
    size = data["artifact_size"]
    version = data["producer_version"]
    if (
        not _DIGEST.fullmatch(key)
        or not _DIGEST.fullmatch(digest)
        or not isinstance(arch, str)
        or not arch
        or type(size) is not int
        or size <= 0
        or not isinstance(version, list)
        or len(version) != 2
        or any(type(part) is not int or part < 0 for part in version)
        or not isinstance(data["layouts"], list)
    ):
        raise PackIntegrityError("Invalid AOT bundle metadata")
    layouts = []
    expressions = set()
    for expression, size_in_bytes, alignment in data["layouts"]:
        if (
            not isinstance(expression, str)
            or not expression
            or expression in expressions
        ):
            raise PackIntegrityError("Invalid or duplicate layout expression")
        _validate_storage_layout(size_in_bytes, alignment, description=key)
        layouts.append((expression, size_in_bytes, alignment))
        expressions.add(expression)
    return EntryInfo(
        key, arch, (version[0], version[1]), digest, size, tuple(layouts)
    )


def _load(path: str | os.PathLike[str]) -> tuple[PackInfo, _Entries]:
    root = Path(path).expanduser()
    try:
        manifest = json.loads(
            _read_file(root / "manifest.json"), object_pairs_hook=_unique_fields
        )
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "name",
            "entries",
            "sha256",
        }:
            raise PackIntegrityError("Invalid AOT pack manifest")
        digest = manifest.pop("sha256")
        if digest != _checksum(manifest):
            raise PackIntegrityError("AOT pack manifest checksum mismatch")
        schema = manifest["schema_version"]
        name = manifest["name"]
        if type(schema) is not int or schema != _SCHEMA:
            raise PackIntegrityError(f"Unsupported AOT pack schema {schema!r}")
        if name is not None and not isinstance(name, str):
            raise PackIntegrityError("Invalid AOT pack name")
        if not isinstance(manifest["entries"], list) or not manifest["entries"]:
            raise PackIntegrityError("Expected a nonempty list of AOT bundles")
        entries: _Entries = {}
        for data in manifest["entries"]:
            info = _decode_entry(data)
            if info.key in entries:
                raise PackIntegrityError(f"Duplicate AOT bundle {info.key}")
            blob = _read_file(root / f"{info.key}.ltoir")
            if (
                len(blob) != info.artifact_size
                or hashlib.sha256(blob).hexdigest() != info.artifact_sha256
            ):
                raise PackIntegrityError(f"Corrupt AOT artifact {info.key}")
            entries[info.key] = info, blob
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PackIntegrityError(f"Invalid AOT pack {root}: {exc}") from exc
    return PackInfo(
        schema, name, tuple(i for i, _ in entries.values())
    ), entries


def inspect(pack: str | os.PathLike[str]) -> PackInfo:
    """Validate a pack manifest, scratch layouts, and every LTO-IR artifact."""
    return _load(pack)[0]


@contextmanager
def use(
    pack: str | os.PathLike[str],
    *,
    mode: Literal["auto", "required", "off"] = "auto",
) -> Iterator[PackInfo | None]:
    """Select a trusted pack for provider compilation in the current context.

    An exact hit supplies validated LTO-IR and layouts from an in-memory
    snapshot. On a miss, auto mode uses normal compilation and required mode
    raises PackMissError. During trace finalization, CuTe raises DSLRuntimeError
    with the pack error as its __cause__. Off mode suspends outer selection.
    """
    _validate_mode(mode)
    info = None
    selection: tuple[str, _Entries] = (mode, {})
    if mode != "off":
        info, entries = _load(pack)
        selection = (mode, entries)
    token = _SELECTION.set(selection)
    try:
        yield info
    finally:
        _SELECTION.reset(token)


def _publish(source: Path, destination: Path) -> None:
    """Atomically rename a complete staged directory to an unused name."""
    if os.name == "nt":
        # Windows rename always fails when the destination exists.
        source.rename(destination)
        return
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable") from exc
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    # Linux AT_FDCWD and RENAME_NOREPLACE.
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _check_capture_output(path: Path) -> None:
    if os.path.lexists(path):
        raise CaptureError(f"AOT pack output already exists: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            source = Path(temp) / "source"
            target = Path(temp) / "target"
            source.mkdir()
            _publish(source, target)
            source.mkdir()
            try:
                _publish(source, target)
            except FileExistsError:
                return
    except OSError as exc:
        raise CaptureError(
            f"Cannot publish a new AOT pack under {path.parent} "
            f"with an atomic no-replace rename: {exc}"
        ) from exc
    raise CaptureError(
        f"AOT capture requires atomic no-replace renames under {path.parent}"
    )


@contextmanager
def capture(
    output: str | os.PathLike[str], *, name: str | None = None
) -> Iterator[None]:
    """Collect resolved providers in this context and publish a new pack.

    Run the workload's compilation inside this context. If it raises or
    resolves no bundles, capture leaves the destination untouched.
    """
    if _CAPTURE.get() is not None:
        raise CaptureError("Nested AOT captures are unsupported")
    if name is not None and not isinstance(name, str):
        raise TypeError("AOT pack name must be a string")
    path = Path(output).expanduser().absolute()
    _check_capture_output(path)
    entries: _Entries = {}
    owner = os.getpid()
    token = _CAPTURE.set((owner, entries))
    try:
        yield
    finally:
        _CAPTURE.reset(token)
    if os.getpid() != owner:
        raise CaptureError("AOT capture crossed a process boundary")
    if not entries:
        raise CaptureError("AOT capture observed no provider bundles")
    ordered = tuple(entries[key] for key in sorted(entries))
    info = PackInfo(_SCHEMA, name, tuple(entry for entry, _ in ordered))
    manifest = asdict(info)
    manifest["sha256"] = _checksum(manifest)
    try:
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            staging = Path(temp) / "pack"
            staging.mkdir()
            for entry, blob in ordered:
                (staging / f"{entry.key}.ltoir").write_bytes(blob)
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            _publish(staging, path)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise CaptureError(
                f"AOT pack output already exists: {path}"
            ) from exc
        raise CaptureError(f"Failed to publish AOT pack {path}: {exc}") from exc


def _validate_mode(mode: str) -> None:
    if mode not in {"auto", "required", "off"}:
        raise ValueError(f"Unsupported AOT pack mode {mode!r}")


@lru_cache(maxsize=1)
def _environment_entries(path: str) -> _Entries:
    # An environment-selected pack stays a validated snapshot in this process.
    return _load(path)[1]


def _selection() -> tuple[str, _Entries] | None:
    selected = _SELECTION.get()
    if selected is not None:
        return None if selected[0] == "off" else selected
    mode = os.environ.get(PACK_MODE_ENV, "").strip().lower() or "auto"
    _validate_mode(mode)
    if mode == "off":
        return None
    path = os.environ.get(PACK_PATH_ENV)
    if not path:
        if mode == "required":
            raise PackMissError(
                f"{PACK_MODE_ENV}=required needs {PACK_PATH_ENV}"
            )
        return None
    if not os.path.isabs(path):
        raise PackError(f"{PACK_PATH_ENV} must be absolute, got {path!r}")
    return mode, _environment_entries(path)


def active() -> bool:
    return _selection() is not None or capturing()


def capturing() -> bool:
    return _CAPTURE.get() is not None


def record(
    key: str,
    arch: str,
    blob: bytes,
    layouts: Mapping[str, ScratchLayout],
    producer_version: tuple[int, int],
) -> None:
    capture_state = _CAPTURE.get()
    if capture_state is None:
        return
    owner, entries = capture_state
    if owner != os.getpid():
        raise CaptureError("AOT capture crossed a process boundary")
    if not _DIGEST.fullmatch(key) or not blob:
        raise CaptureError("Expected an exact bundle key and nonempty LTO-IR")
    info = EntryInfo(
        key,
        arch,
        producer_version,
        hashlib.sha256(blob).hexdigest(),
        len(blob),
        tuple(
            (expression, layout.size_in_bytes, layout.alignment)
            for expression, layout in sorted(layouts.items())
        ),
    )
    entry = (info, blob)
    if entries.setdefault(key, entry) != entry:
        raise CaptureError(f"Conflicting artifacts for AOT bundle {key}")


def lookup(
    key: str, arch: str, expressions: tuple[str, ...]
) -> tuple[bytes, dict[str, ScratchLayout], tuple[int, int]] | None:
    selection = _selection()
    if selection is None:
        return None
    mode, entries = selection
    reason = "entry is absent"
    if key in entries:
        info, blob = entries[key]
        layouts = {
            expr: ScratchLayout(size, align)
            for expr, size, align in info.layouts
        }
        if info.arch != arch or set(expressions) != layouts.keys():
            raise PackIntegrityError(f"Inconsistent metadata for bundle {key}")
        try:
            from cutlass import CUDA_VERSION

            major, minor = CUDA_VERSION.major, CUDA_VERSION.minor
        except (ImportError, AttributeError) as exc:
            reason = f"cannot determine the CuTe toolchain version: {exc}"
        else:
            producer_major, producer_minor = info.producer_version
            if major == producer_major and minor >= producer_minor:
                return blob, layouts, info.producer_version
            reason = (
                f"producer NVRTC {producer_major}.{producer_minor} and "
                f"consumer CuTe toolchain {major}.{minor} are incompatible"
            )
    if mode == "required":
        raise PackMissError(f"Required AOT bundle {key} ({arch}): {reason}")
    _LOGGER.debug(
        "AOT bundle %s (%s) missed (%s), using the provider cache or JIT",
        key,
        arch,
        reason,
    )
    return None
