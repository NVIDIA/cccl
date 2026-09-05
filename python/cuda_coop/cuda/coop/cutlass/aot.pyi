# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed CUTLASS cooperative-provider AOT pack API."""

from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Literal

class PackError(RuntimeError):
    """Base error for CUTLASS cooperative AOT packs."""

class PackIntegrityError(PackError):
    """A pack is malformed, corrupt, or unsupported."""

class PackMissError(PackError):
    """A required pack has no compatible exact-bundle entry."""

class CaptureError(PackError):
    """A provider resolution cannot be captured or published."""

class EntryInfo:
    """Portable information about one exact provider bundle."""

    entry_id: str
    source_sha256: str
    artifact_sha256: str
    artifact_size: int
    identity_version: int
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

    def __init__(
        self,
        entry_id: str,
        source_sha256: str,
        artifact_sha256: str,
        artifact_size: int,
        identity_version: int,
        provider_abi_version: int,
        bundle_format: str,
        compute_arch: str,
        sm_arch: str,
        compiler_options: tuple[str, ...],
        layout_expressions: tuple[str, ...],
        symbols: tuple[str, ...],
        producer_compiler: str,
        producer_version: tuple[int, int],
        producer_toolkit_version: str | None,
    ) -> None:
        """Describe one validated provider artifact and its exact identity."""

class PackInfo:
    """Validated information about one relocatable AOT pack."""

    path: Path
    name: str | None
    schema_version: int
    provider_abi_version: int
    writer_version: str
    entries: tuple[EntryInfo, ...]

    def __init__(
        self,
        path: Path,
        name: str | None,
        schema_version: int,
        provider_abi_version: int,
        writer_version: str,
        entries: tuple[EntryInfo, ...],
    ) -> None:
        """Describe one validated pack and its entries."""

    @property
    def artifact_bytes(self) -> int:
        """Return total provider artifact bytes in the pack."""

class CaptureResult:
    """Result published by a successful capture context."""

    path: Path
    name: str | None
    observations: int
    entries: tuple[EntryInfo, ...]

    def __init__(
        self,
        path: Path,
        name: str | None,
        observations: int,
        entries: tuple[EntryInfo, ...],
    ) -> None:
        """Describe one published capture and its observed entries."""

    @property
    def artifact_bytes(self) -> int:
        """Return total provider artifact bytes in the capture."""

class Capture:
    """Context manager that captures resolved exact provider bundles."""

    output: Path
    name: str | None

    def __init__(
        self,
        output: str | PathLike[str],
        *,
        name: str | None,
    ) -> None:
        """Prepare a create-only capture at ``output``."""

    @property
    def result(self) -> CaptureResult:
        """Return the published result after a successful context exit."""

    def __enter__(self) -> Capture:
        """Begin observing exact provider-bundle resolutions."""

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Publish the capture atomically, or discard it after failure."""

def capture(
    output: str | PathLike[str],
    *,
    name: str | None = None,
) -> Capture:
    """Capture all provider-owned exact bundles resolved in the context."""

def inspect(pack: str | PathLike[str]) -> PackInfo:
    """Validate and describe one AOT pack."""

def use(
    pack: str | PathLike[str],
    *,
    mode: Literal["auto", "required", "off"] = "auto",
) -> Iterator[PackInfo | None]:
    """Select one exact-bundle pack for provider resolution in this context."""

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
