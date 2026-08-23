# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed CUTLASS cooperative-provider AOT pack API."""

from contextlib import AbstractContextManager
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
    ) -> None: ...

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
    ) -> None: ...
    @property
    def artifact_bytes(self) -> int: ...

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
    ) -> None: ...
    @property
    def artifact_bytes(self) -> int: ...

class Capture:
    """Context manager that captures resolved exact provider bundles."""

    output: Path
    name: str | None

    def __init__(
        self,
        output: str | PathLike[str],
        *,
        name: str | None,
    ) -> None: ...
    @property
    def result(self) -> CaptureResult: ...
    def __enter__(self) -> Capture: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...

def capture(
    output: str | PathLike[str],
    *,
    name: str | None = None,
) -> Capture: ...
def inspect(pack: str | PathLike[str]) -> PackInfo: ...
def use(
    pack: str | PathLike[str],
    *,
    mode: Literal["auto", "required", "off"] = "auto",
) -> AbstractContextManager[PackInfo | None]: ...

__all__: list[str]
