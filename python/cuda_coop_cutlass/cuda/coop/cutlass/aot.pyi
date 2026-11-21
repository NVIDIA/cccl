# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Types for ahead-of-time cooperative provider capture and selection."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from os import PathLike
from typing import Literal

class PackError(RuntimeError): ...
class PackIntegrityError(PackError): ...
class PackMissError(PackError): ...
class CaptureError(PackError): ...

@dataclass(frozen=True)
class EntryInfo:
    key: str
    arch: str
    producer_version: tuple[int, int]
    artifact_sha256: str
    artifact_size: int
    layouts: tuple[tuple[str, int, int], ...]

@dataclass(frozen=True)
class PackInfo:
    schema_version: int
    name: str | None
    entries: tuple[EntryInfo, ...]

    @property
    def artifact_bytes(self) -> int: ...

def inspect(pack: str | PathLike[str]) -> PackInfo: ...
def capture(
    output: str | PathLike[str], *, name: str | None = None
) -> AbstractContextManager[None]: ...
def use(
    pack: str | PathLike[str],
    *,
    mode: Literal["auto", "required", "off"] = "auto",
) -> AbstractContextManager[PackInfo | None]: ...
