# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile direct CUB providers to ephemeral NVRTC LTO IR."""

from __future__ import annotations

import functools
import os
from pathlib import Path

from cuda.bindings import nvrtc

from cuda.coop._headers import resolve_include_paths

_REQUIRED_HEADERS = (
    "cub/block/block_reduce.cuh",
    "cuda/functional",
    "cuda/std/cstdint",
    "cuda/std/functional",
)


class NvrtcCompilationError(RuntimeError):
    """NVRTC rejected a generated ``cuda.coop`` provider."""


def _check(result, *, program=None, operation: str) -> None:
    if result == nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return
    log = ""
    if program is not None:
        try:
            log_error, size = nvrtc.nvrtcGetProgramLogSize(program)
            if log_error == nvrtc.nvrtcResult.NVRTC_SUCCESS and size:
                buffer = bytearray(size)
                (log_error,) = nvrtc.nvrtcGetProgramLog(program, buffer)
                if log_error == nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    log = bytes(buffer).rstrip(b"\0").decode(errors="replace")
        except Exception:  # noqa: BLE001 - preserve the original NVRTC result
            pass
    detail = f"\n{log}" if log else ""
    raise NvrtcCompilationError(f"NVRTC {operation} failed with {result!s}{detail}")


def include_paths() -> tuple[str, ...]:
    """Resolve the exact CCCL and CUDA include directories for providers."""

    paths = resolve_include_paths(
        start=Path(__file__),
        required_headers=_REQUIRED_HEADERS,
    )
    return tuple(os.fspath(path) for path in paths.as_tuple())


def version() -> tuple[int, int]:
    """Return the loaded NVRTC library version."""

    error, major, minor = nvrtc.nvrtcVersion()
    _check(error, operation="version query")
    return int(major), int(minor)


@functools.lru_cache(maxsize=None)
def compile_lto_ir(
    source: str,
    compute_capability: str,
    resolved_include_paths: tuple[str, ...],
) -> bytes:
    """Compile one source string to LTO IR, cached only in this process."""

    options = [
        b"--std=c++17",
        b"--relocatable-device-code=true",
        b"-dlto",
        f"--gpu-architecture=compute_{compute_capability}".encode(),
        *[
            os.fsencode(f"--include-path={include_path}")
            for include_path in resolved_include_paths
        ],
    ]
    error, program = nvrtc.nvrtcCreateProgram(
        source.encode(), b"cuda_coop_block_reduce.cu", 0, [], []
    )
    _check(error, operation="program creation")
    try:
        (error,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
        _check(error, program=program, operation="LTO compilation")
        error, size = nvrtc.nvrtcGetLTOIRSize(program)
        _check(error, program=program, operation="LTO size query")
        image = bytearray(size)
        (error,) = nvrtc.nvrtcGetLTOIR(program, image)
        _check(error, program=program, operation="LTO extraction")
        return bytes(image)
    finally:
        nvrtc.nvrtcDestroyProgram(program)


__all__ = [
    "NvrtcCompilationError",
    "compile_lto_ir",
    "include_paths",
    "version",
]
