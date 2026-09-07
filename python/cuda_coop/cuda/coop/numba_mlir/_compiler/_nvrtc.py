# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile direct CUB providers to ephemeral NVRTC LTO IR."""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numba_cuda_mlir import cuda

from cuda.coop._headers import resolve_include_paths
from cuda.coop._headers._identity import include_dirs_identity
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

_REQUIRED_HEADERS = (
    "cub/block/block_reduce.cuh",
    "cuda/functional",
    "cuda/std/cstdint",
    "cuda/std/functional",
)


class NvrtcCompilationError(RuntimeError):
    """NVRTC rejected a generated ``cuda.coop`` provider."""


@dataclass(frozen=True)
class CompileContext:
    """Exact target, headers, and compiler-library identity for one provider."""

    nvrtc_path: str
    nvrtc_builtins_path: str | None
    nvrtc_version: tuple[int, int]
    include_dirs: tuple[str, ...]
    header_identity: str
    architecture: str


def _load_nvrtc():
    """Import the bindings only after the selected toolkit is preloaded."""

    from cuda.bindings import nvrtc

    return nvrtc


def _check(nvrtc: Any, result: Any, *, program=None, operation: str) -> None:
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


def _version(nvrtc: Any) -> tuple[int, int]:
    error, major, minor = nvrtc.nvrtcVersion()
    _check(nvrtc, error, operation="version query")
    return int(major), int(minor)


def _target_architecture(state: Any = None) -> str:
    if state is not None:
        metadata = getattr(state, "metadata", None)
        targetoptions = (
            metadata.get("targetoptions") if isinstance(metadata, dict) else None
        )
        chip = targetoptions.get("chip") if isinstance(targetoptions, dict) else None
        if chip is not None:
            if not isinstance(chip, str):
                raise TypeError("Numba-CUDA-MLIR target chip must be a string")
            match = re.fullmatch(r"sm_([1-9][0-9]*(?:a|f)?)", chip)
            if match is None:
                raise ValueError(f"unsupported Numba-CUDA-MLIR target chip {chip!r}")
            return match.group(1)
    major, minor = cuda.get_current_device().compute_capability
    return f"{int(major)}{int(minor)}"


def resolve_compile_context(state: Any = None) -> CompileContext:
    """Resolve a fail-closed compiler identity before provider caching."""

    paths = resolve_include_paths(
        start=Path(__file__),
        required_headers=_REQUIRED_HEADERS,
    )
    include_dirs = tuple(os.fspath(path) for path in paths.as_tuple())
    libraries = preload_toolkit_compiler_libraries(paths.cuda)
    nvrtc = _load_nvrtc()
    nvrtc_version = _version(nvrtc)
    validate_nvrtc_version(libraries, nvrtc_version)
    identity = include_dirs_identity(include_dirs)
    return CompileContext(
        nvrtc_path=libraries.nvrtc_path,
        nvrtc_builtins_path=libraries.nvrtc_builtins_path,
        nvrtc_version=nvrtc_version,
        include_dirs=include_dirs,
        header_identity=identity.digest,
        architecture=_target_architecture(state),
    )


@functools.lru_cache(maxsize=None)
def compile_lto_ir(
    source: str,
    context: CompileContext,
    program_name: str = "cuda_coop_block_reduce.cu",
) -> bytes:
    """Compile one source string to LTO IR, cached only in this process."""

    nvrtc = _load_nvrtc()
    loaded_version = _version(nvrtc)
    if loaded_version != context.nvrtc_version:
        raise RuntimeError(
            "loaded NVRTC version changed after compile-context resolution: "
            f"expected {context.nvrtc_version}, got {loaded_version}"
        )
    options = [
        b"--std=c++17",
        b"--relocatable-device-code=true",
        b"-dlto",
        f"--gpu-architecture=compute_{context.architecture}".encode(),
        *[
            os.fsencode(f"--include-path={include_dir}")
            for include_dir in context.include_dirs
        ],
    ]
    error, program = nvrtc.nvrtcCreateProgram(
        source.encode(), os.fsencode(program_name), 0, [], []
    )
    _check(nvrtc, error, operation="program creation")
    try:
        (error,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
        _check(nvrtc, error, program=program, operation="LTO compilation")
        error, size = nvrtc.nvrtcGetLTOIRSize(program)
        _check(nvrtc, error, program=program, operation="LTO size query")
        image = bytearray(size)
        (error,) = nvrtc.nvrtcGetLTOIR(program, image)
        _check(nvrtc, error, program=program, operation="LTO extraction")
        return bytes(image)
    finally:
        nvrtc.nvrtcDestroyProgram(program)


__all__ = [
    "CompileContext",
    "NvrtcCompilationError",
    "compile_lto_ir",
    "resolve_compile_context",
]
