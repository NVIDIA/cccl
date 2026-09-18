# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile providers with the CUDA toolkit selected by their headers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda.coop._headers import resolve_include_paths
from cuda.coop._headers._identity import include_dirs_identity
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

from ._layout import _decode_layout_probe_name, _PreparedLayoutProbes
from ._types import ScratchLayout


@dataclass(frozen=True)
class CompileContext:
    include_dirs: tuple[str, ...]
    header_identity: str
    toolkit_root: str
    toolkit_version: tuple[int, int]
    nvrtc_path: str
    nvrtc_builtins_path: str
    nvrtc_version: tuple[int, int]
    nvjitlink_path: str
    nvjitlink_version: tuple[int, int]


def _load_nvrtc():
    from cuda.bindings import nvrtc

    return nvrtc


def resolve_compile_context(required_headers: tuple[str, ...]) -> CompileContext:
    paths = resolve_include_paths(
        start=Path(__file__),
        configured_roots=(os.environ.get("CUDA_COOP_CCCL_ROOT"),),
        required_headers=required_headers,
    )
    include_dirs = tuple(map(str, paths.as_tuple()))
    libraries = preload_toolkit_compiler_libraries(paths.cuda)
    nvrtc = _load_nvrtc()
    error, major, minor = nvrtc.nvrtcVersion()
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC version query failed: {error}")
    version = int(major), int(minor)
    validate_nvrtc_version(libraries, version)
    return CompileContext(
        include_dirs,
        include_dirs_identity(include_dirs).digest,
        libraries.toolkit_root,
        libraries.toolkit_version,
        libraries.nvrtc_path,
        libraries.nvrtc_builtins_path,
        version,
        libraries.nvjitlink_path,
        libraries.nvjitlink_version,
    )


def compiler_options(context: CompileContext, arch: str) -> tuple[bytes, ...]:
    return (
        b"--std=c++17",
        b"--relocatable-device-code=true",
        b"-default-device",
        b"-dlto",
        b"-DCCCL_DISABLE_BF16_SUPPORT",
        f"--gpu-architecture={arch}".encode("ascii"),
        *(os.fsencode(f"--include-path={path}") for path in context.include_dirs),
    )


def _program_log(nvrtc: Any, program: Any) -> str:
    error, size = nvrtc.nvrtcGetProgramLogSize(program)
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return f"Cannot retrieve NVRTC diagnostic size: {error}"
    log = bytearray(size)
    error = nvrtc.nvrtcGetProgramLog(program, log)[0]
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return f"Cannot retrieve NVRTC diagnostics: {error}"
    return log.rstrip(b"\0").decode("utf-8", errors="replace")


def compile_ltoir(source: str, options: tuple[bytes, ...]) -> bytes:
    return _compile_ltoir(source, options)[0]


def compile_ltoir_with_layouts(
    prepared: _PreparedLayoutProbes, options: tuple[bytes, ...]
) -> tuple[bytes, dict[str, ScratchLayout]]:
    """Recover all requested layouts from the same program as its LTO-IR."""

    return _compile_ltoir(prepared.source, options, prepared)


def _compile_ltoir(
    source: str,
    options: tuple[bytes, ...],
    prepared: _PreparedLayoutProbes | None = None,
) -> tuple[bytes, dict[str, ScratchLayout]]:
    nvrtc = _load_nvrtc()
    error, program = nvrtc.nvrtcCreateProgram(
        source.encode("utf-8"), b"cuda_coop_cutlass_bundle.cu", 0, [], []
    )
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"Cannot create CUTLASS provider NVRTC program: {error}")
    failed = False
    try:
        expressions = () if prepared is None else prepared.expressions
        for expression in expressions:
            error = nvrtc.nvrtcAddNameExpression(program, expression.encode())[0]
            if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(
                    f"Cannot register NVRTC storage layout probe: {error}"
                )
        error = nvrtc.nvrtcCompileProgram(program, len(options), list(options))[0]
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(
                f"CUTLASS provider compilation failed:\n{_program_log(nvrtc, program)}"
            )
        layouts: dict[str, ScratchLayout] = {}
        for expression in expressions:
            error, name = nvrtc.nvrtcGetLoweredName(program, expression.encode())
            if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(
                    f"Cannot retrieve NVRTC storage layout probe: {error}"
                )
            assert prepared is not None
            layouts[expression] = _decode_layout_probe_name(
                name, symbol=prepared.symbol, expression=expression
            )
        error, size = nvrtc.nvrtcGetLTOIRSize(program)
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS or size <= 0:
            raise RuntimeError(f"Cannot retrieve CUTLASS provider LTO-IR size: {error}")
        blob = bytearray(size)
        error = nvrtc.nvrtcGetLTOIR(program, blob)[0]
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"Cannot retrieve CUTLASS provider LTO-IR: {error}")
        return bytes(blob), layouts
    except BaseException:
        failed = True
        raise
    finally:
        error = nvrtc.nvrtcDestroyProgram(program)[0]
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS and not failed:
            raise RuntimeError(
                f"Cannot destroy CUTLASS provider NVRTC program: {error}"
            )
