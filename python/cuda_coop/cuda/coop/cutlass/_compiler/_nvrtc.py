# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile generated C++ with the CUDA toolkit selected by its headers."""

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
    """Record header and compiler-library identity for an NVRTC cache key.

    Paths and versions keep a bundle tied to the toolkit selected by its
    headers. The header digest detects changed contents within the include
    roots.
    """

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
    """Import bindings after the selected toolkit libraries are preloaded."""

    from cuda.bindings import nvrtc

    return nvrtc


def resolve_compile_context(
    required_headers: tuple[str, ...],
) -> CompileContext:
    """Resolve headers and bind compiler libraries to their toolkit.

    Validate the loaded NVRTC version before recording the context used for
    compilation and cache identity. Link-library identity is retained because
    the resulting LTO-IR will be consumed by the toolkit linker.
    """

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
    """Build device-only C++17 options that emit relocatable LTO-IR.

    The architecture and ordered include roots are part of bundle identity.
    These options request external device code for the later kernel link.
    """

    return (
        b"--std=c++17",
        b"--relocatable-device-code=true",
        b"-default-device",
        b"-dlto",
        b"-DCCCL_DISABLE_BF16_SUPPORT",
        f"--gpu-architecture={arch}".encode("ascii"),
        *(
            os.fsencode(f"--include-path={path}")
            for path in context.include_dirs
        ),
    )


def _program_log(nvrtc: Any, program: Any) -> str:
    """Read NVRTC diagnostics, or explain why retrieval failed."""

    error, size = nvrtc.nvrtcGetProgramLogSize(program)
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return f"Cannot retrieve NVRTC diagnostic size: {error}"
    log = bytearray(size)
    error = nvrtc.nvrtcGetProgramLog(program, log)[0]
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return f"Cannot retrieve NVRTC diagnostics: {error}"
    return log.rstrip(b"\0").decode("utf-8", errors="replace")


def compile_ltoir(source: str, options: tuple[bytes, ...]) -> bytes:
    """Compile source without layout queries and return its LTO-IR bytes.

    The shared compiler helper reports the NVRTC log on compilation failure
    and destroys the program without replacing an earlier failure.
    """

    return _compile_ltoir(source, options)[0]


def compile_ltoir_with_layouts(
    prepared: _PreparedLayoutProbes, options: tuple[bytes, ...]
) -> tuple[bytes, dict[str, ScratchLayout]]:
    """Compile provider code and evaluate its layouts in the same program.

    ``prepared`` contains the provider source, probe template, and registered
    name expressions; ``options`` selects the compiler settings. The result
    pairs LTO-IR bytes with layouts indexed by those expressions. Registering
    names before compilation makes NVRTC instantiate the requested probes.
    """

    return _compile_ltoir(prepared.source, options, prepared)


def _compile_ltoir(
    source: str,
    options: tuple[bytes, ...],
    prepared: _PreparedLayoutProbes | None = None,
) -> tuple[bytes, dict[str, ScratchLayout]]:
    """Keep probe registration, layout extraction, and code on one program.

    The bundle compiler reaches this helper on a cache miss, after it has
    selected and preloaded the toolkit libraries. The query names encode
    C++ constant values in template arguments, so compilation can return
    storage requirements without executing a device function.

    Queries are registered before compilation and decoded before program
    destruction. With no ``prepared`` probes the layout map is empty. Cleanup
    always destroys the program without replacing an earlier compile or query
    failure with a destruction error.

    Parameters
    ----------
    source : str
        Complete C++ translation unit. With prepared probes, this must be
        the prepared source containing their variable-template declaration.
    options : tuple of bytes
        NVRTC flags selecting the target architecture, headers, and LTO-IR.
    prepared : _PreparedLayoutProbes or None
        Probe names and decoder identity from ``_prepare_layout_probes``.
        None compiles code without requesting layout metadata.

    Returns
    -------
    tuple
        LTO-IR bytes and a map from query expressions to byte sizes and
        alignments, all obtained from this NVRTC program.
    """

    nvrtc = _load_nvrtc()
    error, program = nvrtc.nvrtcCreateProgram(
        source.encode("utf-8"), b"cuda_coop_cutlass_bundle.cu", 0, [], []
    )
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(
            f"Cannot create CUTLASS provider NVRTC program: {error}"
        )
    failed = False
    try:
        expressions = () if prepared is None else prepared.expressions
        for expression in expressions:
            error = nvrtc.nvrtcAddNameExpression(program, expression.encode())[
                0
            ]
            if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(
                    f"Cannot register NVRTC storage layout probe: {error}"
                )
        error = nvrtc.nvrtcCompileProgram(program, len(options), list(options))[
            0
        ]
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(
                "CUTLASS provider compilation failed:\n"
                f"{_program_log(nvrtc, program)}"
            )
        layouts: dict[str, ScratchLayout] = {}
        for expression in expressions:
            error, name = nvrtc.nvrtcGetLoweredName(
                program, expression.encode()
            )
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
            raise RuntimeError(
                f"Cannot retrieve CUTLASS provider LTO-IR size: {error}"
            )
        blob = bytearray(size)
        error = nvrtc.nvrtcGetLTOIR(program, blob)[0]
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(
                f"Cannot retrieve CUTLASS provider LTO-IR: {error}"
            )
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
