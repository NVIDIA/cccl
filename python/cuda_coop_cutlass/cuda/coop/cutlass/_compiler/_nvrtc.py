# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile generated C++ with the CUDA toolkit selected by its headers."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from multiprocessing.util import Finalize
from pathlib import Path
from typing import Any

from cuda.coop.cutlass._headers import resolve_include_paths
from cuda.coop.cutlass._headers._identity import include_dirs_identity
from cuda.coop.cutlass._headers._toolkit import (
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
        configured_roots=(os.environ.get("CUDA_COOP_CUTLASS_CCCL_ROOT"),),
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


def bundle_options(arch: str) -> tuple[bytes, ...]:
    """Code-generation options shared by JIT compilation and AOT lookup."""

    return (
        b"--std=c++17",
        b"--relocatable-device-code=true",
        b"-default-device",
        b"-dlto",
        b"-DCCCL_DISABLE_BF16_SUPPORT",
        f"--gpu-architecture={arch}".encode("ascii"),
    )


def compiler_options(context: CompileContext, arch: str) -> tuple[bytes, ...]:
    """Build device-only C++17 options that emit relocatable LTO-IR.

    The architecture and ordered include roots are part of bundle identity.
    These options request external device code for the later kernel link.
    """

    return bundle_options(arch) + tuple(
        os.fsencode(f"--include-path={path}") for path in context.include_dirs
    )


_PCH_ENV = "CUDA_COOP_CUTLASS_PROVIDER_PCH"
_PCH_DIRECTORIES: dict[str, str | None] = {}
_PCH_LOCK = threading.Lock()


def _reset_pch_after_fork() -> None:
    # The child must neither reuse nor remove the parent's directories.
    global _PCH_DIRECTORIES, _PCH_LOCK
    _PCH_DIRECTORIES = {}
    _PCH_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pch_after_fork)


def pch_options(context: CompileContext) -> tuple[bytes, ...]:
    """Select a process-private directory for automatic NVRTC PCH reuse.

    NVRTC checks the source prefix, target and compiler options, but does
    not check header contents. Give each header digest its own directory.
    These flags apply only to a JIT compile and do not identify its LTO-IR.
    """

    mode = os.environ.get(_PCH_ENV, "").strip().lower() or "auto"
    if mode not in ("auto", "off"):
        raise ValueError(f"{_PCH_ENV} must be 'auto' or 'off', got {mode!r}")
    if mode == "off" or context.nvrtc_version < (12, 8):
        return ()
    with _PCH_LOCK:
        if context.header_identity not in _PCH_DIRECTORIES:
            try:
                directory = tempfile.mkdtemp(prefix="cuda-coop-cutlass-pch-")
            except OSError:
                return ()
            # Multiprocessing fork workers on Python <= 3.12 exit via
            # os._exit; Finalize runs there as well as at interpreter exit.
            Finalize(
                None, shutil.rmtree, args=(directory, True), exitpriority=0
            )
            _PCH_DIRECTORIES[context.header_identity] = directory
        selected = _PCH_DIRECTORIES[context.header_identity]
        if selected is None:
            return ()
    return b"--pch", b"--pch-dir=" + os.fsencode(selected)


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


def _disable_pch(options: tuple[bytes, ...], reason: str) -> None:
    # The registered finalizer still owns the directory. In-flight programs
    # may also be using it, so leave it in place until process exit.
    with _PCH_LOCK:
        for digest, directory in _PCH_DIRECTORIES.items():
            if directory and b"--pch-dir=" + os.fsencode(directory) in options:
                _PCH_DIRECTORIES[digest] = None
                break
        else:
            return
    logging.getLogger(__name__).warning(
        "Disabling NVRTC PCH for this header set: %s", reason
    )


def _compile_program(
    nvrtc: Any, program: Any, options: tuple[bytes, ...]
) -> None:
    result = nvrtc.nvrtcResult
    error = nvrtc.nvrtcCompileProgram(program, len(options), list(options))[0]
    if b"--pch" in options:
        status = nvrtc.nvrtcGetPCHCreateStatus(program)[0]
        if error == result.NVRTC_SUCCESS:
            # A failure to write an automatic PCH can still produce valid IR.
            if status in (
                result.NVRTC_ERROR_PCH_CREATE,
                result.NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED,
            ):
                _disable_pch(options, status.name)
            return

        log = _program_log(nvrtc, program)
        # A completed checkpoint followed by a frontend compilation error
        # already provides the source diagnostic. A hit and an unsuccessful
        # attempt are ambiguous, so let one plain compile disambiguate them.
        if (
            error == result.NVRTC_ERROR_COMPILATION
            and status == result.NVRTC_SUCCESS
        ):
            raise RuntimeError(f"CUTLASS provider compilation failed:\n{log}")

        # Unknown compiler failures (including localized diagnostics) get a
        # single plain attempt. Reuse the program to preserve registered probes.
        plain = [
            option
            for option in options
            if option != b"--pch" and not option.startswith(b"--pch-dir=")
        ]
        error = nvrtc.nvrtcCompileProgram(program, len(plain), plain)[0]
        if error == result.NVRTC_SUCCESS:
            _disable_pch(options, log.strip() or "PCH compilation failed")
            return
        raise RuntimeError(
            "CUTLASS provider compilation failed with PCH:\n"
            f"{log}\nRetry without PCH:\n{_program_log(nvrtc, program)}"
        )
    if error != result.NVRTC_SUCCESS:
        raise RuntimeError(
            "CUTLASS provider compilation failed:\n"
            f"{_program_log(nvrtc, program)}"
        )


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
        _compile_program(nvrtc, program, options)
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
