# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile generated CUDA wrappers with an identified toolkit and header set.

The backend generates C++ wrappers around CUB operations, but those wrappers
must become device code before a Numba kernel can call them. This module
resolves the compiler libraries and headers, invokes NVRTC, and returns an
LTO IR image or PTX text for later linking. It does not launch GPU work.

The same source can produce different code with different headers, compiler
versions, targets, or options. ``CompileContext`` records the header and
library dependencies; the target and options also participate in cache keys
and private provider symbol identities. Context resolution preloads the chosen
toolkit's libraries before importing the CUDA bindings. ``compile`` dumps
source before cache lookup, so developers can inspect generated code even
when compilation is reused.
"""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda.coop._core._source_dump import dump_source
from cuda.coop._headers import resolve_include_paths
from cuda.coop._headers._identity import include_dirs_identity
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

from ._artifacts import check_in, version
from ._caching import disk_cache

_REQUIRED_HEADERS = (
    "cub/block/block_load.cuh",
    "cub/block/block_store.cuh",
    "cuda/barrier",
    "cuda/devices",
    "cuda/experimental/coop/algorithm",
    "cuda/experimental/coop/group",
    "cuda/functional",
    "cuda/hierarchy",
    "cuda/std/cstdint",
    "cuda/std/functional",
    "cuda/std/type_traits",
)


@dataclass(frozen=True)
class CompileContext:
    """All compiler inputs that participate in provider and cache identity."""

    toolkit_root: str
    toolkit_version: tuple[int, int]
    nvrtc_path: str
    nvrtc_builtins_path: str
    nvjitlink_path: str
    nvrtc_version: version
    nvjitlink_version: tuple[int, int]
    include_dirs: tuple[str, ...]
    header_identity: str

    @property
    def symbol_suffix(self) -> str:
        digest = hashlib.sha256()
        values: tuple[object, ...] = (
            self.toolkit_root,
            self.toolkit_version,
            self.nvrtc_path,
            self.nvrtc_builtins_path,
            self.nvjitlink_path,
            self.nvrtc_version,
            self.nvjitlink_version,
            self.include_dirs,
            self.header_identity,
        )
        for value in values:
            digest.update(repr(value).encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]


def _load_nvrtc():
    """Import CUDA bindings only after exact toolkit libraries are preloaded."""

    import cuda.bindings.nvrtc as _nvrtc_bindings

    return _nvrtc_bindings


def _nvrtc_version(nvrtc: Any) -> version:
    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    return version(int(major), int(minor))


def CHECK_NVRTC(err, prog, *, nvrtc=None):
    nvrtc = _load_nvrtc() if nvrtc is None else nvrtc
    if err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return
    original_err = err
    log_err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
    if log_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(
            f"NVRTC error: {original_err}; failed to get log size: {log_err}"
        )
    log = bytearray(logsize)
    log_result = nvrtc.nvrtcGetProgramLog(prog, log)
    log_err = log_result[0] if isinstance(log_result, tuple) else log_result
    if log_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(
            f"NVRTC error: {original_err}; failed to get log: {log_err}"
        )
    rendered = bytes(log).rstrip(b"\0").decode("ascii", errors="replace")
    raise RuntimeError(f"NVRTC error: {original_err}: {rendered}")


def _dump_source(cpp, cc, code, compiler_options):
    return dump_source(
        cpp,
        backend="numba_mlir",
        identity=(cc, code, compiler_options),
    )


def _include_options(include_dirs: tuple[str, ...]) -> list[bytes]:
    return [os.fsencode(f"--include-path={path}") for path in include_dirs]


def _compiler_options(
    *, cc: int, rdc: bool, code: str, include_dirs: tuple[str, ...]
) -> tuple[bytes, ...]:
    """Return the exact ordered NVRTC option set used for cache identity."""

    check_in("rdc", rdc, [True, False])
    check_in("code", code, ["lto", "ptx"])
    options = [
        b"--std=c++17",
        *_include_options(include_dirs),
        f"--gpu-architecture=compute_{cc}".encode("ascii"),
    ]
    if rdc:
        options.append(b"--relocatable-device-code=true")
    if code == "lto":
        options.append(b"-dlto")
    options.append(b"-DCCCL_DISABLE_BF16_SUPPORT")
    return tuple(options)


def compiler_identity(
    *, context: CompileContext, cc: int, rdc: bool, code: str
) -> tuple[int, bool, str, tuple[bytes, ...]]:
    """Return target and option identity for provider symbols and LTO reuse."""

    return (
        int(cc),
        bool(rdc),
        str(code),
        _compiler_options(
            cc=int(cc),
            rdc=bool(rdc),
            code=str(code),
            include_dirs=context.include_dirs,
        ),
    )


@functools.lru_cache(maxsize=8)
@disk_cache
def compile_impl(
    cpp: str,
    cc: int,
    rdc: bool,
    code: str,
    toolkit_root: str,
    toolkit_version: tuple[int, int],
    nvrtc_path: str,
    nvrtc_builtins_path: str,
    nvjitlink_path: str,
    nvrtc_version: version,
    nvjitlink_version: tuple[int, int],
    include_dirs: tuple[str, ...],
    header_identity: str,
    compiler_options: tuple[bytes, ...],
) -> bytes | str:
    """Compile one source unit using a complete, explicit cache identity.

    The memory and disk cache decorators key all arguments, including toolkit
    paths and header identity that are not otherwise read by the function body.
    Those fields prevent artifacts from different compiler installations or
    header sets from sharing an entry. Callers must resolve and preload the
    matching compiler context before entering this function.

    On a cache miss, verify that the supplied option tuple matches the request
    and that the loaded NVRTC version still matches the context. Compile the
    source, retrieve the requested image, and destroy the NVRTC program on both
    success and failure. A cleanup error does not replace an earlier compilation
    error. Cache hits bypass these body-level checks.

    Parameters
    ----------
    cpp : str
        Complete CUDA C++ translation unit.
    cc : int
        Compute capability encoded as major times ten plus minor.
    rdc : bool
        Whether to enable relocatable device code.
    code : {"lto", "ptx"}
        Requested output format.
    toolkit_root : str
        Selected toolkit root, retained in cache identity.
    toolkit_version : tuple of int
        Selected toolkit version, retained in cache identity.
    nvrtc_path, nvrtc_builtins_path, nvjitlink_path : str
        Exact compiler-library paths from the preloaded context.
    nvrtc_version : version
        Expected loaded NVRTC version.
    nvjitlink_version : tuple of int
        Selected linker version, retained in cache identity.
    include_dirs : tuple of str
        Ordered header search roots.
    header_identity : str
        Header-content identity supplied by context resolution.
    compiler_options : tuple of bytes
        Exact ordered options produced by ``_compiler_options`` for this
        request.

    Returns
    -------
    bytes or str
        LTO image bytes for ``"lto"`` or ASCII-decoded source for ``"ptx"``.

    Raises
    ------
    RuntimeError
        Options or loaded compiler version disagree with the request, or NVRTC
        compilation, image retrieval, or program cleanup fails.
    ValueError
        The requested output format or relocatable-code option is invalid.
    """

    del (
        toolkit_root,
        toolkit_version,
        nvrtc_path,
        nvrtc_builtins_path,
        nvjitlink_path,
        nvjitlink_version,
        header_identity,
    )
    expected_options = _compiler_options(
        cc=cc,
        rdc=rdc,
        code=code,
        include_dirs=include_dirs,
    )
    if compiler_options != expected_options:
        raise RuntimeError(
            "NVRTC compiler-option identity does "
            "not match the requested compile."
        )
    nvrtc = _load_nvrtc()
    loaded_version = _nvrtc_version(nvrtc)
    if loaded_version != nvrtc_version:
        raise RuntimeError(
            "loaded NVRTC version changed after compile-context resolution: "
            f"expected {nvrtc_version}, got {loaded_version}"
        )

    err, prog = nvrtc.nvrtcCreateProgram(cpp.encode(), b"code.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram error: {err}")
    had_error = False
    try:
        (err,) = nvrtc.nvrtcCompileProgram(
            prog, len(compiler_options), list(compiler_options)
        )
        CHECK_NVRTC(err, prog, nvrtc=nvrtc)
        if code == "lto":
            err, size = nvrtc.nvrtcGetLTOIRSize(prog)
            CHECK_NVRTC(err, prog, nvrtc=nvrtc)
            image = bytearray(size)
            (err,) = nvrtc.nvrtcGetLTOIR(prog, image)
            CHECK_NVRTC(err, prog, nvrtc=nvrtc)
            return bytes(image)
        err, size = nvrtc.nvrtcGetPTXSize(prog)
        CHECK_NVRTC(err, prog, nvrtc=nvrtc)
        image = bytearray(size)
        (err,) = nvrtc.nvrtcGetPTX(prog, image)
        CHECK_NVRTC(err, prog, nvrtc=nvrtc)
        return bytes(image).decode("ascii")
    except Exception:
        had_error = True
        raise
    finally:
        (destroy_err,) = nvrtc.nvrtcDestroyProgram(prog)
        if destroy_err != nvrtc.nvrtcResult.NVRTC_SUCCESS and not had_error:
            raise RuntimeError(f"nvrtcDestroyProgram error: {destroy_err}")


def resolve_compile_context() -> CompileContext:
    """Resolve one header/toolkit identity and preload its compiler libraries.

    ``CUDA_COOP_CCCL_ROOT`` optionally selects a CCCL source checkout or
    packaged header bundle instead of automatic source-tree/wheel discovery.
    Read it on each call; unset or empty uses discovery. The header resolver
    expands ``~`` and resolves relative paths from the current working
    directory. A configured root must supply all required CCCL headers; an
    invalid or incomplete selection raises rather than falling back. This
    variable selects CCCL headers, not the CUDA Toolkit installation.

    Resolve CUDA headers separately, then preload NVRTC, its builtins, and
    nvJitLink from that toolkit before importing the CUDA NVRTC bindings.
    Check that the loaded NVRTC version matches the selected toolkit. This
    ordering keeps wrapper compilation and subsequent linking tied to the same
    installation.

    Hash the resolved header roots and their contents into the returned context.
    The context is used both for artifact cache keys and provider symbol
    qualification. Resolution is lazy at its callers; this function itself is
    not memoized and may load process-wide compiler libraries. Algorithms
    retain their resolved context, so changing ``CUDA_COOP_CCCL_ROOT`` affects
    subsequent resolutions, not contexts already held by providers or supplied
    explicitly to ``compile``.

    Returns
    -------
    CompileContext
        Frozen snapshot of exact library paths/versions, ordered include roots,
        and header identity for a provider compilation.
    """

    include_paths = resolve_include_paths(
        start=Path(__file__),
        configured_roots=(os.environ.get("CUDA_COOP_CCCL_ROOT"),),
        required_headers=_REQUIRED_HEADERS,
    )
    include_dirs = tuple(str(path) for path in include_paths.as_tuple())
    libraries = preload_toolkit_compiler_libraries(include_paths.cuda)
    nvrtc = _load_nvrtc()
    loaded_nvrtc_version = _nvrtc_version(nvrtc)
    validate_nvrtc_version(libraries, tuple(loaded_nvrtc_version))
    return CompileContext(
        toolkit_root=libraries.toolkit_root,
        toolkit_version=libraries.toolkit_version,
        nvrtc_path=libraries.nvrtc_path,
        nvrtc_builtins_path=libraries.nvrtc_builtins_path,
        nvjitlink_path=libraries.nvjitlink_path,
        nvrtc_version=loaded_nvrtc_version,
        nvjitlink_version=libraries.nvjitlink_version,
        include_dirs=include_dirs,
        header_identity=include_dirs_identity(include_dirs).digest,
    )


def compile(
    *, context: CompileContext | None = None, **kwargs: Any
) -> tuple[version, bytes | str]:
    """Compile generated provider source with resolved compiler/cache identity.

    Build the ordered options from the selected context, optionally dump the
    source through the shared source-dump hook, and pass every context field to
    ``compile_impl``. Dumping occurs before cache lookup so source inspection
    also works when artifact compilation is reused from memory or disk.

    Parameters
    ----------
    context : CompileContext, optional
        Previously resolved and preloaded compiler context. ``None`` resolves
        one now. Supplying a context reuses its identity; it does not reload the
        library paths stored in it.
    **kwargs : dict
        Required ``cpp`` source string, integer ``cc`` target, boolean ``rdc``,
        and ``code`` equal to ``"lto"`` or ``"ptx"``. These are forwarded to
        ``compile_impl`` with the context and generated compiler options.

    Returns
    -------
    tuple
        ``(nvrtc_version, image)`` with LTO bytes or PTX text according to
        ``code``. The version is taken from the selected context.

    Raises
    ------
    RuntimeError
        Compiler identity checks or NVRTC compilation fail.
    ValueError
        An output format or relocatable-code option is invalid.
    """

    context = resolve_compile_context() if context is None else context
    compiler_options = _compiler_options(
        cc=kwargs["cc"],
        rdc=kwargs["rdc"],
        code=kwargs["code"],
        include_dirs=context.include_dirs,
    )
    _dump_source(kwargs["cpp"], kwargs["cc"], kwargs["code"], compiler_options)
    return context.nvrtc_version, compile_impl(
        **kwargs,
        toolkit_root=context.toolkit_root,
        toolkit_version=context.toolkit_version,
        nvrtc_path=context.nvrtc_path,
        nvrtc_builtins_path=context.nvrtc_builtins_path,
        nvjitlink_path=context.nvjitlink_path,
        nvrtc_version=context.nvrtc_version,
        nvjitlink_version=context.nvjitlink_version,
        include_dirs=context.include_dirs,
        header_identity=context.header_identity,
        compiler_options=compiler_options,
    )
