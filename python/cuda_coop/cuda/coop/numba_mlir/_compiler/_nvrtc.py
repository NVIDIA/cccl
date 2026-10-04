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

``compile_with_layouts`` reads C++ scratch sizes and alignments from the same
NVRTC program that emits the provider image. An appended probe encodes each
value in a compiler-generated name. The image and its ordered layouts are
cached as one result, so a cache hit supplies both without another compile
or a separate PTX link.
"""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

from cuda.coop._core._source_dump import dump_source
from cuda.coop._headers import resolve_include_paths
from cuda.coop._headers._identity import include_dirs_identity
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

from ._artifacts import check_in, version
from ._caching import disk_cache
from ._layout import decode_layout_name, prepare_layout_queries

_LayoutResult: TypeAlias = tuple[bytes, tuple[tuple[int, int], ...]]

_REQUIRED_HEADERS = (
    "cub/block/block_load.cuh",
    "cub/block/block_store.cuh",
    "cuda/barrier",
    "cuda/devices",
    "cuda/functional",
    "cuda/hierarchy",
    "cuda/std/cstdint",
    "cuda/std/functional",
    "cuda/std/type_traits",
)


@dataclass(frozen=True)
class CompileContext:
    """Record header and toolkit inputs shared by provider compilations.

    Exact library paths and versions identify the selected compiler and
    linker. ``include_dirs`` preserves header search order, while
    ``header_identity`` identifies their contents. Source, target, and
    compiler options enter each compilation's cache key separately.
    """

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
        """Return a short digest of every toolkit and header field.

        Raw C-ABI helpers add it to their symbols to distinguish code built
        with different headers or toolkits.
        """

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


@dataclass(frozen=True)
class CompilerIdentity:
    """Identify the target and options shared by compatible providers.

    The options include ordered header paths. Callers retain the corresponding
    ``CompileContext`` separately for its library and header-content identity.
    """

    cc: int
    rdc: bool
    code: str
    compiler_options: tuple[bytes, ...]


def _load_nvrtc():
    """Import CUDA bindings after the selected toolkit is preloaded."""

    import cuda.bindings.nvrtc as _nvrtc_bindings

    return _nvrtc_bindings


def _nvrtc_version(nvrtc: Any) -> version:
    """Read the bound NVRTC library's version or report its query failure."""

    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    return version(int(major), int(minor))


def _check_nvrtc(err, prog, *, nvrtc=None):
    """Raise an NVRTC failure with the program's compilation log.

    ``compile_impl`` checks compilation, name-query, and image-retrieval
    results here so failures include NVRTC's diagnostic text. Keep the
    original result code even if fetching the log also fails; a secondary
    logging error must not hide the operation that failed.

    Parameters
    ----------
    err : nvrtcResult
        Result code returned by the operation being checked. Success returns
        immediately without fetching a log.
    prog : nvrtcProgram
        Live program whose log can explain the failure. The caller owns its
        lifetime and destroys it after diagnostics have been collected.
    nvrtc : module or None
        Bindings for the already selected NVRTC library. If omitted, import
        the bindings; toolkit selection and preloading must have happened first.

    Raises
    ------
    RuntimeError
        NVRTC reported failure. The message includes its result code and either
        the program log or the reason that the log could not be read.
    """

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
    """Include the target and output options in the saved source identity."""

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
) -> CompilerIdentity:
    """Identify target and options that qualify provider symbols.

    The options include the context's ordered header paths. Providers also use
    this identity to group compatible specializations into one compilation. The
    complete identity also needs the context's library and header-content
    identities, which callers retain separately.
    """

    return CompilerIdentity(
        cc=int(cc),
        rdc=bool(rdc),
        code=str(code),
        compiler_options=_compiler_options(
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
    layout_queries: tuple[str, ...] = (),
    layout_symbol: str = "",
) -> bytes | str | _LayoutResult:
    """Compile one source unit using a complete, explicit cache identity.

    The memory and disk cache decorators key all arguments, including toolkit
    paths and header identity that are not otherwise read by the function
    body. Those fields prevent artifacts from different compiler installations
    or header sets from sharing an entry. Callers must resolve and preload the
    matching compiler context before entering this function.

    On a cache miss, verify that the supplied option tuple matches the request
    and that the loaded NVRTC version still matches the context. Compile the
    source, retrieve the requested image, and destroy the NVRTC program on
    both success and failure. A cleanup error does not replace an earlier
    compilation error. Cache hits bypass these body-level checks.

    Register distinct layout expressions before compilation and decode their
    names while the program is still alive. Restore duplicate entries in the
    caller's order when returning layouts with an LTO image. The image and
    layouts form one cache value, tying metadata to the same compilation.

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
        Exact ordered options from ``_compiler_options`` for this request.
    layout_queries : tuple of str, optional
        Ordered NVRTC name expressions whose lowered names encode storage
        layouts. Duplicate expressions are evaluated once and retain their
        positions in the returned layouts.
    layout_symbol : str, optional
        Expected symbol used to validate and decode lowered layout names.

    Returns
    -------
    bytes, str, or tuple
        LTO image bytes for ``"lto"`` or ASCII-decoded source for ``"ptx"``.
        With LTO layout queries, return ``(image, layouts)`` in query order.

    Raises
    ------
    RuntimeError
        Options or loaded compiler version disagree with the request, or NVRTC
        compilation, image retrieval, or program cleanup fails.
    ValueError
        The output format or relocatable-code option is invalid, or a lowered
        layout name does not match the expected encoding.
    """

    # These arguments are only used by the cache decorators to distinguish
    # compiler and header contexts. Delete their local bindings to mark them
    # as intentionally unused by the compilation body.
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
            "NVRTC compiler-option identity does not match the "
            "requested compile."
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
        for expression in dict.fromkeys(layout_queries):
            (err,) = nvrtc.nvrtcAddNameExpression(
                prog, expression.encode("utf-8")
            )
            _check_nvrtc(err, prog, nvrtc=nvrtc)
        (err,) = nvrtc.nvrtcCompileProgram(
            prog, len(compiler_options), list(compiler_options)
        )
        _check_nvrtc(err, prog, nvrtc=nvrtc)
        layouts = {}
        for expression in dict.fromkeys(layout_queries):
            err, lowered_name = nvrtc.nvrtcGetLoweredName(
                prog, expression.encode("utf-8")
            )
            _check_nvrtc(err, prog, nvrtc=nvrtc)
            layouts[expression] = decode_layout_name(
                lowered_name, symbol=layout_symbol, expression=expression
            )
        if code == "lto":
            err, size = nvrtc.nvrtcGetLTOIRSize(prog)
            _check_nvrtc(err, prog, nvrtc=nvrtc)
            image = bytearray(size)
            (err,) = nvrtc.nvrtcGetLTOIR(prog, image)
            _check_nvrtc(err, prog, nvrtc=nvrtc)
            result = bytes(image)
            if layout_queries:
                return result, tuple(
                    layouts[expression] for expression in layout_queries
                )
            return result
        err, size = nvrtc.nvrtcGetPTXSize(prog)
        _check_nvrtc(err, prog, nvrtc=nvrtc)
        image = bytearray(size)
        (err,) = nvrtc.nvrtcGetPTX(prog, image)
        _check_nvrtc(err, prog, nvrtc=nvrtc)
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
    ordering keeps wrapper compilation and later linking on the same toolkit.

    Hash the resolved header roots and their contents into the returned
    context. The context is used both for artifact cache keys and provider
    symbol qualification. Resolution is lazy at its callers; this function
    itself is not memoized and may load process-wide compiler libraries.
    Algorithms retain their resolved context, so changing
    ``CUDA_COOP_CCCL_ROOT`` affects subsequent resolutions, not contexts
    already held by providers or supplied explicitly to ``compile``.

    Returns
    -------
    CompileContext
        Frozen snapshot of exact library paths/versions, ordered include
        roots, and header identity for a provider compilation.
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
) -> tuple[version, bytes | str | _LayoutResult]:
    """Compile provider source with a resolved header and toolkit context.

    Build the ordered options from the selected context, optionally dump the
    source through the shared source-dump hook, and pass every context field
    to ``compile_impl``. Dumping occurs before cache lookup so source
    inspection also works when compilation is reused from memory or disk.

    Parameters
    ----------
    context : CompileContext, optional
        Previously resolved and preloaded compiler context. ``None`` resolves
        one now. Supplying a context reuses its identity; it does not reload
        the library paths stored in it.
    **kwargs : dict
        Required ``cpp`` source string, integer ``cc`` target, boolean
        ``rdc``, and ``code`` equal to ``"lto"`` or ``"ptx"``. These are
        forwarded to ``compile_impl`` with the context and generated options.
        Internal callers can also supply ``layout_queries`` and
        ``layout_symbol``; use ``compile_with_layouts`` to prepare these from
        C++ type names.

    Returns
    -------
    tuple
        ``(nvrtc_version, result)`` with LTO bytes or PTX text according to
        ``code``. With LTO layout queries, ``result`` is ``(image, layouts)``
        in query order. The version is taken from the selected context.

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


def compile_with_layouts(
    *, layout_types: tuple[str, ...], code: str = "lto", **kwargs: Any
) -> tuple[version, _LayoutResult]:
    """Cache provider LTO IR and its ordered storage layouts together.

    The same NVRTC program evaluates ``sizeof`` and ``alignof`` and emits
    the provider image. Query expressions are part of the cache identity, and
    their ordered layouts are stored with the image. A later cache hit can
    therefore recover both without compiling or linking a metadata program.

    Parameters
    ----------
    layout_types : tuple of str
        C++ type names or type expressions visible at the end of the supplied
        source. The result has one layout per entry in the same order, with
        duplicates preserved. An empty tuple uses ordinary compilation
        without adding a probe to the source.
    code : {"lto"}, optional
        Output format. Only LTO supports the combined image/layout result.
    **kwargs : dict
        Arguments for ``compile``: ``cpp`` source, integer ``cc`` target,
        boolean ``rdc``, and optional resolved ``context``. The source must
        make every requested type visible to an appended layout probe.

    Returns
    -------
    tuple
        ``(nvrtc_version, (image, layouts))``. The image is LTO bytes; each
        layout is a ``(size, alignment)`` pair in bytes. For an empty type
        tuple, ``layouts`` is also empty.

    Raises
    ------
    ValueError
        The output format is not LTO, a type expression is empty, or NVRTC
        returns an unexpected layout name or invalid size/alignment pair.
    RuntimeError
        Compiler identity checks or NVRTC compilation fail.
    """

    if code != "lto":
        raise ValueError("storage layout queries require LTO compilation")
    source, symbol, queries = prepare_layout_queries(
        kwargs["cpp"], layout_types
    )
    if not queries:
        compiler_version, image = compile(code=code, **kwargs)
        return compiler_version, (cast(bytes, image), ())
    kwargs["cpp"] = source
    return cast(
        tuple[version, _LayoutResult],
        compile(
            code=code, layout_queries=queries, layout_symbol=symbol, **kwargs
        ),
    )
