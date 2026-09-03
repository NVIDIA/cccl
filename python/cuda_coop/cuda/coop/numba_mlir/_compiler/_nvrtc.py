# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile and cache Numba-CUDA-MLIR wrappers with an exact CUDA toolchain."""

from __future__ import annotations

import functools
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda.coop._headers import resolve_include_paths
from cuda.coop._headers._identity import include_dirs_identity
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

from ._artifacts import check_in, version
from ._caching import disk_cache

_NVRTC_DUMP_DIR_ENV = "CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR"
_REQUIRED_HEADERS = (
    "cub/block/block_load.cuh",
    "cub/block/block_store.cuh",
    "cuda/barrier",
    "cuda/devices",
    "cuda/experimental/coop.cuh",
    "cuda/experimental/group.cuh",
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

    from cuda.bindings import nvrtc

    return nvrtc


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
        raise RuntimeError(f"NVRTC error: {original_err}; failed to get log: {log_err}")
    rendered = bytes(log).rstrip(b"\0").decode("ascii", errors="replace")
    raise RuntimeError(f"NVRTC error: {original_err}: {rendered}")


def _dump_source(cpp, cc, code, compiler_options):
    """Dump one content-addressed pre-NVRTC source file when requested."""

    dump_dir = os.environ.get(_NVRTC_DUMP_DIR_ENV)
    if not dump_dir:
        return None
    root = Path(dump_dir).expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for value in (str(cc), str(code), repr(compiler_options), cpp):
        digest.update(value.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    suffix = "lto" if code == "lto" else "ptx"
    destination = root / (
        f"cuda_coop_numba_mlir_{digest.hexdigest()[:16]}_cc{cc}_{suffix}.cu"
    )
    if destination.is_file():
        return destination
    fd, temporary = tempfile.mkstemp(
        dir=root, prefix=f".{destination.name}.", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as source_file:
            source_file.write(cpp)
            source_file.flush()
            os.fsync(source_file.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


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
) -> tuple[object, ...]:
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
    cpp,
    cc,
    rdc,
    code,
    toolkit_root,
    toolkit_version,
    nvrtc_path,
    nvrtc_builtins_path,
    nvjitlink_path,
    nvrtc_version,
    nvjitlink_version,
    include_dirs,
    header_identity,
    compiler_options,
):
    """Compile one cache-key-complete source unit."""

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
            "NVRTC compiler-option identity does not match the requested compile."
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
    """Resolve headers and exact same-root compiler libraries lazily."""

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


def compile(*, context: CompileContext | None = None, **kwargs):
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
