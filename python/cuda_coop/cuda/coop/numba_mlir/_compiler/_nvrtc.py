# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile and cache Numba-backend CUDA wrappers with NVRTC.

The module resolves toolkit/header identity up front so generated symbols and
disk-cache entries remain tied to the compiler inputs that produced them.
"""

import functools
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cuda.bindings import nvrtc
from cuda.coop._headers import resolve_include_paths
from cuda.coop._headers._identity import include_dirs_identity
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

from ._artifacts import check_in, version
from ._caching import disk_cache

_NVRTC_DUMP_DIR_ENV = "CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR"


@dataclass(frozen=True)
class CompileContext:
    """Resolved NVRTC and header identity for one compilation."""

    nvrtc_path: str | None
    nvrtc_version: version
    include_dirs: tuple[str, ...]
    header_identity: str

    @property
    def symbol_suffix(self) -> str:
        digest = hashlib.sha256()
        digest.update((self.nvrtc_path or "").encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(f"{self.nvrtc_version.major}.{self.nvrtc_version.minor}".encode())
        digest.update(b"\0")
        for include_dir in self.include_dirs:
            digest.update(include_dir.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
        digest.update(self.header_identity.encode("utf-8"))
        return digest.hexdigest()[:16]


def CHECK_NVRTC(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
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
        raise RuntimeError(
            f"NVRTC error: {original_err}: {bytes(log).decode('ascii', errors='replace')}"
        )


def _dump_source(cpp, cc, code):
    """Dump one content-addressed pre-NVRTC source file when requested."""

    dump_dir = os.environ.get(_NVRTC_DUMP_DIR_ENV)
    if not dump_dir:
        return None

    root = Path(dump_dir).expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(str(cc).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(code).encode("ascii"))
    digest.update(b"\0")
    digest.update(cpp.encode("utf-8"))
    source_hash = digest.hexdigest()[:16]
    suffix = "lto" if code == "lto" else "ptx"
    destination = root / (f"cuda_coop_numba_mlir_{source_hash}_cc{cc}_{suffix}.cu")
    if destination.is_file():
        return destination

    fd, temporary = tempfile.mkstemp(
        dir=root,
        prefix=f".{destination.name}.",
        text=True,
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


# cpp is the C++ source code
# cc = 800 for Ampere, 900 Hopper, etc
# rdc is true or false
# code is lto or ptx
# @cache
@functools.lru_cache(maxsize=8)  # Always enabled
@disk_cache  # Optional, see caching.py
def compile_impl(
    cpp,
    cc,
    rdc,
    code,
    nvrtc_path,
    nvrtc_version,
    include_dirs,
    header_identity,
):
    del header_identity
    check_in("rdc", rdc, [True, False])
    check_in("code", code, ["lto", "ptx"])

    opts = [b"--std=c++17"]
    for path in include_dirs:
        opts += [f"--include-path={path}".encode("ascii")]
    opts += [f"--gpu-architecture=compute_{cc}".encode("ascii")]
    if rdc:
        opts += [b"--relocatable-device-code=true"]

    if code == "lto":
        opts += [b"-dlto"]

    # Some strange linking issues
    opts += [b"-DCCCL_DISABLE_BF16_SUPPORT"]

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cpp), b"code.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram error: {err}")

    had_error = False
    try:
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        CHECK_NVRTC(err, prog)

        if code == "lto":
            err, ltoSize = nvrtc.nvrtcGetLTOIRSize(prog)
            CHECK_NVRTC(err, prog)

            lto = bytearray(ltoSize)
            (err,) = nvrtc.nvrtcGetLTOIR(prog, lto)
            CHECK_NVRTC(err, prog)

            return bytes(lto)

        elif code == "ptx":
            err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
            CHECK_NVRTC(err, prog)

            ptx = bytearray(ptxSize)
            (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
            CHECK_NVRTC(err, prog)

            return bytes(ptx).decode("ascii")
    except Exception:
        had_error = True
        raise
    finally:
        (destroy_err,) = nvrtc.nvrtcDestroyProgram(prog)
        if destroy_err != nvrtc.nvrtcResult.NVRTC_SUCCESS and not had_error:
            raise RuntimeError(f"nvrtcDestroyProgram error: {destroy_err}")


def resolve_compile_context() -> CompileContext:
    """Resolve the exact inputs that participate in NVRTC cache identity."""

    include_paths = resolve_include_paths(
        start=Path(__file__),
        configured_roots=(os.environ.get("CUDA_COOP_CCCL_ROOT"),),
    )
    include_dirs = tuple(str(path) for path in include_paths.as_tuple())
    libraries = preload_toolkit_compiler_libraries(include_paths.cuda)
    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    validate_nvrtc_version(libraries, (major, minor))
    nvrtc_version = version(major, minor)
    header_identity = include_dirs_identity(include_dirs).digest
    return CompileContext(
        nvrtc_path=libraries.nvrtc_path,
        nvrtc_version=nvrtc_version,
        include_dirs=include_dirs,
        header_identity=header_identity,
    )


def compile(*, context: CompileContext | None = None, **kwargs):
    _dump_source(kwargs["cpp"], kwargs["cc"], kwargs["code"])
    context = resolve_compile_context() if context is None else context
    return context.nvrtc_version, compile_impl(
        **kwargs,
        nvrtc_path=context.nvrtc_path,
        nvrtc_version=context.nvrtc_version,
        include_dirs=context.include_dirs,
        header_identity=context.header_identity,
    )
