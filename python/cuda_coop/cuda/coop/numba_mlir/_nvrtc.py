# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import hashlib
import os
import tempfile
from pathlib import Path

from cuda.bindings import nvrtc
from cuda.coop._headers import resolve_include_paths
from cuda.coop._headers._identity import include_dirs_identity

from ._caching import disk_cache
from ._common import check_in, version

_NVRTC_DUMP_DIR_ENV = "CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR"


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
        try:
            os.replace(temporary, destination)
        except FileExistsError:
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
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


def compile(**kwargs):
    _dump_source(kwargs["cpp"], kwargs["cc"], kwargs["code"])
    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    nvrtc_version = version(major, minor)

    include_paths = resolve_include_paths(
        start=Path(__file__),
        configured_roots=(os.environ.get("CUDA_COOP_CCCL_ROOT"),),
    )
    include_dirs = tuple(str(path) for path in include_paths.as_tuple())
    header_identity = include_dirs_identity(include_dirs).digest
    return nvrtc_version, compile_impl(
        **kwargs,
        nvrtc_path=nvrtc.__file__,
        nvrtc_version=nvrtc_version,
        include_dirs=include_dirs,
        header_identity=header_identity,
    )
