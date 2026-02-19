# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import os

from cuda.bindings import nvrtc

from ._caching import disk_cache
from ._common import check_in, version


def CHECK_NVRTC(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * logsize
        err = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC error: {log.decode('ascii')}")


_NVRTC_COMPILE_COUNTER = 0
_NVRTC_COMPILE_COUNTER_ENABLED = None
_NVRTC_DUMP_COUNTER = 0


def _is_compile_counter_enabled():
    global _NVRTC_COMPILE_COUNTER_ENABLED
    if _NVRTC_COMPILE_COUNTER_ENABLED is None:
        val = os.environ.get("NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT")
        _NVRTC_COMPILE_COUNTER_ENABLED = val is not None and val.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return _NVRTC_COMPILE_COUNTER_ENABLED


def _get_dump_dir():
    dump_dir = os.environ.get("NUMBA_CCCL_COOP_NVRTC_DUMP_DIR")
    if dump_dir:
        return dump_dir
    dump_enabled = os.environ.get("NUMBA_CCCL_COOP_NVRTC_DUMP")
    if dump_enabled and dump_enabled.lower() in ("1", "true", "yes", "on"):
        return "/tmp/cccl_nvrtc"
    return None


def _dump_source(cpp, cc, code):
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    os.makedirs(dump_dir, exist_ok=True)
    global _NVRTC_DUMP_COUNTER
    _NVRTC_DUMP_COUNTER += 1
    suffix = "lto" if code == "lto" else "ptx"
    filename = f"nvrtc_{_NVRTC_DUMP_COUNTER:04d}_cc{cc}_{suffix}.cu"
    path = os.path.join(dump_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(cpp)


def _set_compile_counter_enabled(enabled):
    global _NVRTC_COMPILE_COUNTER_ENABLED
    _NVRTC_COMPILE_COUNTER_ENABLED = enabled


def reset_compile_counter():
    global _NVRTC_COMPILE_COUNTER
    _NVRTC_COMPILE_COUNTER = 0


def get_compile_counter():
    return _NVRTC_COMPILE_COUNTER


# cpp is the C++ source code
# cc = 800 for Ampere, 900 Hopper, etc
# rdc is true or false
# code is lto or ptx
# @cache
@functools.lru_cache(maxsize=32)  # Always enabled
@disk_cache  # Optional, see caching.py
def compile_impl(cpp, cc, rdc, code, nvrtc_path, nvrtc_version):
    _dump_source(cpp, cc, code)
    if _is_compile_counter_enabled():
        global _NVRTC_COMPILE_COUNTER
        _NVRTC_COMPILE_COUNTER += 1
    check_in("rdc", rdc, [True, False])
    check_in("code", code, ["lto", "ptx"])

    opts = [b"--std=c++17"]

    # TODO: move this to a module-level import (after docs env modernization).
    from cuda.cccl import get_include_paths

    include_paths = get_include_paths()
    # print(f"NVRTC include paths: {include_paths}")
    # include_paths.cub = '/home/trentn/src/cccl/cub'
    # include_paths.thrust = '/home/trentn/src/cccl/thrust'
    # include_paths.libcudacxx = '/home/trentn/src/cccl/libcudacxx'
    for path in include_paths.as_tuple():
        if path is not None:
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

    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    CHECK_NVRTC(err, prog)

    if code == "lto":
        err, ltoSize = nvrtc.nvrtcGetLTOIRSize(prog)
        CHECK_NVRTC(err, prog)

        lto = b" " * ltoSize
        (err,) = nvrtc.nvrtcGetLTOIR(prog, lto)
        CHECK_NVRTC(err, prog)

        (err,) = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return lto

    elif code == "ptx":
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        CHECK_NVRTC(err, prog)

        ptx = b" " * ptxSize
        (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
        CHECK_NVRTC(err, prog)

        (err,) = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return ptx.decode("ascii")


def compile(**kwargs):
    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    nvrtc_version = version(major, minor)
    return nvrtc_version, compile_impl(
        **kwargs, nvrtc_path=nvrtc.__file__, nvrtc_version=nvrtc_version
    )
