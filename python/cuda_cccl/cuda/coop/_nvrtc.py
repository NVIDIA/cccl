# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

from cuda.bindings import nvrtc

from ._caching import disk_cache
from ._common import check_in, version


def CHECK_NVRTC(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * logsize
        err = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC error: {log.decode('ascii')}")


# cpp is the C++ source code
# cc = 800 for Ampere, 900 Hopper, etc
# rdc is true or false
# code is lto or ptx
# @cache
@functools.lru_cache(maxsize=32)  # Always enabled
@disk_cache  # Optional, see caching.py
def compile_impl(cpp, cc, rdc, code, nvrtc_path, nvrtc_version):
    check_in("rdc", rdc, [True, False])
    check_in("code", code, ["lto", "ptx"])

    opts = [b"--std=c++17"]

    # TODO: move this to a module-level import (after docs env modernization).
    from cuda.cccl import get_include_paths

    for path in get_include_paths().as_tuple():
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
