# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
from cuda import nvrtc
from cuda.cooperative.experimental._caching import disk_cache
from cuda.cooperative.experimental._common import check_in, version
import importlib.resources as pkg_resources
import functools

def CHECK_NVRTC(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * logsize
        err = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC error: {log.decode('ascii')}")


def get_cuda_path():
    cuda_path = os.environ.get('CUDA_PATH', '')
    if os.path.exists(cuda_path):
        return cuda_path

    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    default_path = '/usr/local/cuda'
    if os.path.exists(default_path):
        return default_path

    return None


# cpp is the C++ source code
# cc = 800 for Ampere, 900 Hopper, etc
# rdc is true or false
# code is lto or ptx
# @cache
@functools.lru_cache(maxsize=32) # Always enabled
@disk_cache # Optional, see caching.py
def compile_impl(cpp, cc, rdc, code, nvrtc_path, nvrtc_version):
    check_in('rdc', rdc, [True, False])
    check_in('code', code, ['lto', 'ptx'])

    with pkg_resources.path('cuda', '_include') as include_path:
        cub_path = include_path
        thrust_path = include_path
        libcudacxx_path = os.path.join(include_path, 'libcudacxx')
        cuda_include_path = os.path.join(get_cuda_path(), 'include')

    opts = [b"--std=c++17", \
            bytes(f"--include-path={cub_path}", encoding='ascii'), \
            bytes(f"--include-path={thrust_path}", encoding='ascii'), \
            bytes(f"--include-path={libcudacxx_path}", encoding='ascii'), \
            bytes(f"--include-path={cuda_include_path}", encoding='ascii'), \
            bytes(f"--gpu-architecture=compute_{cc}", encoding='ascii')]
    if rdc:
        opts += [b"--relocatable-device-code=true"]

    if code == 'lto':
        opts += [b"-dlto"]

    # Some strange linking issues
    opts += [b"-DCCCL_DISABLE_BF16_SUPPORT"]

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cpp), b"code.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram error: {err}")

    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    CHECK_NVRTC(err, prog)

    if code == 'lto':
        err, ltoSize = nvrtc.nvrtcGetLTOIRSize(prog)
        CHECK_NVRTC(err, prog)

        lto = b" " * ltoSize
        err, = nvrtc.nvrtcGetLTOIR(prog, lto)
        CHECK_NVRTC(err, prog)

        err, = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return lto

    elif code == 'ptx':
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        CHECK_NVRTC(err, prog)

        ptx = b" " * ptxSize
        err, = nvrtc.nvrtcGetPTX(prog, ptx)
        CHECK_NVRTC(err, prog)

        err, = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return ptx.decode('ascii')

def compile(**kwargs):

    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    nvrtc_version = version(major, minor)
    return nvrtc_version, compile_impl(**kwargs, \
                        nvrtc_path=nvrtc.__file__, \
                        nvrtc_version=nvrtc_version)
