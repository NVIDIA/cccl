# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import ctypes
from functools import lru_cache
from typing import List, Optional

from . import _cccl as cccl


def _get_cuda_path() -> Optional[str]:
    cuda_path = os.environ.get("CUDA_PATH", "")
    if os.path.exists(cuda_path):
        return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    default_path = "/usr/local/cuda"
    if os.path.exists(default_path):
        return default_path

    return None


@lru_cache()
def get_bindings() -> ctypes.CDLL:
    # TODO: once docs env supports Python >= 3.9, we
    # can move this to a module-level import.
    from importlib.resources import as_file, files

    with as_file(files("cuda.parallel.experimental")) as f:
        cccl_c_path = str(f / "cccl" / "libcccl.c.parallel.so")
    _bindings = ctypes.CDLL(cccl_c_path)
    _bindings.cccl_device_reduce.restype = ctypes.c_int
    _bindings.cccl_device_reduce.argtypes = [
        cccl.DeviceReduceBuildResult,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulonglong),
        cccl.Iterator,
        cccl.Iterator,
        ctypes.c_ulonglong,
        cccl.Op,
        cccl.Value,
        ctypes.c_void_p,
    ]
    _bindings.cccl_device_reduce_cleanup.restype = ctypes.c_int
    return _bindings


@lru_cache()
def get_paths() -> List[bytes]:
    # TODO: once docs env supports Python >= 3.9, we
    # can move this to a module-level import.
    from importlib.resources import as_file, files

    with as_file(files("cuda.parallel")) as f:
        # Using `.parent` for compatibility with pip install --editable:
        cub_include_path = str(f.parent / "_include")
    thrust_include_path = cub_include_path
    libcudacxx_include_path = str(os.path.join(cub_include_path, "libcudacxx"))
    cuda_include_path = None
    cuda_path = _get_cuda_path()
    if cuda_path is not None:
        cuda_include_path = str(os.path.join(cuda_path, "include"))
    paths = [
        f"-I{path}".encode()
        for path in (
            cub_include_path,
            thrust_include_path,
            libcudacxx_include_path,
            cuda_include_path,
        )
        if path is not None
    ]
    return paths
