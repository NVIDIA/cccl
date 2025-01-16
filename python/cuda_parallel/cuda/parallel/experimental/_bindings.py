# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from functools import lru_cache
from typing import List

from cuda.cccl import get_include_paths  # type: ignore[import-not-found]

from . import _cccl as cccl


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
    paths = [
        f"-I{path}".encode()
        for path in get_include_paths().as_tuple()
        if path is not None
    ]
    return paths
