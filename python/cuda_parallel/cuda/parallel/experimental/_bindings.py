# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, List

from numba import cuda

from cuda.cccl import get_include_paths  # type: ignore[import-not-found]


@lru_cache()
def get_bindings() -> ctypes.CDLL:
    # TODO: once docs env supports Python >= 3.9, we
    # can move this to a module-level import.
    from importlib.resources import as_file, files

    so_path = "cccl/libcccl.c.parallel.so"
    with as_file(files("cuda.parallel.experimental")) as f:
        cccl_c_path = f / so_path
    if not cccl_c_path.exists():
        # This may be needed to support editable builds (see PR #3762).
        for sp in sys.path:
            cccl_c_path = Path(sp).resolve() / "cuda/parallel/experimental" / so_path
            if cccl_c_path.exists():
                break
        else:
            raise RuntimeError(f"Unable to locate {so_path}")
    _bindings = ctypes.CDLL(str(cccl_c_path))
    return _bindings


@lru_cache()
def get_paths() -> List[bytes]:
    paths = [
        f"-I{path}".encode()
        for path in get_include_paths().as_tuple()
        if path is not None
    ]
    return paths


def call_build(build_impl_fn: Callable, *args, **kwargs):
    """Calls given build_impl_fn callable while providing compute capability and paths

    Returns result of the call.
    """
    cc_major, cc_minor = cuda.get_current_device().compute_capability
    cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
    error = build_impl_fn(
        *args,
        cc_major,
        cc_minor,
        ctypes.c_char_p(cub_path),
        ctypes.c_char_p(thrust_path),
        ctypes.c_char_p(libcudacxx_path),
        ctypes.c_char_p(cuda_include_path),
        **kwargs,
    )
    return error
