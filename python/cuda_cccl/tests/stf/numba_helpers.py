# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba helpers for cuda.stf tests. Not shipped in the wheel.
Convert task.get_arg_cai() / task.args_cai() (stf_cai) to Numba device arrays.
Requires numba-cuda. Import from tests.stf when running from source.
"""

from __future__ import annotations


def get_arg_numba(task, index):
    """Return one task argument as a Numba device array. task.get_arg_cai(index) returns stf_cai."""
    from numba import cuda

    return cuda.from_cuda_array_interface(
        task.get_arg_cai(index), owner=None, sync=False
    )


def numba_arguments(task):
    """
    Return all task buffer arguments as Numba device arrays. Same shape as task.args_cai():
    None, a single array, or a tuple of arrays.
    """
    from numba import cuda

    out = task.args_cai()
    if out is None:
        return None
    if isinstance(out, tuple):
        return tuple(
            cuda.from_cuda_array_interface(o, owner=None, sync=False) for o in out
        )
    return cuda.from_cuda_array_interface(out, owner=None, sync=False)


__all__ = ["get_arg_numba", "numba_arguments"]
