# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def cai_to_numba(cai: dict):
    from numba import cuda

    return cuda.from_cuda_array_interface(cai, owner=None, sync=False)
