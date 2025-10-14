# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba

from .._common import make_binary_tempfile
from .._types import (
    Algorithm,
    Dependency,
    DependentReference,
    Invocable,
    Pointer,
    TemplateParameter,
)


def exclusive_sum(dtype, threads_in_warp=32):
    """Computes an exclusive warp-wide prefix sum using addition (+) as the scan operator.
    The value of 0 is applied as the initial value, and is assigned to the output in *lane*\ :sub:`0`.

    Example:
        The code snippet below illustrates an exclusive prefix sum of 32 integer items:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``exclusive_sum`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum

        Suppose the set of input ``thread_data`` across the warp of threads is
        ``{ [1, 1, 1, 1], [1, 1, 1, 1], ..., [1, 1, 1, 1] }``.
        The corresponding output ``thread_data`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [28, 29, 30, 31] }``.

    Args:
        dtype: Data type being scanned
        threads_in_warp: The number of threads in a warp

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm(
        "WarpScan",
        "ExclusiveSum",
        "warp_scan",
        ["cub/warp/warp_scan.cuh"],
        [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
        [
            [
                Pointer(numba.uint8),
                DependentReference(Dependency("T")),
                DependentReference(Dependency("T"), True),
            ]
        ],
        fake_return=True,
    )
    specialization = template.specialize(
        {"T": dtype, "VIRTUAL_WARP_THREADS": threads_in_warp}
    )
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir(threads=threads_in_warp)
        ],
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )
