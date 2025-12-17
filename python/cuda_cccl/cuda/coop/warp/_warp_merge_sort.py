# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import make_binary_tempfile
from .._types import (
    Algorithm,
    Constant,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    Invocable,
    Pointer,
    TemplateParameter,
    numba_type_to_wrapper,
)


def merge_sort_keys(
    dtype, items_per_thread, compare_op, threads_in_warp=32, methods=None
):
    """Performs a warp-wide merge sort over a :ref:`blocked arrangement <flexible-data-arrangement>` of keys.

    Example:
        The code snippet below illustrates a sort of 128 integer keys that
        are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across a warp of 32 threads
        where each thread owns 4 consecutive keys. We start by importing necessary modules:

        Below is the code snippet that demonstrates the usage of the ``merge_sort_keys`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_merge_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin merge-sort
            :end-before: example-end merge-sort

        Suppose the set of input ``thread_keys`` across the warp of threads is
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [124, 125, 126, 127] }``.
        The corresponding output ``thread_keys`` in those threads will be
        ``{ [127, 126, 125, 124], [123, 122, 121, 120], ..., [3, 2, 1, 0] }``.

    Args:
        dtype: Numba data type of the keys to be sorted
        threads_in_warp: The number of threads in a warp
        items_per_thread: The number of items each thread owns
        compare_op: Comparison function object which returns true if the first argument is ordered before the second one

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm(
        "WarpMergeSort",
        "Sort",
        "warp_merge_sort",
        ["cub/warp/warp_merge_sort.cuh"],
        [
            TemplateParameter("KeyT"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("VIRTUAL_WARP_THREADS"),
        ],
        [
            [
                Pointer(numba.uint8),
                DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
                DependentPythonOperator(
                    Constant(numba.int8),
                    [Dependency("KeyT"), Dependency("KeyT")],
                    Dependency("Op"),
                ),
            ]
        ],
        type_definitions=[numba_type_to_wrapper(dtype, methods=methods)],
    )
    specialization = template.specialize(
        {
            "KeyT": dtype,
            "VIRTUAL_WARP_THREADS": threads_in_warp,
            "ITEMS_PER_THREAD": items_per_thread,
            "Op": compare_op,
        }
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
