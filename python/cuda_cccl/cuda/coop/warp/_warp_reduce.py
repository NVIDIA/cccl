# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import make_binary_tempfile
from .._types import (
    Algorithm,
    Dependency,
    DependentPythonOperator,
    DependentReference,
    Invocable,
    Pointer,
    TemplateParameter,
    numba_type_to_wrapper,
)


def reduce(dtype, binary_op, threads_in_warp=32, methods=None):
    """Computes a warp-wide reduction for lane\ :sub:`0` using the specified binary reduction functor.
    Each thread contributes one input element.

    Warning:
        The return value is undefined in threads other than thread\ :sub:`0`.

    Example:
        The code snippet below illustrates a max reduction of 32 integer items that
        are partitioned across a warp of threads.

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``reduce`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce
            :end-before: example-end reduce

        Suppose the set of inputs across the warp of threads is
        ``{ 0, 1, 2, 3, ..., 31 }``.
        The corresponding output in the threads lane\ :sub:`0` will be ``{ 31 }``.

    Args:
        dtype: Data type being reduced
        threads_in_warp: The number of threads in a warp
        binary_op: Binary reduction function

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm(
        "WarpReduce",
        "Reduce",
        "warp_reduce",
        ["cub/warp/warp_reduce.cuh"],
        [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
        [
            [
                Pointer(numba.uint8),
                DependentReference(Dependency("T")),
                DependentPythonOperator(
                    Dependency("T"),
                    [Dependency("T"), Dependency("T")],
                    Dependency("Op"),
                ),
                DependentReference(Dependency("T"), True),
            ]
        ],
        type_definitions=[numba_type_to_wrapper(dtype, methods=methods)],
    )
    specialization = template.specialize(
        {"T": dtype, "VIRTUAL_WARP_THREADS": threads_in_warp, "Op": binary_op}
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


def sum(dtype, threads_in_warp=32):
    """Computes a warp-wide reduction for lane\ :sub:`0` using addition (+) as the reduction operator.
    Each thread contributes one input element.

    Warning:
        The return value is undefined in threads other than thread\ :sub:`0`.

    Example:
        The code snippet below illustrates a reduction of 32 integer items that
        are partitioned across a warp of threads.

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``reduce`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum
            :end-before: example-end sum

        Suppose the set of inputs across the warp of threads is
        ``{ 1, 1, 1, 1, ..., 1 }``.
        The corresponding output in the threads lane\ :sub:`0` will be ``{ 32 }``.

    Args:
        dtype: Data type being reduced
        threads_in_warp: The number of threads in a warp

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm(
        "WarpReduce",
        "Sum",
        "warp_reduce",
        ["cub/warp/warp_reduce.cuh"],
        [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
        [
            [
                Pointer(numba.uint8),
                DependentReference(Dependency("T")),
                DependentReference(Dependency("T"), True),
            ]
        ],
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
