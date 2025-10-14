# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Callable, Literal, Tuple, Union

import numba

from .._common import (
    CUB_BLOCK_REDUCE_ALGOS,
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    DependentReference,
    Invocable,
    Pointer,
    TemplateParameter,
    Value,
    numba_type_to_wrapper,
)

if TYPE_CHECKING:
    import numpy as np


def _reduce(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int]],
    items_per_thread: int,
    binary_op: Callable,
    algorithm: Literal["raking", "raking_commutative_only", "warp_reductions"],
    methods: dict = None,
) -> Callable:
    if algorithm not in CUB_BLOCK_REDUCE_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    specialization_kwds = {
        "T": dtype,
        "BLOCK_DIM_X": dim[0],
        "ALGORITHM": CUB_BLOCK_REDUCE_ALGOS[algorithm],
        "BLOCK_DIM_Y": dim[1],
        "BLOCK_DIM_Z": dim[2],
    }

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    if binary_op is None:
        cpp_method_name = "Sum"

        if items_per_thread == 1:
            parameters = [
                # Signatures:
                # T Sum(T);
                [
                    Pointer(numba.uint8),
                    DependentReference(Dependency("T")),
                    DependentReference(Dependency("T"), is_output=True),
                ],
                # T Sum(T, int num_valid);
                [
                    Pointer(numba.uint8),
                    DependentReference(Dependency("T")),
                    Value(numba.int32),
                    DependentReference(Dependency("T"), is_output=True),
                ],
            ]

        else:
            assert items_per_thread > 1, items_per_thread
            specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread

            parameters = [
                # Signatures:
                # T Sum(T(&)[ITEMS_PER_THREAD]);
                [
                    Pointer(numba.uint8),
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    DependentReference(Dependency("T"), is_output=True),
                ],
            ]

    else:
        cpp_method_name = "Reduce"
        specialization_kwds["Op"] = binary_op

        if items_per_thread == 1:
            parameters = [
                # Signatures:
                # T Reduce(T, Op);
                [
                    Pointer(numba.uint8),
                    DependentReference(Dependency("T")),
                    DependentPythonOperator(
                        Dependency("T"),
                        [Dependency("T"), Dependency("T")],
                        Dependency("Op"),
                    ),
                    DependentReference(Dependency("T"), is_output=True),
                ],
                # T Reduce(T, Op, int num_valid);
                [
                    Pointer(numba.uint8),
                    DependentReference(Dependency("T")),
                    DependentPythonOperator(
                        Dependency("T"),
                        [Dependency("T"), Dependency("T")],
                        Dependency("Op"),
                    ),
                    Value(numba.int32),
                    DependentReference(Dependency("T"), is_output=True),
                ],
            ]

        else:
            assert items_per_thread > 1, items_per_thread
            specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread

            parameters = [
                # Signatures:
                # T Reduce(T(&)[ITEMS_PER_THREAD], Op);
                [
                    Pointer(numba.uint8),
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    DependentPythonOperator(
                        Dependency("T"),
                        [Dependency("T"), Dependency("T")],
                        Dependency("Op"),
                    ),
                    DependentReference(Dependency("T"), is_output=True),
                ],
            ]

    template = Algorithm(
        "BlockReduce",
        cpp_method_name,
        "block_reduce",
        ["cub/block/block_reduce.cuh"],
        template_parameters,
        parameters,
        type_definitions=[numba_type_to_wrapper(dtype, methods=methods)],
    )

    specialization = template.specialize(specialization_kwds)
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )


def reduce(
    dtype,
    threads_per_block,
    binary_op,
    items_per_thread=1,
    algorithm="warp_reductions",
    methods=None,
):
    """Creates an operation that computes a block-wide reduction for thread :sub:`0` using the
    specified binary reduction functor.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(item: dtype) -> dtype)`: Each thread contributes a single item to the reduction.
    - `(items: numba.types.Array) -> dtype`: Each thread contributes an array of items to the
        reduction. The array must contain at least `items_per_thread` items; only the first
        `items_per_thread` items will be included in the reduction.
    - `(item: dtype, num_valid: int) -> dtype`: The first `num_valid` threads contribute a
        single item to the reduction. The items contributed by all other threads are ignored.

    Args:
        dtype: Data type being reduced
        threads_per_block: The number of threads in a block, either an integer
            or a tuple of 2 or 3 integers
        binary_op: Binary reduction function
        items_per_thread: The number of items each thread contributes to the reduction
        algorithm: Algorithm to use for the reduction (one of "raking",
            "raking_commutative_only", "warp_reductions")
        methods: A dict of methods for user-defined types

    Warning:
        The return value is undefined in threads other than thread :sub:`0`.

    Example:
        The code snippet below illustrates a max reduction of 128 integer items that are
        partitioned across 128 threads.

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce
            :end-before: example-end reduce

        Suppose the set of inputs across the block of threads is ``{ 0, 1, 2, 3, ..., 127 }``.
        The corresponding output in the threads thread :sub:`0` will be ``{ 127 }``.
    """
    return _reduce(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        binary_op=binary_op,
        algorithm=algorithm,
        methods=methods,
    )


def sum(
    dtype,
    threads_per_block,
    items_per_thread=1,
    algorithm="warp_reductions",
    methods=None,
):
    """Creates an operation that computes a block-wide reduction for thread :sub:`0` using
    addition (+) as the reduction operator.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(item: dtype) -> dtype)`: Each thread contributes a single item to the reduction.
    - `(items: numba.types.Array) -> dtype`: Each thread contributes an array of items to the
        reduction. The array must contain at least `items_per_thread` items; only the
        first `items_per_thread` items will be included in the reduction.
    - `(item: dtype, num_valid: int) -> dtype`: The first `num_valid` threads contribute a
        single item to the reduction. The items contributed by all other threads are ignored.

    Args:
        dtype: Data type being reduced
        threads_per_block: The number of threads in a block, either an integer
            or a tuple of 2 or 3 integers
        items_per_thread: The number of items each thread owns
        algorithm: Algorithm to use for the reduction (one of "raking",
            "raking_commutative_only", "warp_reductions")
        methods: A dict of methods for user-defined types

    Warning:
        The return value is undefined in threads other than thread :sub:`0`.

    Example:
        The code snippet below illustrates a sum of 128 integer items that are partitioned
        across 128 threads.

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum
            :end-before: example-end sum

        Suppose the set of inputs across the block of threads is ``{ 1, 1, 1, 1, ..., 1 }``.
        The corresponding output in the threads thread :sub:`0` will be ``{ 128 }``.
    """
    return _reduce(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        binary_op=None,
        algorithm=algorithm,
        methods=methods,
    )
