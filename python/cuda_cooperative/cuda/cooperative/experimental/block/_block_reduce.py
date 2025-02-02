# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from cuda.cooperative.experimental._common import make_binary_tempfile
from cuda.cooperative.experimental._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentOperator,
    DependentReference,
    Invocable,
    Pointer,
    TemplateParameter,
    Value,
    numba_type_to_wrapper,
)


def reduce(dtype, threads_in_block, binary_op, items_per_thread=1, methods=None):
    """Creates an operation that computes a block-wide reduction for thread\ :sub:`0` using the
    specified binary reduction functor.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(item: dtype) -> dtype)`: Each thread contributes a single item to the reduction.
    - `(items: numba.types.Array) -> dtype`: Each thread contributes an array of items to the
        reduction. The array must be 1D and contain at least `items_per_thread` items; only the
        first `items_per_thread` items will be included in the reduction.
    - `(item: dtype, num_valid: int) -> dtype`: The first `num_valid` threads contribute a
        single item to the reduction. The items contributed by all other threads are ignored.

    Args:
        dtype: Data type being reduced
        threads_in_block: The number of threads in a block
        binary_op: Binary reduction function
        items_per_thread: The number of items each thread contributes to the reduction

    Warning:
        The return value is undefined in threads other than thread\ :sub:`0`.

    Example:
        The code snippet below illustrates a max reduction of 128 integer items that are
        partitioned across 128 threads.

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce
            :end-before: example-end reduce

        Suppose the set of inputs across the block of threads is ``{ 0, 1, 2, 3, ..., 127 }``.
        The corresponding output in the threads thread\ :sub:`0` will be ``{ 127 }``.
    """
    template = Algorithm(
        "BlockReduce",
        "Reduce",
        "block_reduce",
        ["cub/block/block_reduce.cuh"],
        [TemplateParameter("T"), TemplateParameter("BLOCK_DIM_X")],
        [
            # Signatures:
            # T Reduce(T(&)[ITEMS_PER_THREAD], Op);
            [
                Pointer(numba.uint8),
                DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                DependentOperator(
                    Dependency("T"),
                    [Dependency("T"), Dependency("T")],
                    Dependency("Op")
                ),
                DependentReference(Dependency("T"), True)
            ],
            # T Reduce(T&, Op);
            [
                Pointer(numba.uint8),
                DependentReference(Dependency("T")),
                DependentOperator(
                    Dependency("T"),
                    [Dependency("T"), Dependency("T")],
                    Dependency("Op"),
                ),
                DependentReference(Dependency("T"), True),
            ],
            # T Reduce(T&, Op, int num_valid);
            [
                Pointer(numba.uint8),
                DependentReference(Dependency("T")),
                DependentOperator(
                    Dependency("T"),
                    [Dependency("T"), Dependency("T")],
                    Dependency("Op")
                ),
                Value(numba.int32),
                DependentReference(Dependency("T"), True)
            ]
        ],
        type_definitions=[numba_type_to_wrapper(dtype, methods=methods)],
    )
    specialization = template.specialize(
        {"T": dtype, "BLOCK_DIM_X": threads_in_block, "ITEMS_PER_THREAD": items_per_thread, "Op": binary_op}
    )

    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.get_temp_storage_bytes(),
        algorithm=specialization,
    )


def sum(dtype, threads_in_block, items_per_thread=1, methods=None):
    """Creates an operation that computes a block-wide reduction for thread\ :sub:`0` using
    addition (+) as the reduction operator.

    Returns a callable object that can be linked to and invoked from device code. It can be
    invoked with the following signatures:

    - `(item: dtype) -> dtype)`: Each thread contributes a single item to the reduction.
    - `(items: numba.types.Array) -> dtype`: Each thread contributes an array of items to the
        reduction. The array must be 1D and contain at least `items_per_thread` items; only the
        first `items_per_thread` items will be included in the reduction.
    - `(item: dtype, num_valid: int) -> dtype`: The first `num_valid` threads contribute a
        single item to the reduction. The items contributed by all other threads are ignored.

    Args:
        dtype: Data type being reduced
        threads_in_block: The number of threads in a block
        items_per_thread: The number of items each thread owns

    Warning:
        The return value is undefined in threads other than thread\ :sub:`0`.

    Example:
        The code snippet below illustrates a sum of 128 integer items that are partitioned
        across 128 threads.

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum
            :end-before: example-end sum

        Suppose the set of inputs across the block of threads is ``{ 1, 1, 1, 1, ..., 1 }``.
        The corresponding output in the threads thread\ :sub:`0` will be ``{ 128 }``.
    """
    template = Algorithm(
        "BlockReduce",
        "Sum",
        "block_reduce",
        ["cub/block/block_reduce.cuh"],
        [TemplateParameter("T"), TemplateParameter("BLOCK_DIM_X")],
        [
            # Signatures:
            # T Sum(T(&)[ITEMS_PER_THREAD]);
            [
                Pointer(numba.uint8),
                DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                DependentReference(Dependency("T"), True)
            ],
            # T Sum(T&);
            [
                Pointer(numba.uint8),
                DependentReference(Dependency("T")),
                DependentReference(Dependency("T"), True),
            ],
            # T Sum(T&, int num_valid);
            [
                Pointer(numba.uint8),
                DependentReference(Dependency("T")),
                Value(numba.int32),
                DependentReference(Dependency("T"), True),
            ]
        ],
        type_definitions=[numba_type_to_wrapper(dtype, methods=methods)],
    )
    specialization = template.specialize({"T": dtype, "BLOCK_DIM_X": threads_in_block, "ITEMS_PER_THREAD": items_per_thread})
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.get_temp_storage_bytes(),
        algorithm=specialization,
    )
