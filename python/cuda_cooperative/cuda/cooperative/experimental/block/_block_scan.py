# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Callable, Literal, Union

import numba

from cuda.cooperative.experimental._common import (
    CUB_BLOCK_SCAN_ALGOS,
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from cuda.cooperative.experimental._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentOperator,
    DependentReference,
    Invocable,
    Pointer,
    TemplateParameter,
)
from cuda.cooperative.experimental._typing import DimType

if TYPE_CHECKING:
    import numpy as np


def _scan(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    items_per_thread: int = 1,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: Literal["+"] = "+",
    block_prefix_callback_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    if algorithm not in CUB_BLOCK_SCAN_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    if mode == "exclusive":
        cpp_func_prefix = "Exclusive"
    else:
        if mode != "inclusive":
            raise ValueError(f"Unsupported mode: {mode}")
        cpp_func_prefix = "Inclusive"

    specialization_kwds = {
        "T": dtype,
        "BLOCK_DIM_X": dim[0],
        "ALGORITHM": CUB_BLOCK_SCAN_ALGOS[algorithm],
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

    fake_return = False

    if scan_op != "+":
        raise ValueError(f"Unsupported scan_op: {scan_op}")
    else:
        if items_per_thread == 1:
            parameters = [
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T, # input
                #     T& # output
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T input
                    DependentReference(Dependency("T")),
                    # T& output
                    DependentReference(Dependency("T"), is_output=True),
                ],
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T,                     # input
                #     T&,                    # output
                #     BlockPrefixCallbackOp& # block_prefix_callback_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T input
                    DependentReference(Dependency("T")),
                    # T& output
                    DependentReference(Dependency("T"), is_output=True),
                    # BlockPrefixCallbackOp& block_prefix_callback_op
                    DependentOperator(
                        Dependency("T"),
                        [Dependency("T")],
                        Dependency("BlockPrefixCallbackOp"),
                    ),
                ],
            ]

            fake_return = True

        else:
            assert items_per_thread > 1, items_per_thread

            parameters = [
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ITEMS_PER_THREAD, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T (&)[ITEMS_PER_THREAD], # input
                #     T (&)[ITEMS_PER_THREAD]  # output
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T (&)[ITEMS_PER_THREAD] input
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T (&)[ITEMS_PER_THREAD] output
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                ],
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ITEMS_PER_THREAD, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T (&)[ITEMS_PER_THREAD], # input
                #     T (&)[ITEMS_PER_THREAD], # output
                #     BlockPrefixCallbackOp&   # block_prefix_callback_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T (&)[ITEMS_PER_THREAD] input
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T (&)[ITEMS_PER_THREAD] output
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # BlockPrefixCallbackOp& block_prefix_callback_op
                    DependentOperator(
                        Dependency("T"),
                        [Dependency("T")],
                        Dependency("BlockPrefixCallbackOp"),
                    ),
                ],
            ]

            specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread

    template = Algorithm(
        "BlockScan",
        f"{cpp_func_prefix}Sum",
        "block_scan",
        ["cub/block/block_scan.cuh"],
        template_parameters,
        parameters,
        fake_return=fake_return,
    )

    if block_prefix_callback_op is not None:
        specialization_kwds["BlockPrefixCallbackOp"] = block_prefix_callback_op

    specialization = template.specialize(specialization_kwds)
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.get_temp_storage_bytes(),
        algorithm=specialization,
    )


def exclusive_sum(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an exclusive block-wide prefix scan using addition (+) as the
    scan operator.  The value of 0 is applied as the initial value, and is
    assigned to the first output element in the first thread.

    Example:
        The code snippet below illustrates an exclusive prefix sum of 512
        integer items that are partitioned in a
        :ref:`blocked arrangement <flexible-data-arrangement>` across 128
        threads where each thread owns 4 consecutive items.

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the
        ``exclusive_sum`` API:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum

        Suppose the set of input ``thread_data`` across the block of threads is
        ``{ [1, 1, 1, 1], [1, 1, 1, 1], ..., [1, 1, 1, 1] }``.

        The corresponding output ``thread_data`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.

    Args:
        dtype: Supplies the data type of the input and output arrays.

        threads_per_block: Supplies the number of threads in the block, either
            as an integer for a 1D block or a tuple of two or three integers
            for a 2D or 3D block, respectively.

        items_per_thread: Supplies the number of items partitioned onto each
            thread.

        prefix_op: Optionally supplies a callable that will be invoked by the
            first warp of threads in a block with the block aggregate value;
            only the return value of the first lane in the warp is applied as
            the prefix value.

        algorithm: Optionally supplies the algorithm to use for the block-wide
            scan.  Must be one of the following: ``"raking"``,
            ``"raking_memoize"``, or ``"warp_scans"``.  The default is
            ``"raking"``.

    Returns:
        A callable that can be linked to a CUDA kernel and invoked to perform
        the block-wide exclusive prefix scan.

    """
    return _scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="exclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
    )


def inclusive_sum(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an inclusive block-wide prefix scan using addition (+) as the
    scan operator.

    Args:
        dtype: Supplies the data type of the input and output arrays.

        threads_per_block: Supplies the number of threads in the block, either
            as an integer for a 1D block or a tuple of two or three integers
            for a 2D or 3D block, respectively.

        items_per_thread: Supplies the number of items partitioned onto each
            thread.

        prefix_op: Optionally supplies a callable that will be invoked by the
            first warp of threads in a block with the block aggregate value;
            only the return value of the first lane in the warp is applied as
            the prefix value.

        algorithm: Optionally supplies the algorithm to use for the block-wide
            scan.  Must be one of the following: ``"raking"``,
            ``"raking_memoize"``, or ``"warp_scans"``.  The default is
            ``"raking"``.

    Returns:
        A callable that can be linked to a CUDA kernel and invoked to perform
        the block-wide inclusive prefix scan.
    """
    return _scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="inclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
    )
