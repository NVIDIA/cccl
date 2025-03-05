# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Callable, Literal, Union

import numba

from cuda.cooperative.experimental._common import (
    make_binary_tempfile,
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

if TYPE_CHECKING:
    import numpy as np

CUB_BLOCK_SCAN_ALGOS = {
    "raking": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING",
    "raking_memoize": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE",
    "warp_scans": "::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS",
}


def _scan(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: int,
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

    # Normalize the dtype parameter.
    dtype = normalize_dtype_param(dtype)

    if mode == "exclusive":
        cpp_func_prefix = "Exclusive"
    else:
        if mode != "inclusive":
            raise ValueError(f"Unsupported mode: {mode}")
        cpp_func_prefix = "Inclusive"

    specialization_kwds = {
        "T": dtype,
        "BLOCK_DIM_X": threads_per_block,
        "ALGORITHM": CUB_BLOCK_SCAN_ALGOS[algorithm],
    }

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ALGORITHM"),
    ]

    fake_return = False

    if scan_op != "+":
        raise ValueError(f"Unsupported scan_op: {scan_op}")
    else:
        if items_per_thread == 1:
            parameters = [
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM>(
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
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM>(
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

            template_parameters.append(TemplateParameter("ITEMS_PER_THREAD"))

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
    threads_per_block: int,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an exclusive block-wide prefix sum.
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
    threads_per_block: int,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an inclusive block-wide prefix sum.
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
