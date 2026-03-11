# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
cuda.coop.block_exchange
====================================

This module provides a set of :ref:`collective <collective-primitives>` methods
for rearranging data partitioned across CUDA thread blocks.

Supported C++ APIs
++++++++++++++++++

The following :cpp:class:`cub.BlockExchange` APIs are supported:

    StripedToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD])
    StripedToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::false_type)
    StripedToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::true_type)
    StripedToBlocked void (T (&)[ITEMS_PER_THREAD])

Unsupported C++ APIs
++++++++++++++++++++

The following :cpp:class:`cub.BlockExchange` APIs are not yet supported:

    BlockedToStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD])
    BlockedToStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::false_type)
    BlockedToStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::true_type)
    BlockedToStriped void (T (&)[ITEMS_PER_THREAD])

    BlockedToWarpStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD])
    BlockedToWarpStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::false_type)
    BlockedToWarpStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::true_type)
    BlockedToWarpStriped void (T (&)[ITEMS_PER_THREAD])

    ScatterToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])
    ScatterToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD], ::cuda::std::false_type)
    ScatterToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD], ::cuda::std::true_type)
    ScatterToBlocked template void (T (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])

    ScatterToStripedFlagged template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD], ValidFlag (&)[ITEMS_PER_THREAD])
    ScatterToStripedFlagged template void (T (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD], ValidFlag (&)[ITEMS_PER_THREAD])

    ScatterToStripedGuarded template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])
    ScatterToStripedGuarded template void (T (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])

    ScatterToStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])
    ScatterToStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD], ::cuda::std::false_type)
    ScatterToStriped template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD], ::cuda::std::true_type)
    ScatterToStriped template void (T (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])

    WarpStripedToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD])
    WarpStripedToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::false_type)
    WarpStripedToBlocked template void (const T (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], ::cuda::std::true_type)
    WarpStripedToBlocked void (T (&)[ITEMS_PER_THREAD])
"""

from enum import IntEnum, auto

import numba

from .._common import (
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    Dependency,
    DependentArray,
    Invocable,
    Pointer,
    TemplateParameter,
    numba_type_to_wrapper,
)
from .._typing import (
    DimType,
    DtypeType,
)


class BlockExchangeType(IntEnum):
    """
    Enum representing the type of block exchange operation.  Currently
    only :py:attr:`StripedToBlocked` is supported.
    """

    StripedToBlocked = auto()


def make_exchange(
    block_exchange_type: BlockExchangeType,
    dtype: DtypeType,
    threads_per_block: DimType,
    items_per_thread: int,
    warp_time_slicing: bool = False,
    methods: dict = None,
):
    """
    Creates a block-wide exchange primitive for rearranging data
    partitioned across CUDA thread blocks.

    Example:
        The snippet below shows how to create and invoke the returned
        ``block_exchange`` primitive for striped-to-blocked exchange.

        .. code-block:: python

           block_exchange = coop.block.make_exchange(
               coop.block.BlockExchangeType.StripedToBlocked,
               dtype=numba.int32,
               threads_per_block=128,
               items_per_thread=4,
           )
           temp_storage_bytes = block_exchange.temp_storage_bytes

           @cuda.jit(link=block_exchange.files)
           def kernel(thread_data):
               temp_storage = cuda.shared.array(
                   shape=temp_storage_bytes,
                   dtype=numba.uint8,
               )
               block_exchange(temp_storage, thread_data)

    :param block_exchange_type: Exchange mode to perform. Currently, only
        :py:attr:`StripedToBlocked` is supported.
    :param dtype: Data type of input and output values.
    :type dtype: :py:class:`cuda.coop._typing.DtypeType`
    :param threads_per_block: Number of threads in the block.
    :type threads_per_block: :py:class:`cuda.coop._typing.DimType`
    :param items_per_thread: Number of items owned by each thread.
    :type items_per_thread: int
    :param warp_time_slicing: Whether to use warp time-slicing. If true,
        shared memory usage is reduced at the expense of parallelism.
    :type warp_time_slicing: bool, optional

    :param methods: Optional method dictionary for user-defined types.
    :type methods: dict, optional

    :raises ValueError: If ``block_exchange_type`` is not a valid enum
        value of :py:class:`BlockExchangeType`.

    :raises ValueError: If ``items_per_thread`` is less than 1.

    :returns: An :py:class:`cuda.coop._types.Invocable` object
        representing the specialized kernel callable from a Numba JIT'd
        CUDA kernel.

    """
    # Validate initial parameters.
    if block_exchange_type not in BlockExchangeType:
        raise ValueError(
            "block_exchange_type must be a valid BlockExchangeType enum "
            f"value; got: {block_exchange_type!r}"
        )
    elif block_exchange_type != BlockExchangeType.StripedToBlocked:
        raise ValueError(
            "block_exchange_type must be BlockExchangeType.StripedToBlocked; "
            f"got: {block_exchange_type!r}"
        )

    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    # Normalize parameters.
    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    specialization_kwds = {
        "T": dtype,
        "BLOCK_DIM_X": dim[0],
        "ITEMS_PER_THREAD": items_per_thread,
        "WARP_TIME_SLICING": int(warp_time_slicing),
        "BLOCK_DIM_Y": dim[1],
        "BLOCK_DIM_Z": dim[2],
    }

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("WARP_TIME_SLICING"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    # In other modules, like block scan, items_per_thread affects the calling
    # convention the user needs to adopt in their kernel, as it pertains to
    # input and return parameters, e.g., with items_per_thread = 1, you'd use
    #   output = block_scan(temp_storage, input)
    # Whereas with items_per_thread > 1, you'd use:
    #   block_scan(temp_storage, input, output)
    # This idiom does not apply for block exchange, so we don't specialize
    # the parameters based on items_per_thread.
    parameters = [
        # Signature:
        # void BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD,
        #                    WARP_TIME_SLICING, BLOCK_DIM_Y, BLOCK_DIM_Z>(
        #     temp_storage
        # )::StripedToBlocked(
        #   const T (&)[ITEMS_PER_THREAD] input_items,
        #   OutputT (&)[ITEMS_PER_THREAD] output_items,
        # )
        [
            # temp_storage
            Pointer(numba.uint8),
            # T (&)[ITEMS_PER_THREAD] items
            DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
        ],
        # Signature:
        # void BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD,
        #                    WARP_TIME_SLICING, BLOCK_DIM_Y, BLOCK_DIM_Z>(
        #     temp_storage
        # )::StripedToBlocked(
        #   const T (&)[ITEMS_PER_THREAD] input_items,
        #   OutputT (&)[ITEMS_PER_THREAD] output_items,
        # )
        [
            # temp_storage
            Pointer(numba.uint8),
            # const T (&)[ITEMS_PER_THREAD] input_items
            DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
            # OutputT (&)[ITEMS_PER_THREAD] output_items
            DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
        ],
    ]

    # If we have a non-None `methods`, we're dealing with user-defined types.
    if methods is not None:
        type_definitions = [
            numba_type_to_wrapper(dtype, methods=methods),
        ]
    else:
        type_definitions = None

    template = Algorithm(
        "BlockExchange",
        "StripedToBlocked",
        "block_exchange",
        ["cub/block/block_exchange.cuh"],
        template_parameters,
        parameters,
        fake_return=False,
        type_definitions=type_definitions,
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
