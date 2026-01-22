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
from typing import TYPE_CHECKING

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    Invocable,
    TemplateParameter,
    numba_type_to_wrapper,
)
from .._typing import (
    DimType,
    DtypeType,
)

if TYPE_CHECKING:
    from ._rewrite import CoopNode


class BlockExchangeType(IntEnum):
    """
    Enum representing the type of block exchange operation.
    """

    StripedToBlocked = auto()
    BlockedToStriped = auto()
    WarpStripedToBlocked = auto()
    BlockedToWarpStriped = auto()
    ScatterToBlocked = auto()
    ScatterToStriped = auto()
    ScatterToStripedGuarded = auto()
    ScatterToStripedFlagged = auto()


class exchange(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        block_exchange_type: BlockExchangeType,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        warp_time_slicing: bool = False,
        methods: dict = None,
        unique_id: int = None,
        temp_storage=None,
        use_output_items: bool = False,
        offset_dtype: DtypeType = None,
        valid_flag_dtype: DtypeType = None,
        node: "CoopNode" = None,
    ):
        # Validate initial parameters.
        if block_exchange_type not in BlockExchangeType:
            raise ValueError(
                "block_exchange_type must be a valid BlockExchangeType enum "
                f"value; got: {block_exchange_type!r}"
            )

        if items_per_thread < 1:
            raise ValueError("items_per_thread must be greater than or equal to 1")
        if methods is not None and items_per_thread > 1:
            raise ValueError("items_per_thread must be 1 when using user-defined types")

        uses_ranks = block_exchange_type in (
            BlockExchangeType.ScatterToBlocked,
            BlockExchangeType.ScatterToStriped,
            BlockExchangeType.ScatterToStripedGuarded,
            BlockExchangeType.ScatterToStripedFlagged,
        )
        uses_valid_flags = (
            block_exchange_type == BlockExchangeType.ScatterToStripedFlagged
        )

        if uses_ranks:
            if offset_dtype is None:
                raise ValueError(
                    "offset_dtype must be provided for scatter block exchange types"
                )
            offset_dtype = normalize_dtype_param(offset_dtype)
        elif offset_dtype is not None:
            raise ValueError(
                "offset_dtype is only supported for scatter block exchange types"
            )

        if uses_valid_flags:
            if valid_flag_dtype is None:
                raise ValueError(
                    "valid_flag_dtype must be provided for ScatterToStripedFlagged"
                )
            valid_flag_dtype = normalize_dtype_param(valid_flag_dtype)
        elif valid_flag_dtype is not None:
            raise ValueError(
                "valid_flag_dtype is only supported for ScatterToStripedFlagged"
            )

        self.node = node
        self.block_exchange_type = block_exchange_type
        self.items_per_thread = items_per_thread
        self.dim = dim = normalize_dim_param(threads_per_block)
        self.dtype = dtype = normalize_dtype_param(dtype)
        self.unique_id = unique_id
        self.temp_storage = temp_storage
        self.warp_time_slicing = warp_time_slicing
        self.offset_dtype = offset_dtype
        self.valid_flag_dtype = valid_flag_dtype

        specialization_kwds = {
            "T": dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "WARP_TIME_SLICING": int(warp_time_slicing),
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }
        if uses_ranks:
            specialization_kwds["OffsetT"] = offset_dtype
        if uses_valid_flags:
            specialization_kwds["ValidFlag"] = valid_flag_dtype

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("WARP_TIME_SLICING"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        method_name = {
            BlockExchangeType.StripedToBlocked: "StripedToBlocked",
            BlockExchangeType.BlockedToStriped: "BlockedToStriped",
            BlockExchangeType.WarpStripedToBlocked: "WarpStripedToBlocked",
            BlockExchangeType.BlockedToWarpStriped: "BlockedToWarpStriped",
            BlockExchangeType.ScatterToBlocked: "ScatterToBlocked",
            BlockExchangeType.ScatterToStriped: "ScatterToStriped",
            BlockExchangeType.ScatterToStripedGuarded: "ScatterToStripedGuarded",
            BlockExchangeType.ScatterToStripedFlagged: "ScatterToStripedFlagged",
        }[block_exchange_type]

        input_items = DependentArray(
            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="input_items"
        )
        output_items = DependentArray(
            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="output_items"
        )
        ranks = DependentArray(
            Dependency("OffsetT"), Dependency("ITEMS_PER_THREAD"), name="ranks"
        )
        valid_flags = DependentArray(
            Dependency("ValidFlag"), Dependency("ITEMS_PER_THREAD"), name="valid_flags"
        )

        method = []
        if use_output_items:
            method.extend([input_items, output_items])
        else:
            method.append(input_items)
        if uses_ranks:
            method.append(ranks)
        if uses_valid_flags:
            method.append(valid_flags)

        parameters = [method]

        # If we have a non-None `methods`, we're dealing with user-defined types.
        if methods is not None:
            type_definitions = [
                numba_type_to_wrapper(dtype, methods=methods),
            ]
        else:
            type_definitions = None

        self.algorithm = Algorithm(
            "BlockExchange",
            method_name,
            "block_exchange",
            ["cub/block/block_exchange.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            unique_id=unique_id,
        )

        self.specialization = self.algorithm.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        block_exchange_type: BlockExchangeType,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        warp_time_slicing: bool = False,
        methods: dict = None,
        use_output_items: bool = False,
        offset_dtype: DtypeType = None,
        valid_flag_dtype: DtypeType = None,
    ):
        algo = cls(
            block_exchange_type=block_exchange_type,
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            warp_time_slicing=warp_time_slicing,
            methods=methods,
            use_output_items=use_output_items,
            offset_dtype=offset_dtype,
            valid_flag_dtype=valid_flag_dtype,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
