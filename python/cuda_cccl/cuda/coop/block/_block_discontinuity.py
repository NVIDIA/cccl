# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
cuda.coop.block_discontinuity
====================================

This module provides a set of block-wide discontinuity flagging primitives
based on :cpp:class:`cub::BlockDiscontinuity`.
"""

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

import numba

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    DependentReference,
    Invocable,
    TemplateParameter,
    TempStoragePointer,
    numba_type_to_wrapper,
)
from .._typing import (
    DimType,
    DtypeType,
)

if TYPE_CHECKING:
    from ._rewrite import CoopNode


class BlockDiscontinuityType(IntEnum):
    HEADS = auto()
    TAILS = auto()
    HEADS_AND_TAILS = auto()


class discontinuity(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        flag_op,
        flag_dtype: DtypeType,
        block_discontinuity_type: BlockDiscontinuityType = BlockDiscontinuityType.HEADS,
        methods: dict = None,
        unique_id: int = None,
        temp_storage=None,
        tile_predecessor_item: Any = None,
        tile_successor_item: Any = None,
        node: "CoopNode" = None,
    ) -> None:
        """
        Flags discontinuities across a block using the supplied comparison
        operator.

        Example:
            The snippet below flags heads (changes) between adjacent items.

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_discontinuity_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_discontinuity_api.py
                :language: python
                :dedent:
                :start-after: example-begin flag-heads
                :end-before: example-end flag-heads
        """
        if block_discontinuity_type not in BlockDiscontinuityType:
            raise ValueError(
                "block_discontinuity_type must be a valid BlockDiscontinuityType "
                f"value; got: {block_discontinuity_type!r}"
            )
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be greater than or equal to 1")

        if flag_op is None:
            raise ValueError("flag_op must be provided for block discontinuity")

        self.node = node
        self.block_discontinuity_type = block_discontinuity_type
        self.items_per_thread = items_per_thread
        self.dim = dim = normalize_dim_param(threads_per_block)
        self.dtype = dtype = normalize_dtype_param(dtype)
        self.flag_dtype = flag_dtype = normalize_dtype_param(flag_dtype)
        self.unique_id = unique_id
        self.temp_storage = temp_storage
        self.tile_predecessor_item = tile_predecessor_item
        self.tile_successor_item = tile_successor_item
        self.flag_op = flag_op

        if block_discontinuity_type == BlockDiscontinuityType.HEADS:
            if tile_successor_item is not None:
                raise ValueError("tile_successor_item is not valid for HEADS")
        elif block_discontinuity_type == BlockDiscontinuityType.TAILS:
            if tile_predecessor_item is not None:
                raise ValueError("tile_predecessor_item is not valid for TAILS")

        specialization_kwds = {
            "T": dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
            "FlagT": flag_dtype,
            "FlagOp": flag_op,
        }

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        method_name = {
            BlockDiscontinuityType.HEADS: "FlagHeads",
            BlockDiscontinuityType.TAILS: "FlagTails",
            BlockDiscontinuityType.HEADS_AND_TAILS: "FlagHeadsAndTails",
        }[block_discontinuity_type]

        input_items = DependentArray(
            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="input_items"
        )
        head_flags = DependentArray(
            Dependency("FlagT"), Dependency("ITEMS_PER_THREAD"), name="head_flags"
        )
        tail_flags = DependentArray(
            Dependency("FlagT"), Dependency("ITEMS_PER_THREAD"), name="tail_flags"
        )
        flag_op_param = DependentPythonOperator(
            ret_dtype=Dependency("FlagT"),
            arg_dtypes=[Dependency("T"), Dependency("T")],
            op=Dependency("FlagOp"),
            name="flag_op",
        )

        tile_predecessor_param = DependentReference(
            Dependency("T"), name="tile_predecessor_item"
        )
        tile_successor_param = DependentReference(
            Dependency("T"), name="tile_successor_item"
        )

        if block_discontinuity_type == BlockDiscontinuityType.HEADS:
            method = [head_flags, input_items, flag_op_param]
            if tile_predecessor_item is not None:
                method.append(tile_predecessor_param)
        elif block_discontinuity_type == BlockDiscontinuityType.TAILS:
            method = [tail_flags, input_items, flag_op_param]
            if tile_successor_item is not None:
                method.append(tile_successor_param)
        else:
            if tile_predecessor_item is not None and tile_successor_item is not None:
                method = [
                    head_flags,
                    tile_predecessor_param,
                    tail_flags,
                    tile_successor_param,
                    input_items,
                    flag_op_param,
                ]
            elif tile_predecessor_item is not None:
                method = [
                    head_flags,
                    tile_predecessor_param,
                    tail_flags,
                    input_items,
                    flag_op_param,
                ]
            elif tile_successor_item is not None:
                method = [
                    head_flags,
                    tail_flags,
                    tile_successor_param,
                    input_items,
                    flag_op_param,
                ]
            else:
                method = [head_flags, tail_flags, input_items, flag_op_param]

        if temp_storage is not None:
            method.insert(
                0,
                TempStoragePointer(
                    numba.types.uint8,
                    is_array_pointer=True,
                    name="temp_storage",
                ),
            )

        parameters = [method]

        if methods is not None:
            type_definitions = [
                numba_type_to_wrapper(dtype, methods=methods),
            ]
        else:
            type_definitions = None

        self.algorithm = Algorithm(
            "BlockDiscontinuity",
            method_name,
            "block_discontinuity",
            ["cub/block/block_discontinuity.cuh"],
            template_parameters,
            parameters,
            self,
            unique_id=unique_id,
            type_definitions=type_definitions,
        )
        self.specialization = self.algorithm.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        flag_op,
        block_discontinuity_type: BlockDiscontinuityType = BlockDiscontinuityType.HEADS,
        flag_dtype: DtypeType = None,
        methods: dict = None,
        temp_storage: Any = None,
        tile_predecessor_item: Any = None,
        tile_successor_item: Any = None,
    ):
        if flag_dtype is None:
            flag_dtype = numba.types.boolean

        algo = cls(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            flag_op=flag_op,
            flag_dtype=flag_dtype,
            block_discontinuity_type=block_discontinuity_type,
            methods=methods,
            temp_storage=temp_storage,
            tile_predecessor_item=tile_predecessor_item,
            tile_successor_item=tile_successor_item,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
