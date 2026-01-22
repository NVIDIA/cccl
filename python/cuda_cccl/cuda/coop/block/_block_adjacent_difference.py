# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
cuda.coop.block_adjacent_difference
====================================

Block-wide adjacent difference primitives based on :cpp:class:`cub::BlockAdjacentDifference`.
"""

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

import numba

from .._common import normalize_dim_param, normalize_dtype_param
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
    Value,
    numba_type_to_wrapper,
)
from .._typing import DimType, DtypeType

if TYPE_CHECKING:
    from ._rewrite import CoopNode


class BlockAdjacentDifferenceType(IntEnum):
    SubtractLeft = auto()
    SubtractRight = auto()


class adjacent_difference(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        block_adjacent_difference_type: BlockAdjacentDifferenceType,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        difference_op,
        methods: dict = None,
        unique_id: int = None,
        valid_items: Any = None,
        tile_predecessor_item: Any = None,
        tile_successor_item: Any = None,
        temp_storage: Any = None,
        node: "CoopNode" = None,
    ) -> None:
        if block_adjacent_difference_type not in BlockAdjacentDifferenceType:
            raise ValueError(
                "block_adjacent_difference_type must be a valid "
                "BlockAdjacentDifferenceType value; got: "
                f"{block_adjacent_difference_type!r}"
            )
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be greater than or equal to 1")
        if difference_op is None:
            raise ValueError("difference_op must be provided")
        if tile_predecessor_item is not None and tile_successor_item is not None:
            raise ValueError(
                "Only one of tile_predecessor_item or tile_successor_item may be set"
            )

        self.node = node
        self.block_adjacent_difference_type = block_adjacent_difference_type
        self.items_per_thread = items_per_thread
        self.dim = dim = normalize_dim_param(threads_per_block)
        self.dtype = dtype = normalize_dtype_param(dtype)
        self.unique_id = unique_id
        self.temp_storage = temp_storage
        self.valid_items = valid_items
        self.tile_predecessor_item = tile_predecessor_item
        self.tile_successor_item = tile_successor_item
        self.difference_op = difference_op

        use_partial_tile = valid_items is not None
        use_tile_item = (
            tile_predecessor_item is not None or tile_successor_item is not None
        )

        if block_adjacent_difference_type == BlockAdjacentDifferenceType.SubtractLeft:
            method_name = (
                "SubtractLeftPartialTile" if use_partial_tile else "SubtractLeft"
            )
            if tile_successor_item is not None:
                raise ValueError("tile_successor_item is not valid for SubtractLeft")
            tile_item_name = "tile_predecessor_item"
        else:
            method_name = (
                "SubtractRightPartialTile" if use_partial_tile else "SubtractRight"
            )
            if tile_predecessor_item is not None:
                raise ValueError("tile_predecessor_item is not valid for SubtractRight")
            tile_item_name = "tile_successor_item"

        specialization_kwds = {
            "T": dtype,
            "BLOCK_DIM_X": dim[0],
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
            "ITEMS_PER_THREAD": items_per_thread,
            "DiffOp": difference_op,
        }

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        input_items = DependentArray(
            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="input_items"
        )
        output_items = DependentArray(
            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="output_items"
        )
        difference_op_param = DependentPythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=[Dependency("T"), Dependency("T")],
            op=Dependency("DiffOp"),
            name="difference_op",
        )

        method = [input_items, output_items, difference_op_param]
        if use_partial_tile:
            method.append(Value(numba.int32, name="valid_items"))
        if use_tile_item:
            method.append(DependentReference(Dependency("T"), name=tile_item_name))

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

        type_definitions = None
        if methods is not None:
            type_definitions = [numba_type_to_wrapper(dtype, methods=methods)]

        self.algorithm = Algorithm(
            "BlockAdjacentDifference",
            method_name,
            "block_adjacent_difference",
            ["cub/block/block_adjacent_difference.cuh"],
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
        difference_op,
        block_adjacent_difference_type: BlockAdjacentDifferenceType = BlockAdjacentDifferenceType.SubtractLeft,
        methods: dict = None,
        valid_items: Any = None,
        tile_predecessor_item: Any = None,
        tile_successor_item: Any = None,
        temp_storage: Any = None,
    ):
        algo = cls(
            block_adjacent_difference_type,
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            difference_op=difference_op,
            methods=methods,
            valid_items=valid_items,
            tile_predecessor_item=tile_predecessor_item,
            tile_successor_item=tile_successor_item,
            temp_storage=temp_storage,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
