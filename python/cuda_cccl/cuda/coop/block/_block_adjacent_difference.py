# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
        """
        Creates a block-wide adjacent-difference primitive backed by
        ``cub::BlockAdjacentDifference``.

        Example:
            The snippet below computes left-differences for scalar inputs.

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_adjacent_difference_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_adjacent_difference_api.py
                :language: python
                :dedent:
                :start-after: example-begin subtract-left
                :end-before: example-end subtract-left

        :param block_adjacent_difference_type: Selects whether adjacent
            differences are computed against the left or right neighbor.
        :type block_adjacent_difference_type: BlockAdjacentDifferenceType

        :param dtype: Element dtype for the per-thread input/output item arrays.
        :type dtype: DtypeType

        :param threads_per_block: CUDA block dimensions as an int or
            ``(x, y, z)`` tuple.
        :type threads_per_block: DimType

        :param items_per_thread: Number of items processed by each thread.
            Must be greater than or equal to ``1``.
        :type items_per_thread: int

        :param difference_op: Binary callable used to compute adjacent
            differences.
        :type difference_op: Callable

        :param methods: Optional user-defined-type adapter methods.
        :type methods: dict, optional

        :param unique_id: Optional unique suffix used for generated symbols.
        :type unique_id: int, optional

        :param valid_items: Optional count of valid items for partial-tile
            APIs.
        :type valid_items: Any, optional

        :param tile_predecessor_item: Optional tile predecessor item for
            boundary handling; valid only for ``SubtractLeft``.
        :type tile_predecessor_item: Any, optional

        :param tile_successor_item: Optional tile successor item for boundary
            handling; valid only for ``SubtractRight``.
        :type tile_successor_item: Any, optional

        :param temp_storage: Optional explicit temporary storage argument.
        :type temp_storage: Any, optional

        :param node: Internal rewrite node used by single-phase rewriting.
        :type node: CoopNode, optional

        :raises ValueError: If ``block_adjacent_difference_type`` is invalid.
        :raises ValueError: If ``items_per_thread < 1``.
        :raises ValueError: If ``difference_op`` is not provided.
        :raises ValueError: If both tile boundary items are provided.
        :raises ValueError: If a tile boundary argument is incompatible with
            the selected adjacent-difference direction.
        """
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


def _build_adjacent_difference_spec(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    difference_op=None,
    block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
    **kwargs,
):
    kw = dict(kwargs)
    if threads_per_block is None:
        threads_per_block = kw.pop("dim", None)
    spec = {
        "block_adjacent_difference_type": block_adjacent_difference_type,
        "dtype": dtype,
        "threads_per_block": threads_per_block,
        "items_per_thread": items_per_thread,
        "difference_op": difference_op,
    }
    spec.update(kw)
    return spec


def _make_adjacent_difference_two_phase(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    difference_op=None,
    block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
    **kwargs,
):
    spec = _build_adjacent_difference_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        difference_op=difference_op,
        block_adjacent_difference_type=block_adjacent_difference_type,
        **kwargs,
    )
    return adjacent_difference.create(**spec)


def _make_adjacent_difference_rewrite(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    difference_op=None,
    block_adjacent_difference_type=BlockAdjacentDifferenceType.SubtractLeft,
    **kwargs,
):
    spec = _build_adjacent_difference_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        difference_op=difference_op,
        block_adjacent_difference_type=block_adjacent_difference_type,
        **kwargs,
    )
    return adjacent_difference(**spec)
