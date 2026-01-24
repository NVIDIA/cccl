# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
cuda.coop.block_shuffle
====================================

Block-wide shuffle primitives based on :cpp:class:`cub::BlockShuffle`.
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
    DependentPointerReference,
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


class BlockShuffleType(IntEnum):
    Offset = auto()
    Rotate = auto()
    Up = auto()
    Down = auto()


class shuffle(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        block_shuffle_type: BlockShuffleType,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int = None,
        distance: int = None,
        block_prefix: Any = None,
        block_suffix: Any = None,
        methods: dict = None,
        unique_id: int = None,
        temp_storage: Any = None,
        node: "CoopNode" = None,
    ) -> None:
        """
        Shuffles items across a thread block using the selected shuffle type.

        Example:
            The snippet below demonstrates a scalar offset shuffle.

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_shuffle_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_shuffle_api.py
                :language: python
                :dedent:
                :start-after: example-begin offset-scalar
                :end-before: example-end offset-scalar
        """
        if block_shuffle_type not in BlockShuffleType:
            raise ValueError(
                "block_shuffle_type must be a valid BlockShuffleType value; got: "
                f"{block_shuffle_type!r}"
            )

        self.node = node
        self.block_shuffle_type = block_shuffle_type
        self.dim = dim = normalize_dim_param(threads_per_block)
        self.dtype = dtype = normalize_dtype_param(dtype)
        self.items_per_thread = items_per_thread
        self.distance = distance
        self.block_prefix = block_prefix
        self.block_suffix = block_suffix
        self.unique_id = unique_id
        self.temp_storage = temp_storage

        method_name = {
            BlockShuffleType.Offset: "Offset",
            BlockShuffleType.Rotate: "Rotate",
            BlockShuffleType.Up: "Up",
            BlockShuffleType.Down: "Down",
        }[block_shuffle_type]

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        specialization_kwds = {
            "T": self.dtype,
            "BLOCK_DIM_X": dim[0],
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }

        use_array_inputs = (
            block_shuffle_type
            in (
                BlockShuffleType.Up,
                BlockShuffleType.Down,
            )
            and items_per_thread is not None
        )
        fake_return = not use_array_inputs
        if block_shuffle_type in (BlockShuffleType.Up, BlockShuffleType.Down):
            fake_return = True

        if use_array_inputs:
            if items_per_thread is None or items_per_thread < 1:
                raise ValueError("items_per_thread must be >= 1 for Up/Down shuffles")
            if block_shuffle_type == BlockShuffleType.Up and block_prefix is not None:
                raise ValueError("block_prefix is not valid for Up shuffles")
            if block_shuffle_type == BlockShuffleType.Down and block_suffix is not None:
                raise ValueError("block_suffix is not valid for Down shuffles")
            specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread
            method = [
                DependentArray(
                    Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="input_items"
                ),
                DependentArray(
                    Dependency("T"),
                    Dependency("ITEMS_PER_THREAD"),
                    name="output_items",
                ),
            ]
            if block_shuffle_type == BlockShuffleType.Up and block_suffix is not None:
                method.append(
                    DependentPointerReference(
                        Dependency("T"),
                        name="block_suffix",
                        is_array_pointer=True,
                    )
                )
            if block_shuffle_type == BlockShuffleType.Down and block_prefix is not None:
                method.append(
                    DependentPointerReference(
                        Dependency("T"),
                        name="block_prefix",
                        is_array_pointer=True,
                    )
                )
        else:
            if items_per_thread is not None and block_shuffle_type not in (
                BlockShuffleType.Up,
                BlockShuffleType.Down,
            ):
                raise ValueError("items_per_thread is only valid for Up/Down shuffles")
            if block_prefix is not None or block_suffix is not None:
                raise ValueError(
                    "block_prefix/block_suffix are only valid for Up/Down shuffles"
                )
            method = [
                DependentReference(Dependency("T"), name="input_item"),
                DependentReference(Dependency("T"), name="output_item", is_output=True),
            ]
            if distance is None:
                distance = 1
            self.distance = distance
            method.append(Value(numba.types.int32, name="distance"))

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
            type_definitions = [numba_type_to_wrapper(self.dtype, methods=methods)]

        self.algorithm = Algorithm(
            "BlockShuffle",
            method_name,
            "block_shuffle",
            ["cub/block/block_shuffle.cuh"],
            template_parameters,
            parameters,
            self,
            unique_id=unique_id,
            type_definitions=type_definitions,
            fake_return=fake_return,
        )
        self.specialization = self.algorithm.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int = None,
        block_shuffle_type: BlockShuffleType = BlockShuffleType.Up,
        distance: int = None,
        methods: dict = None,
        temp_storage: Any = None,
    ):
        algo = cls(
            block_shuffle_type,
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            distance=distance,
            block_prefix=None,
            block_suffix=None,
            methods=methods,
            temp_storage=temp_storage,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
