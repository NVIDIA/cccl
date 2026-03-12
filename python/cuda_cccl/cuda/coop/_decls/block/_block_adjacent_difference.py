# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum
import inspect
from typing import Any, Callable, Optional, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import signature
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ...block._block_adjacent_difference import (
    _make_adjacent_difference_rewrite as _make_block_adjacent_difference_rewrite,
)
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopSimpleInstanceType,
    TempStorageType,
    ThreadDataType,
    register,
    register_global,
    validate_items_per_thread,
    validate_temp_storage,
)


# =============================================================================
# Adjacent Difference
# =============================================================================
@register_global(coop.block.adjacent_difference)
class CoopBlockAdjacentDifferenceDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.adjacent_difference
    impl_key = _make_block_adjacent_difference_rewrite
    primitive_name = "coop.block.adjacent_difference"
    is_constructor = False
    minimum_num_args = 2
    default_difference_type = coop.block.BlockAdjacentDifferenceType.SubtractLeft

    @staticmethod
    def signature(
        items: types.Array,
        output_items: types.Array,
        items_per_thread: int = None,
        difference_op: Optional[Callable] = None,
        block_adjacent_difference_type: coop.block.BlockAdjacentDifferenceType = None,
        valid_items: Optional[int] = None,
        tile_predecessor_item: Optional[Any] = None,
        tile_successor_item: Optional[Any] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockAdjacentDifferenceDecl.signature).bind(
            items,
            output_items,
            items_per_thread=items_per_thread,
            difference_op=difference_op,
            block_adjacent_difference_type=block_adjacent_difference_type,
            valid_items=valid_items,
            tile_predecessor_item=tile_predecessor_item,
            tile_successor_item=tile_successor_item,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments["output_items"]

        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )
        if not isinstance(output_items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'output_items' to be a device or "
                "thread-data array"
            )

        using_thread_data = isinstance(items, ThreadDataType) or isinstance(
            output_items, ThreadDataType
        )
        if not using_thread_data:
            if isinstance(items, types.Array) and isinstance(output_items, types.Array):
                if items.dtype != output_items.dtype:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'items' and "
                        "'output_items' to have matching dtypes"
                    )

        items_per_thread = bound.arguments.get("items_per_thread")
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        block_adjacent_difference_type = bound.arguments.get(
            "block_adjacent_difference_type"
        )
        if block_adjacent_difference_type is None:
            block_adjacent_difference_type = self.default_difference_type
        if isinstance(block_adjacent_difference_type, enum.IntEnum):
            if (
                block_adjacent_difference_type
                not in coop.block.BlockAdjacentDifferenceType
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_adjacent_difference_type' "
                    "to be a BlockAdjacentDifferenceType enum value"
                )
        else:
            if not isinstance(block_adjacent_difference_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_adjacent_difference_type' "
                    "to be a BlockAdjacentDifferenceType enum value"
                )
            if (
                block_adjacent_difference_type.instance_class
                is not coop.block.BlockAdjacentDifferenceType
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_adjacent_difference_type' "
                    "to be a BlockAdjacentDifferenceType enum value"
                )

        difference_op = bound.arguments.get("difference_op")
        if difference_op is None:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'difference_op' to be specified"
            )

        valid_items = bound.arguments.get("valid_items")
        if valid_items is not None and not isinstance(
            valid_items, (types.Integer, types.IntegerLiteral)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'valid_items' to be an integer"
            )

        tile_predecessor_item = bound.arguments.get("tile_predecessor_item")
        tile_successor_item = bound.arguments.get("tile_successor_item")
        if tile_predecessor_item is not None and tile_successor_item is not None:
            raise errors.TypingError(
                f"{self.primitive_name} accepts only one of 'tile_predecessor_item' "
                "or 'tile_successor_item'"
            )
        if (
            block_adjacent_difference_type
            == coop.block.BlockAdjacentDifferenceType.SubtractLeft
            and tile_successor_item is not None
        ):
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'tile_successor_item' for "
                "SubtractLeft"
            )
        if (
            block_adjacent_difference_type
            == coop.block.BlockAdjacentDifferenceType.SubtractRight
            and tile_predecessor_item is not None
        ):
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'tile_predecessor_item' for "
                "SubtractRight"
            )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        arglist = [items, output_items]
        if items_per_thread is not None:
            arglist.append(items_per_thread)
        arglist.append(difference_op)
        if block_adjacent_difference_type is not None:
            arglist.append(block_adjacent_difference_type)
        if valid_items is not None:
            arglist.append(valid_items)
        if tile_predecessor_item is not None:
            arglist.append(tile_predecessor_item)
        if tile_successor_item is not None:
            arglist.append(tile_successor_item)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


# =============================================================================
# Instance-related Adjacent Difference Scaffolding
# =============================================================================
class CoopBlockAdjacentDifferenceInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockAdjacentDifferenceDecl


block_adjacent_difference_instance_type = CoopBlockAdjacentDifferenceInstanceType()


@typeof_impl.register(coop.block.adjacent_difference)
def typeof_block_adjacent_difference_instance(*args, **kwargs):
    return block_adjacent_difference_instance_type


@register
class CoopBlockAdjacentDifferenceInstanceDecl(CoopInstanceTemplate):
    key = block_adjacent_difference_instance_type
    instance_type = block_adjacent_difference_instance_type
    primitive_name = "coop.block.adjacent_difference"


@register_model(CoopBlockAdjacentDifferenceInstanceType)
class CoopBlockAdjacentDifferenceInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockAdjacentDifferenceInstanceType)
def lower_constant_block_adjacent_difference_instance_type(
    context, builder, typ, value
):
    return context.get_dummy_value()
