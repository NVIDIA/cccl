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

from ...block._block_discontinuity import (
    _make_discontinuity_rewrite as _make_block_discontinuity_rewrite,
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
# Discontinuity
# =============================================================================
@register_global(coop.block.discontinuity)
class CoopBlockDiscontinuityDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.discontinuity
    impl_key = _make_block_discontinuity_rewrite
    primitive_name = "coop.block.discontinuity"
    is_constructor = False
    minimum_num_args = 2
    default_discontinuity_type = coop.block.BlockDiscontinuityType.HEADS

    @staticmethod
    def signature(
        items: types.Array,
        head_flags: types.Array,
        tail_flags: types.Array = None,
        items_per_thread: int = None,
        flag_op: Optional[Callable] = None,
        block_discontinuity_type: coop.block.BlockDiscontinuityType = None,
        tile_predecessor_item: Optional[Any] = None,
        tile_successor_item: Optional[Any] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockDiscontinuityDecl.signature).bind(
            items,
            head_flags,
            tail_flags=tail_flags,
            items_per_thread=items_per_thread,
            flag_op=flag_op,
            block_discontinuity_type=block_discontinuity_type,
            tile_predecessor_item=tile_predecessor_item,
            tile_successor_item=tile_successor_item,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        head_flags = bound.arguments["head_flags"]
        tail_flags = bound.arguments.get("tail_flags")

        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )
        if not isinstance(head_flags, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'head_flags' to be a device or "
                "thread-data array"
            )
        if tail_flags is not None and not isinstance(
            tail_flags, (types.Array, ThreadDataType)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tail_flags' to be a device or "
                "thread-data array"
            )

        using_thread_data = isinstance(items, ThreadDataType) or isinstance(
            head_flags, ThreadDataType
        )
        if tail_flags is not None and isinstance(tail_flags, ThreadDataType):
            using_thread_data = True

        items_per_thread = bound.arguments.get("items_per_thread")
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        block_discontinuity_type = bound.arguments.get("block_discontinuity_type")
        if block_discontinuity_type is None:
            block_discontinuity_type = self.default_discontinuity_type
        discontinuity_value = None
        if isinstance(block_discontinuity_type, enum.IntEnum):
            if block_discontinuity_type not in coop.block.BlockDiscontinuityType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_discontinuity_type' "
                    "to be a BlockDiscontinuityType enum value"
                )
            discontinuity_value = block_discontinuity_type
        else:
            if not isinstance(block_discontinuity_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_discontinuity_type' "
                    "to be a BlockDiscontinuityType enum value"
                )
            if (
                block_discontinuity_type.instance_class
                is not coop.block.BlockDiscontinuityType
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_discontinuity_type' "
                    "to be a BlockDiscontinuityType enum value"
                )

        if (
            discontinuity_value == coop.block.BlockDiscontinuityType.HEADS_AND_TAILS
            and tail_flags is None
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tail_flags' for HEADS_AND_TAILS"
            )

        flag_op = bound.arguments.get("flag_op")
        if flag_op is None:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'flag_op' to be specified"
            )

        tile_predecessor_item = bound.arguments.get("tile_predecessor_item")
        tile_successor_item = bound.arguments.get("tile_successor_item")
        if tile_predecessor_item is not None and isinstance(
            tile_predecessor_item, (types.Array, ThreadDataType)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tile_predecessor_item' to be a scalar"
            )
        if tile_successor_item is not None and isinstance(
            tile_successor_item, (types.Array, ThreadDataType)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tile_successor_item' to be a scalar"
            )
        if discontinuity_value == coop.block.BlockDiscontinuityType.HEADS:
            if tile_successor_item is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'tile_successor_item' "
                    "for HEADS"
                )
        if discontinuity_value == coop.block.BlockDiscontinuityType.TAILS:
            if tile_predecessor_item is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'tile_predecessor_item' "
                    "for TAILS"
                )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        arglist = [items, head_flags]

        if tail_flags is not None:
            arglist.append(tail_flags)

        if items_per_thread is not None:
            arglist.append(items_per_thread)

        arglist.append(flag_op)

        if block_discontinuity_type is not None:
            arglist.append(block_discontinuity_type)

        if tile_predecessor_item is not None:
            arglist.append(tile_predecessor_item)
        if tile_successor_item is not None:
            arglist.append(tile_successor_item)

        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


# =============================================================================
# Instance-related Discontinuity Scaffolding
# =============================================================================
class CoopBlockDiscontinuityInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockDiscontinuityDecl


block_discontinuity_instance_type = CoopBlockDiscontinuityInstanceType()


@typeof_impl.register(coop.block.discontinuity)
def typeof_block_discontinuity_instance(*args, **kwargs):
    return block_discontinuity_instance_type


@register
class CoopBlockDiscontinuityInstanceDecl(CoopInstanceTemplate):
    key = block_discontinuity_instance_type
    instance_type = block_discontinuity_instance_type
    primitive_name = "coop.block.discontinuity"


@register_model(CoopBlockDiscontinuityInstanceType)
class CoopBlockDiscontinuityInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockDiscontinuityInstanceType)
def lower_constant_block_discontinuity_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
