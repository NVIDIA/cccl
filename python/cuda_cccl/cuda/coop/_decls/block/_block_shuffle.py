# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum
import inspect
from typing import Optional, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import signature
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ...block._block_shuffle import _make_shuffle_rewrite as _make_block_shuffle_rewrite
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
# Shuffle
# =============================================================================
@register_global(coop.block.shuffle)
class CoopBlockShuffleDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.shuffle
    impl_key = _make_block_shuffle_rewrite
    primitive_name = "coop.block.shuffle"
    is_constructor = False
    minimum_num_args = 1
    default_shuffle_type = coop.block.BlockShuffleType.Up

    @staticmethod
    def signature(
        items: Union[types.Array, types.Number],
        output_items: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        block_shuffle_type: coop.block.BlockShuffleType = None,
        distance: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
        block_prefix: types.Array = None,
        block_suffix: types.Array = None,
    ):
        return inspect.signature(CoopBlockShuffleDecl.signature).bind(
            items,
            output_items=output_items,
            items_per_thread=items_per_thread,
            block_shuffle_type=block_shuffle_type,
            distance=distance,
            temp_storage=temp_storage,
            block_prefix=block_prefix,
            block_suffix=block_suffix,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments.get("output_items")

        items_is_array = isinstance(items, (types.Array, ThreadDataType))
        items_is_scalar = isinstance(items, types.Number)

        if not items_is_array and not items_is_scalar:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a scalar or array"
            )

        if items_is_scalar and output_items is not None:
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'output_items' for scalar "
                "shuffle operations"
            )

        if items_is_array:
            if not isinstance(output_items, (types.Array, ThreadDataType)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'output_items' to be a device "
                    "or thread-data array for Up/Down shuffles"
                )

            using_thread_data = isinstance(items, ThreadDataType) or isinstance(
                output_items, ThreadDataType
            )
            if (
                not using_thread_data
                and isinstance(items, types.Array)
                and isinstance(output_items, types.Array)
            ):
                if items.dtype != output_items.dtype:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'items' and "
                        "'output_items' to have matching dtypes"
                    )

        block_shuffle_type = bound.arguments.get("block_shuffle_type")
        if block_shuffle_type is None:
            if items_is_scalar:
                block_shuffle_type = coop.block.BlockShuffleType.Offset
            else:
                block_shuffle_type = self.default_shuffle_type

        block_shuffle_type_value = None
        if isinstance(block_shuffle_type, enum.IntEnum):
            if block_shuffle_type not in coop.block.BlockShuffleType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_shuffle_type' to be a "
                    "BlockShuffleType enum value"
                )
            block_shuffle_type_value = block_shuffle_type
        else:
            if not isinstance(block_shuffle_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_shuffle_type' to be a "
                    "BlockShuffleType enum value"
                )
            if block_shuffle_type.instance_class is not coop.block.BlockShuffleType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_shuffle_type' to be a "
                    "BlockShuffleType enum value"
                )

        array_shuffle = items_is_array
        scalar_shuffle = items_is_scalar
        if block_shuffle_type_value is not None:
            array_shuffle = block_shuffle_type_value in (
                coop.block.BlockShuffleType.Up,
                coop.block.BlockShuffleType.Down,
            )
            scalar_shuffle = block_shuffle_type_value in (
                coop.block.BlockShuffleType.Offset,
                coop.block.BlockShuffleType.Rotate,
                coop.block.BlockShuffleType.Up,
                coop.block.BlockShuffleType.Down,
            )

            if items_is_scalar and not scalar_shuffle:
                raise errors.TypingError(
                    f"{self.primitive_name} requires a valid BlockShuffleType for "
                    "scalar shuffles"
                )
            if items_is_array and not array_shuffle:
                raise errors.TypingError(
                    f"{self.primitive_name} requires Up or Down for array shuffles"
                )

        items_per_thread = bound.arguments.get("items_per_thread")
        if items_is_array:
            if not isinstance(items, ThreadDataType) and not isinstance(
                output_items, ThreadDataType
            ):
                if not two_phase or items_per_thread is not None:
                    items_per_thread = validate_items_per_thread(self, items_per_thread)

        distance = bound.arguments.get("distance")
        if distance is not None and scalar_shuffle:
            if not isinstance(distance, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'distance' to be an integer"
                )
        if distance is not None and array_shuffle:
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'distance' for Up/Down shuffles"
            )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        block_prefix = bound.arguments.get("block_prefix")
        block_suffix = bound.arguments.get("block_suffix")

        if block_prefix is not None or block_suffix is not None:
            if not items_is_array:
                raise errors.TypingError(
                    f"{self.primitive_name} only supports block_prefix/block_suffix "
                    "for Up/Down shuffles with array inputs"
                )
            if block_prefix is not None and block_suffix is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not allow block_prefix and "
                    "block_suffix together"
                )
            if block_shuffle_type_value is not None:
                if block_shuffle_type_value == coop.block.BlockShuffleType.Up:
                    if block_prefix is not None:
                        raise errors.TypingError(
                            f"{self.primitive_name} does not allow block_prefix for "
                            "Up shuffles"
                        )
                if block_shuffle_type_value == coop.block.BlockShuffleType.Down:
                    if block_suffix is not None:
                        raise errors.TypingError(
                            f"{self.primitive_name} does not allow block_suffix for "
                            "Down shuffles"
                        )
            if block_prefix is not None and not isinstance(block_prefix, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_prefix to be a device array"
                )
            if block_suffix is not None and not isinstance(block_suffix, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_suffix to be a device array"
                )

            item_dtype = items.dtype if isinstance(items, types.Array) else None
            if ThreadDataType is not None and isinstance(items, ThreadDataType):
                item_dtype = items.dtype

            if item_dtype is not None:
                if (
                    block_prefix is not None
                    and isinstance(block_prefix, types.Array)
                    and block_prefix.dtype != item_dtype
                ):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires block_prefix to have the "
                        "same dtype as items"
                    )
                if (
                    block_suffix is not None
                    and isinstance(block_suffix, types.Array)
                    and block_suffix.dtype != item_dtype
                ):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires block_suffix to have the "
                        "same dtype as items"
                    )

        if items_is_array:
            arglist = [items, output_items]
            if items_per_thread is not None:
                arglist.append(items_per_thread)
            arglist.append(block_shuffle_type)
            if block_prefix is not None:
                arglist.append(block_prefix)
            if block_suffix is not None:
                arglist.append(block_suffix)
            if temp_storage is not None:
                arglist.append(temp_storage)
            return signature(types.void, *arglist)

        arglist = [items]
        if distance is not None:
            arglist.append(distance)
        arglist.append(block_shuffle_type)
        if temp_storage is not None:
            arglist.append(temp_storage)
        return signature(items, *arglist)


# =============================================================================
# Instance-related Shuffle Scaffolding
# =============================================================================
class CoopBlockShuffleInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockShuffleDecl


block_shuffle_instance_type = CoopBlockShuffleInstanceType()


@typeof_impl.register(coop.block.shuffle)
def typeof_block_shuffle_instance(*args, **kwargs):
    return block_shuffle_instance_type


@register
class CoopBlockShuffleInstanceDecl(CoopInstanceTemplate):
    key = block_shuffle_instance_type
    instance_type = block_shuffle_instance_type
    primitive_name = "coop.block.shuffle"


@register_model(CoopBlockShuffleInstanceType)
class CoopBlockShuffleInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockShuffleInstanceType)
def lower_constant_block_shuffle_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
