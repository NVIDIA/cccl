# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from typing import Optional, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import signature
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ...block._block_radix_rank import (
    _make_radix_rank_rewrite as _make_block_radix_rank_rewrite,
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
# Radix Rank
# =============================================================================
@register_global(coop.block.radix_rank)
class CoopBlockRadixRankDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.radix_rank
    impl_key = _make_block_radix_rank_rewrite
    primitive_name = "coop.block.radix_rank"
    is_constructor = False
    minimum_num_args = 3

    @staticmethod
    def signature(
        items: types.Array,
        ranks: types.Array,
        items_per_thread: int = None,
        begin_bit: Optional[int] = None,
        end_bit: Optional[int] = None,
        descending: Optional[bool] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
        exclusive_digit_prefix: types.Array = None,
    ):
        return inspect.signature(CoopBlockRadixRankDecl.signature).bind(
            items,
            ranks,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            temp_storage=temp_storage,
            exclusive_digit_prefix=exclusive_digit_prefix,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        ranks = bound.arguments["ranks"]

        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )
        if not isinstance(ranks, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'ranks' to be a device or "
                "thread-data array"
            )

        if isinstance(items, types.Array) and isinstance(ranks, types.Array):
            if items.dtype.signed:
                raise errors.TypingError(
                    f"{self.primitive_name} requires unsigned integer item types"
                )
            if not isinstance(ranks.dtype, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires integer 'ranks' arrays"
                )
            if ranks.dtype.bitwidth != 32:
                raise errors.TypingError(
                    f"{self.primitive_name} requires int32 ranks arrays"
                )

        items_per_thread = bound.arguments.get("items_per_thread")
        using_thread_data = isinstance(items, ThreadDataType) or isinstance(
            ranks, ThreadDataType
        )
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        begin_bit = bound.arguments.get("begin_bit")
        end_bit = bound.arguments.get("end_bit")
        if begin_bit is None or end_bit is None:
            raise errors.TypingError(
                f"{self.primitive_name} requires begin_bit and end_bit"
            )
        if not isinstance(begin_bit, types.IntegerLiteral) or not isinstance(
            end_bit, types.IntegerLiteral
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires begin_bit and end_bit to be "
                "integer literals"
            )
        if end_bit.literal_value <= begin_bit.literal_value:
            raise errors.TypingError(
                f"{self.primitive_name} requires end_bit > begin_bit"
            )

        descending = bound.arguments.get("descending")
        if descending is not None and not isinstance(
            descending, (types.Boolean, types.BooleanLiteral, bool)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires descending to be a boolean"
            )

        arglist = [items, ranks]
        if items_per_thread is not None:
            arglist.append(items_per_thread)
        arglist.extend([begin_bit, end_bit])
        if descending is not None:
            arglist.append(descending)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        exclusive_digit_prefix = bound.arguments.get("exclusive_digit_prefix")
        exclusive_prefix_is_none_type = isinstance(
            exclusive_digit_prefix, types.NoneType
        )
        if exclusive_prefix_is_none_type:
            arglist.append(exclusive_digit_prefix)
            exclusive_digit_prefix = None
        if not exclusive_prefix_is_none_type and exclusive_digit_prefix is not None:
            if not isinstance(exclusive_digit_prefix, (types.Array, ThreadDataType)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires exclusive_digit_prefix to be "
                    "a device or thread-data array"
                )
            prefix_dtype = exclusive_digit_prefix.dtype
            if (
                not isinstance(prefix_dtype, types.Integer)
                or prefix_dtype.bitwidth != 32
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires exclusive_digit_prefix to be "
                    "an int32 array"
                )
            arglist.append(exclusive_digit_prefix)
        return signature(types.void, *arglist)


# =============================================================================
# Instance-related Radix Rank Scaffolding
# =============================================================================
class CoopBlockRadixRankInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockRadixRankDecl


block_radix_rank_instance_type = CoopBlockRadixRankInstanceType()


@typeof_impl.register(coop.block.radix_rank)
def typeof_block_radix_rank_instance(*args, **kwargs):
    return block_radix_rank_instance_type


@register
class CoopBlockRadixRankInstanceDecl(CoopInstanceTemplate):
    key = block_radix_rank_instance_type
    instance_type = block_radix_rank_instance_type
    primitive_name = "coop.block.radix_rank"


@register_model(CoopBlockRadixRankInstanceType)
class CoopBlockRadixRankInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockRadixRankInstanceType)
def lower_constant_block_radix_rank_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
