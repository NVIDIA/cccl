# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum
import inspect
from typing import Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import signature
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ...block._block_exchange import (
    _make_exchange_rewrite as _make_block_exchange_rewrite,
)
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopSimpleInstanceType,
    TempStorageType,
    ThreadDataType,
    register,
    validate_items_per_thread,
    validate_src_dst,
    validate_temp_storage,
)


# =============================================================================
# Exchange
# =============================================================================
class CoopBlockExchangeDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.exchange
    impl_key = _make_block_exchange_rewrite
    primitive_name = "coop.block.exchange"
    is_constructor = False
    minimum_num_args = 1
    default_exchange_type = coop.block.BlockExchangeType.StripedToBlocked

    @staticmethod
    def signature(
        items: types.Array,
        output_items: types.Array = None,
        items_per_thread: int = None,
        ranks: types.Array = None,
        valid_flags: types.Array = None,
        block_exchange_type: coop.block.BlockExchangeType = None,
        warp_time_slicing: bool = False,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockExchangeDecl.signature).bind(
            items,
            output_items=output_items,
            items_per_thread=items_per_thread,
            ranks=ranks,
            valid_flags=valid_flags,
            block_exchange_type=block_exchange_type,
            warp_time_slicing=warp_time_slicing,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        items: types.Array,
        output_items: types.Array = None,
        ranks: types.Array = None,
        valid_flags: types.Array = None,
        *,
        items_per_thread: int = None,
        block_exchange_type: coop.block.BlockExchangeType = None,
        warp_time_slicing: bool = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockExchangeDecl.signature_instance).bind(
            items,
            output_items=output_items,
            ranks=ranks,
            valid_flags=valid_flags,
            items_per_thread=items_per_thread,
            block_exchange_type=block_exchange_type,
            warp_time_slicing=warp_time_slicing,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments.get("output_items")
        ranks = bound.arguments.get("ranks")
        valid_flags = bound.arguments.get("valid_flags")
        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )

        using_thread_data = isinstance(items, ThreadDataType) or (
            output_items is not None and isinstance(output_items, ThreadDataType)
        )
        if output_items is not None:
            validate_src_dst(self, items, output_items)

        items_per_thread = bound.arguments.get("items_per_thread")
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        block_exchange_type = bound.arguments.get("block_exchange_type")
        block_exchange_is_none_type = isinstance(block_exchange_type, types.NoneType)
        if block_exchange_type is None or block_exchange_is_none_type:
            if not two_phase:
                block_exchange_type = self.default_exchange_type
            else:
                block_exchange_type = None
        if block_exchange_type is None:
            exchange_type_value = None
        elif isinstance(block_exchange_type, enum.IntEnum):
            if block_exchange_type not in coop.block.BlockExchangeType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_exchange_type' to be "
                    "a BlockExchangeType enum value"
                )
            exchange_type_value = block_exchange_type
        else:
            if not isinstance(block_exchange_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_exchange_type' to be "
                    "a BlockExchangeType enum value"
                )
            if block_exchange_type.instance_class is not coop.block.BlockExchangeType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_exchange_type' to be "
                    "a BlockExchangeType enum value"
                )
            exchange_type_value = None

        if exchange_type_value is None:
            if ranks is not None:
                if not isinstance(ranks, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be a device array"
                    )
                if not isinstance(ranks.dtype, types.Integer):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be an integer array"
                    )
            if valid_flags is not None:
                if not isinstance(valid_flags, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a device "
                        "array"
                    )
                if not isinstance(valid_flags.dtype, (types.Integer, types.Boolean)):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a "
                        "boolean or integer array"
                    )
        else:
            uses_ranks = exchange_type_value in (
                coop.block.BlockExchangeType.ScatterToBlocked,
                coop.block.BlockExchangeType.ScatterToStriped,
                coop.block.BlockExchangeType.ScatterToStripedGuarded,
                coop.block.BlockExchangeType.ScatterToStripedFlagged,
            )
            uses_valid_flags = (
                exchange_type_value
                == coop.block.BlockExchangeType.ScatterToStripedFlagged
            )

            if uses_ranks:
                if ranks is None:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' for scatter exchanges"
                    )
                if not isinstance(ranks, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be a device array"
                    )
                if not isinstance(ranks.dtype, types.Integer):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be an integer array"
                    )
            elif ranks is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'ranks' for "
                    f"{exchange_type_value.name}"
                )

            if uses_valid_flags:
                if valid_flags is None:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' for "
                        "ScatterToStripedFlagged"
                    )
                if not isinstance(valid_flags, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a device "
                        "array"
                    )
                if not isinstance(valid_flags.dtype, (types.Integer, types.Boolean)):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a "
                        "boolean or integer array"
                    )
            elif valid_flags is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'valid_flags' for "
                    f"{exchange_type_value.name}"
                )

        warp_time_slicing = bound.arguments.get("warp_time_slicing")
        warp_time_slicing_is_none_type = isinstance(warp_time_slicing, types.NoneType)
        if warp_time_slicing is None or warp_time_slicing_is_none_type:
            if not two_phase:
                warp_time_slicing = False
            else:
                warp_time_slicing = None
        if warp_time_slicing is not None and not isinstance(
            warp_time_slicing, (types.Boolean, types.BooleanLiteral, bool)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'warp_time_slicing' to be a boolean"
            )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        arglist = [items]
        if output_items is not None:
            arglist.append(output_items)
        if items_per_thread is not None:
            arglist.append(items_per_thread)
        if ranks is not None:
            arglist.append(ranks)
        if valid_flags is not None:
            arglist.append(valid_flags)
        if block_exchange_type is not None:
            arglist.append(block_exchange_type)
        if warp_time_slicing is not None:
            arglist.append(warp_time_slicing)
        if temp_storage is not None:
            arglist.append(temp_storage)

        sig = signature(types.void, *arglist)

        return sig


# =============================================================================
# Block Primitives (Two-phase Instances)
# =============================================================================


class CoopBlockExchangeInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockExchangeDecl


block_exchange_instance_type = CoopBlockExchangeInstanceType()


@typeof_impl.register(coop.block.exchange)
def typeof_block_exchange_instance(*args, **kwargs):
    return block_exchange_instance_type


@register
class CoopBlockExchangeInstanceDecl(CoopInstanceTemplate):
    key = block_exchange_instance_type
    instance_type = block_exchange_instance_type
    primitive_name = "coop.block.exchange"


@register_model(CoopBlockExchangeInstanceType)
class CoopBlockExchangeInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockExchangeInstanceType)
def lower_constant_block_exchange_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
