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

from ...warp._warp_exchange import _make_exchange_rewrite
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopSimpleInstanceType,
    TempStorageType,
    process_items_per_thread,
    register,
    register_global,
    validate_src_dst,
    validate_temp_storage,
    validate_threads_in_warp,
)


# =============================================================================
# Exchange
# =============================================================================
@register_global(coop.warp.exchange)
class CoopWarpExchangeDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.exchange
    impl_key = _make_exchange_rewrite
    primitive_name = "coop.warp.exchange"
    is_constructor = False
    minimum_num_args = 1
    default_exchange_type = coop.warp.WarpExchangeType.StripedToBlocked

    @staticmethod
    def signature(
        items: types.Array,
        output_items: types.Array = None,
        items_per_thread: int = None,
        ranks: types.Array = None,
        warp_exchange_type: coop.warp.WarpExchangeType = None,
        threads_in_warp: int = 32,
        offset_dtype: Optional[types.Type] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpExchangeDecl.signature).bind(
            items,
            output_items=output_items,
            items_per_thread=items_per_thread,
            ranks=ranks,
            warp_exchange_type=warp_exchange_type,
            threads_in_warp=threads_in_warp,
            offset_dtype=offset_dtype,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        items: types.Array,
        output_items: types.Array = None,
        ranks: types.Array = None,
        *,
        items_per_thread: int = None,
        warp_exchange_type: coop.warp.WarpExchangeType = None,
        threads_in_warp: int = None,
        offset_dtype: Optional[types.Type] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpExchangeDecl.signature_instance).bind(
            items,
            output_items=output_items,
            ranks=ranks,
            items_per_thread=items_per_thread,
            warp_exchange_type=warp_exchange_type,
            threads_in_warp=threads_in_warp,
            offset_dtype=offset_dtype,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments.get("output_items")
        ranks = bound.arguments.get("ranks")
        if not isinstance(items, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device array"
            )
        if output_items is not None:
            validate_src_dst(self, items, output_items)

        arglist = [items]
        if output_items is not None:
            arglist.append(output_items)

        process_items_per_thread(self, bound, arglist, two_phase, target_array=items)

        warp_exchange_type = bound.arguments.get("warp_exchange_type")
        warp_exchange_is_none_type = isinstance(warp_exchange_type, types.NoneType)
        if warp_exchange_type is None or warp_exchange_is_none_type:
            if not two_phase:
                warp_exchange_type = self.default_exchange_type
            else:
                warp_exchange_type = None
        if warp_exchange_type is not None:
            if isinstance(warp_exchange_type, enum.IntEnum):
                if warp_exchange_type not in coop.warp.WarpExchangeType:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires a WarpExchangeType value"
                    )
            elif isinstance(warp_exchange_type, types.EnumMember):
                if warp_exchange_type.instance_class is not coop.warp.WarpExchangeType:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires a WarpExchangeType value"
                    )
            else:
                raise errors.TypingError(
                    f"{self.primitive_name} requires a WarpExchangeType value"
                )
            arglist.append(warp_exchange_type)

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

        if warp_exchange_type == coop.warp.WarpExchangeType.ScatterToStriped:
            if ranks is None:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'ranks' for ScatterToStriped"
                )
        if ranks is not None:
            if not isinstance(ranks, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'ranks' to be an array"
                )
            if not isinstance(ranks.dtype, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'ranks' to be integer array"
                )
            arglist.append(ranks)

            offset_dtype = bound.arguments.get("offset_dtype")
            if offset_dtype is not None and not isinstance(
                offset_dtype, (types.DType, types.Type)
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'offset_dtype' to be a dtype"
                )
            if offset_dtype is not None:
                arglist.append(offset_dtype)
        elif ranks is None and warp_exchange_type is not None:
            if warp_exchange_type != coop.warp.WarpExchangeType.ScatterToStriped:
                offset_dtype = bound.arguments.get("offset_dtype")
                if offset_dtype is not None:
                    raise errors.TypingError(
                        f"{self.primitive_name} only accepts 'offset_dtype' with 'ranks'"
                    )
        elif ranks is not None and warp_exchange_type is None and not two_phase:
            raise errors.TypingError(
                f"{self.primitive_name} only accepts 'ranks' for ScatterToStriped"
            )
        elif ranks is None and warp_exchange_type is None and not two_phase:
            offset_dtype = bound.arguments.get("offset_dtype")
            if offset_dtype is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} only accepts 'offset_dtype' with 'ranks'"
                )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


# =============================================================================
# Instance-related Exchange Scaffolding
# =============================================================================
class CoopWarpExchangeInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpExchangeDecl


warp_exchange_instance_type = CoopWarpExchangeInstanceType()


@typeof_impl.register(coop.warp.exchange)
def typeof_warp_exchange_instance(*args, **kwargs):
    return warp_exchange_instance_type


@register
class CoopWarpExchangeInstanceDecl(CoopInstanceTemplate):
    key = warp_exchange_instance_type
    instance_type = warp_exchange_instance_type
    primitive_name = "coop.warp.exchange"


@register_model(CoopWarpExchangeInstanceType)
class CoopWarpExchangeInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpExchangeInstanceType)
def lower_constant_warp_exchange_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
