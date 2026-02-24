# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from typing import Callable, Optional, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import signature
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ...warp._warp_reduce import (
    _make_max_rewrite,
    _make_min_rewrite,
    _make_reduce_rewrite,
    _make_sum_rewrite,
)
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopSimpleInstanceType,
    TempStorageType,
    register,
    register_global,
    validate_temp_storage,
    validate_threads_in_warp,
)


# =============================================================================
# Reduce/Sum/Max/Min
# =============================================================================
@register_global(coop.warp.reduce)
class CoopWarpReduceDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.reduce
    impl_key = _make_reduce_rewrite
    primitive_name = "coop.warp.reduce"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: types.Number,
        binary_op: Optional[Callable] = None,
        threads_in_warp: int = 32,
        valid_items: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpReduceDecl.signature).bind(
            src,
            binary_op=binary_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: types.Number,
        *,
        binary_op: Optional[Callable] = None,
        threads_in_warp: int = None,
        valid_items: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpReduceDecl.signature_instance).bind(
            src,
            binary_op=binary_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        if isinstance(src, types.Array):
            raise errors.TypingError(f"{self.primitive_name} requires a scalar input")
        if not isinstance(src, types.Number):
            raise errors.TypingError(f"{self.primitive_name} requires a numeric input")
        arglist = [src]

        binary_op = bound.arguments.get("binary_op")
        binary_op_is_none_type = isinstance(binary_op, types.NoneType)
        if binary_op is None or binary_op_is_none_type:
            if not two_phase:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'binary_op' to be specified"
                )
        else:
            arglist.append(binary_op)

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

        valid_items = bound.arguments.get("valid_items")
        if valid_items is not None:
            if not isinstance(valid_items, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            arglist.append(valid_items)

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            arglist.append(temp_storage)
            temp_storage = None
        if not temp_storage_is_none_type:
            validate_temp_storage(self, temp_storage)
            if temp_storage is not None:
                arglist.append(temp_storage)

        return signature(src, *arglist)


class _CoopWarpUnaryReduceDecl:
    @staticmethod
    def signature(
        src: types.Number,
        threads_in_warp: int = 32,
        valid_items: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(_CoopWarpUnaryReduceDecl.signature).bind(
            src,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        if isinstance(src, types.Array):
            raise errors.TypingError(f"{self.primitive_name} requires a scalar input")
        if not isinstance(src, types.Number):
            raise errors.TypingError(f"{self.primitive_name} requires a numeric input")
        arglist = [src]

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

        valid_items = bound.arguments.get("valid_items")
        if valid_items is not None:
            if not isinstance(valid_items, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            arglist.append(valid_items)

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            arglist.append(temp_storage)
            temp_storage = None
        if not temp_storage_is_none_type:
            validate_temp_storage(self, temp_storage)
            if temp_storage is not None:
                arglist.append(temp_storage)

        return signature(src, *arglist)


@register_global(coop.warp.sum)
class CoopWarpSumDecl(_CoopWarpUnaryReduceDecl, CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.sum
    impl_key = _make_sum_rewrite
    primitive_name = "coop.warp.sum"
    is_constructor = False
    minimum_num_args = 1


@register_global(coop.warp.max)
class CoopWarpMaxDecl(_CoopWarpUnaryReduceDecl, CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.max
    impl_key = _make_max_rewrite
    primitive_name = "coop.warp.max"
    is_constructor = False
    minimum_num_args = 1


@register_global(coop.warp.min)
class CoopWarpMinDecl(_CoopWarpUnaryReduceDecl, CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.min
    impl_key = _make_min_rewrite
    primitive_name = "coop.warp.min"
    is_constructor = False
    minimum_num_args = 1


# =============================================================================
# Instance-related Reduce/Sum/Max/Min Scaffolding
# =============================================================================
class CoopWarpReduceInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpReduceDecl


warp_reduce_instance_type = CoopWarpReduceInstanceType()


@typeof_impl.register(coop.warp.reduce)
def typeof_warp_reduce_instance(*args, **kwargs):
    return warp_reduce_instance_type


@register
class CoopWarpReduceInstanceDecl(CoopInstanceTemplate):
    key = warp_reduce_instance_type
    instance_type = warp_reduce_instance_type
    primitive_name = "coop.warp.reduce"


@register_model(CoopWarpReduceInstanceType)
class CoopWarpReduceInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpReduceInstanceType)
def lower_constant_warp_reduce_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpSumInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpSumDecl


warp_sum_instance_type = CoopWarpSumInstanceType()


@typeof_impl.register(coop.warp.sum)
def typeof_warp_sum_instance(*args, **kwargs):
    return warp_sum_instance_type


@register
class CoopWarpSumInstanceDecl(CoopInstanceTemplate):
    key = warp_sum_instance_type
    instance_type = warp_sum_instance_type
    primitive_name = "coop.warp.sum"


@register_model(CoopWarpSumInstanceType)
class CoopWarpSumInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpSumInstanceType)
def lower_constant_warp_sum_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpMaxInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpMaxDecl


warp_max_instance_type = CoopWarpMaxInstanceType()


@typeof_impl.register(coop.warp.max)
def typeof_warp_max_instance(*args, **kwargs):
    return warp_max_instance_type


@register
class CoopWarpMaxInstanceDecl(CoopInstanceTemplate):
    key = warp_max_instance_type
    instance_type = warp_max_instance_type
    primitive_name = "coop.warp.max"


@register_model(CoopWarpMaxInstanceType)
class CoopWarpMaxInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpMaxInstanceType)
def lower_constant_warp_max_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpMinInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpMinDecl


warp_min_instance_type = CoopWarpMinInstanceType()


@typeof_impl.register(coop.warp.min)
def typeof_warp_min_instance(*args, **kwargs):
    return warp_min_instance_type


@register
class CoopWarpMinInstanceDecl(CoopInstanceTemplate):
    key = warp_min_instance_type
    instance_type = warp_min_instance_type
    primitive_name = "coop.warp.min"


@register_model(CoopWarpMinInstanceType)
class CoopWarpMinInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpMinInstanceType)
def lower_constant_warp_min_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
