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

from ...block._block_radix_sort import (
    _make_radix_sort_keys_descending_rewrite as _make_block_radix_sort_keys_descending_rewrite,
)
from ...block._block_radix_sort import (
    _make_radix_sort_keys_rewrite as _make_block_radix_sort_keys_rewrite,
)
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopSimpleInstanceType,
    TempStorageType,
    process_items_per_thread,
    register,
    register_global,
    validate_temp_storage,
)


# =============================================================================
# Radix Sort
# =============================================================================
@register_global(coop.block.radix_sort_keys)
class CoopBlockRadixSortDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.radix_sort_keys
    impl_key = _make_block_radix_sort_keys_rewrite
    primitive_name = "coop.block.radix_sort_keys"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        keys: types.Array,
        items_per_thread: int = None,
        begin_bit: Optional[int] = None,
        end_bit: Optional[int] = None,
        values: types.Array = None,
        decomposer: Optional[Callable] = None,
        blocked_to_striped: Optional[bool] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockRadixSortDecl.signature).bind(
            keys,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            values=values,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        keys = bound.arguments["keys"]
        if not isinstance(keys, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'keys' to be a device array"
            )

        arglist = [keys]
        process_items_per_thread(self, bound, arglist, two_phase, target_array=keys)

        begin_bit = bound.arguments.get("begin_bit")
        end_bit = bound.arguments.get("end_bit")
        if (begin_bit is None) != (end_bit is None):
            raise errors.TypingError(
                f"{self.primitive_name} requires both 'begin_bit' and 'end_bit'"
            )
        if begin_bit is not None:
            if not isinstance(begin_bit, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'begin_bit' to be an integer"
                )
            if not isinstance(end_bit, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'end_bit' to be an integer"
                )
            arglist.extend([begin_bit, end_bit])

        values = bound.arguments.get("values")
        values_is_none_type = isinstance(values, types.NoneType)
        if values_is_none_type:
            arglist.append(values)
            values = None
        if values is not None:
            if not isinstance(values, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'values' to be a device array"
                )
            arglist.append(values)

        decomposer = bound.arguments.get("decomposer")
        decomposer_is_none_type = isinstance(decomposer, types.NoneType)
        if decomposer_is_none_type:
            arglist.append(decomposer)
            decomposer = None
        if decomposer is not None:
            arglist.append(decomposer)

        blocked_to_striped = bound.arguments.get("blocked_to_striped")
        blocked_is_none_type = isinstance(blocked_to_striped, types.NoneType)
        if blocked_is_none_type:
            arglist.append(blocked_to_striped)
            blocked_to_striped = None
        if blocked_to_striped is not None:
            if not isinstance(
                blocked_to_striped, (types.Boolean, types.BooleanLiteral, bool)
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'blocked_to_striped' to be a boolean"
                )
            arglist.append(blocked_to_striped)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


@register_global(coop.block.radix_sort_keys_descending)
class CoopBlockRadixSortDescendingDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.radix_sort_keys_descending
    impl_key = _make_block_radix_sort_keys_descending_rewrite
    primitive_name = "coop.block.radix_sort_keys_descending"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        keys: types.Array,
        items_per_thread: int = None,
        begin_bit: Optional[int] = None,
        end_bit: Optional[int] = None,
        values: types.Array = None,
        decomposer: Optional[Callable] = None,
        blocked_to_striped: Optional[bool] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockRadixSortDescendingDecl.signature).bind(
            keys,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            values=values,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        keys = bound.arguments["keys"]
        if not isinstance(keys, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'keys' to be a device array"
            )

        arglist = [keys]
        process_items_per_thread(self, bound, arglist, two_phase, target_array=keys)

        begin_bit = bound.arguments.get("begin_bit")
        end_bit = bound.arguments.get("end_bit")
        if (begin_bit is None) != (end_bit is None):
            raise errors.TypingError(
                f"{self.primitive_name} requires both 'begin_bit' and 'end_bit'"
            )
        if begin_bit is not None:
            if not isinstance(begin_bit, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'begin_bit' to be an integer"
                )
            if not isinstance(end_bit, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'end_bit' to be an integer"
                )
            arglist.extend([begin_bit, end_bit])

        values = bound.arguments.get("values")
        values_is_none_type = isinstance(values, types.NoneType)
        if values_is_none_type:
            arglist.append(values)
            values = None
        if values is not None:
            if not isinstance(values, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'values' to be a device array"
                )
            arglist.append(values)

        decomposer = bound.arguments.get("decomposer")
        decomposer_is_none_type = isinstance(decomposer, types.NoneType)
        if decomposer_is_none_type:
            arglist.append(decomposer)
            decomposer = None
        if decomposer is not None:
            arglist.append(decomposer)

        blocked_to_striped = bound.arguments.get("blocked_to_striped")
        blocked_is_none_type = isinstance(blocked_to_striped, types.NoneType)
        if blocked_is_none_type:
            arglist.append(blocked_to_striped)
            blocked_to_striped = None
        if blocked_to_striped is not None:
            if not isinstance(
                blocked_to_striped, (types.Boolean, types.BooleanLiteral, bool)
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'blocked_to_striped' to be a boolean"
                )
            arglist.append(blocked_to_striped)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


# =============================================================================
# Instance-related Radix Sort Scaffolding
# =============================================================================


class CoopBlockRadixSortInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockRadixSortDecl


block_radix_sort_instance_type = CoopBlockRadixSortInstanceType()


@typeof_impl.register(coop.block.radix_sort_keys)
def typeof_block_radix_sort_instance(*args, **kwargs):
    return block_radix_sort_instance_type


@register
class CoopBlockRadixSortInstanceDecl(CoopInstanceTemplate):
    key = block_radix_sort_instance_type
    instance_type = block_radix_sort_instance_type
    primitive_name = "coop.block.radix_sort_keys"


@register_model(CoopBlockRadixSortInstanceType)
class CoopBlockRadixSortInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockRadixSortInstanceType)
def lower_constant_block_radix_sort_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopBlockRadixSortDescendingInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockRadixSortDescendingDecl


block_radix_sort_descending_instance_type = CoopBlockRadixSortDescendingInstanceType()


@typeof_impl.register(coop.block.radix_sort_keys_descending)
def typeof_block_radix_sort_descending_instance(*args, **kwargs):
    return block_radix_sort_descending_instance_type


@register
class CoopBlockRadixSortDescendingInstanceDecl(CoopInstanceTemplate):
    key = block_radix_sort_descending_instance_type
    instance_type = block_radix_sort_descending_instance_type
    primitive_name = "coop.block.radix_sort_keys_descending"


@register_model(CoopBlockRadixSortDescendingInstanceType)
class CoopBlockRadixSortDescendingInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockRadixSortDescendingInstanceType)
def lower_constant_block_radix_sort_descending_instance_type(
    context, builder, typ, value
):
    return context.get_dummy_value()
