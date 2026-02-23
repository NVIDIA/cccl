# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from typing import Any, Callable, Optional, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import signature
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ...block._block_merge_sort import (
    _make_merge_sort_keys_rewrite as _make_block_merge_sort_keys_rewrite,
)
from ...block._block_merge_sort import (
    _make_merge_sort_pairs_rewrite as _make_block_merge_sort_pairs_rewrite,
)
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopSimpleInstanceType,
    TempStorageType,
    ThreadDataType,
    process_items_per_thread,
    register,
    register_global,
    validate_temp_storage,
)


# =============================================================================
# Merge Sort
# =============================================================================
@register_global(coop.block.merge_sort_keys)
class CoopBlockMergeSortDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.merge_sort_keys
    impl_key = _make_block_merge_sort_keys_rewrite
    primitive_name = "coop.block.merge_sort_keys"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        keys: types.Array,
        items_per_thread: int = None,
        compare_op: Optional[Callable] = None,
        values: types.Array = None,
        valid_items: Optional[int] = None,
        oob_default: Optional[Any] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockMergeSortDecl.signature).bind(
            keys,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            values=values,
            valid_items=valid_items,
            oob_default=oob_default,
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

        compare_op = bound.arguments.get("compare_op")
        compare_op_is_none_type = isinstance(compare_op, types.NoneType)
        if compare_op is None or compare_op_is_none_type:
            if not two_phase:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'compare_op' to be specified"
                )
        else:
            arglist.append(compare_op)

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

        valid_items = bound.arguments.get("valid_items")
        oob_default = bound.arguments.get("oob_default")
        if (valid_items is None) != (oob_default is None):
            raise errors.TypingError(
                f"{self.primitive_name} requires valid_items and oob_default together"
            )
        if valid_items is not None:
            if not isinstance(valid_items, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            if isinstance(oob_default, types.NoneType):
                oob_default = None
            if oob_default is None:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'oob_default' when valid_items is provided"
                )
            arglist.append(valid_items)
            arglist.append(oob_default)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


@register_global(coop.block.merge_sort_pairs)
class CoopBlockMergeSortPairsDecl(CoopBlockMergeSortDecl):
    key = coop.block.merge_sort_pairs
    impl_key = _make_block_merge_sort_pairs_rewrite
    primitive_name = "coop.block.merge_sort_pairs"
    minimum_num_args = 3

    @staticmethod
    def signature(
        keys: types.Array,
        values: types.Array,
        items_per_thread: int = None,
        compare_op: Optional[Callable] = None,
        valid_items: Optional[int] = None,
        oob_default: Optional[Any] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockMergeSortPairsDecl.signature).bind(
            keys,
            values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            valid_items=valid_items,
            oob_default=oob_default,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        keys = bound.arguments["keys"]
        if not isinstance(keys, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'keys' to be a device array"
            )

        values = bound.arguments.get("values")
        values_is_array = isinstance(values, (types.Array, ThreadDataType))
        if not values_is_array and ThreadDataType is not None:
            try:
                values = ThreadDataType.from_array(values)
                values_is_array = True
            except Exception:
                values_is_array = False

        if not values_is_array:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'values' to be a device array"
            )

        arglist = [keys, values]
        process_items_per_thread(self, bound, arglist, two_phase, target_array=keys)

        compare_op = bound.arguments.get("compare_op")
        compare_op_is_none_type = isinstance(compare_op, types.NoneType)
        if compare_op is None or compare_op_is_none_type:
            if not two_phase:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'compare_op' to be specified"
                )
        else:
            arglist.append(compare_op)

        valid_items = bound.arguments.get("valid_items")
        oob_default = bound.arguments.get("oob_default")
        if (valid_items is None) != (oob_default is None):
            raise errors.TypingError(
                f"{self.primitive_name} requires valid_items and oob_default together"
            )
        if valid_items is not None:
            if not isinstance(valid_items, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            if isinstance(oob_default, types.NoneType):
                oob_default = None
            if oob_default is None:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'oob_default' when valid_items is provided"
                )
            arglist.append(valid_items)
            arglist.append(oob_default)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


# =============================================================================
# Instance-related Merge Sort Scaffolding
# =============================================================================


class CoopBlockMergeSortInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockMergeSortDecl


block_merge_sort_instance_type = CoopBlockMergeSortInstanceType()


@typeof_impl.register(coop.block.merge_sort_keys)
def typeof_block_merge_sort_instance(*args, **kwargs):
    return block_merge_sort_instance_type


@register
class CoopBlockMergeSortInstanceDecl(CoopInstanceTemplate):
    key = block_merge_sort_instance_type
    instance_type = block_merge_sort_instance_type
    primitive_name = "coop.block.merge_sort_keys"


@register_model(CoopBlockMergeSortInstanceType)
class CoopBlockMergeSortInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockMergeSortInstanceType)
def lower_constant_block_merge_sort_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopBlockMergeSortPairsInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockMergeSortPairsDecl


block_merge_sort_pairs_instance_type = CoopBlockMergeSortPairsInstanceType()


@typeof_impl.register(coop.block.merge_sort_pairs)
def typeof_block_merge_sort_pairs_instance(*args, **kwargs):
    return block_merge_sort_pairs_instance_type


@register
class CoopBlockMergeSortPairsInstanceDecl(CoopInstanceTemplate):
    key = block_merge_sort_pairs_instance_type
    instance_type = block_merge_sort_pairs_instance_type
    primitive_name = "coop.block.merge_sort_pairs"


@register_model(CoopBlockMergeSortPairsInstanceType)
class CoopBlockMergeSortPairsInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockMergeSortPairsInstanceType)
def lower_constant_block_merge_sort_pairs_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
