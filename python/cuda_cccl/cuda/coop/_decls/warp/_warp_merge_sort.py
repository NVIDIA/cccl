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

from ...warp._warp_merge_sort import (
    _make_merge_sort_keys_rewrite,
    _make_merge_sort_pairs_rewrite,
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
    validate_threads_in_warp,
)


# =============================================================================
# Merge Sort
# =============================================================================
@register_global(coop.warp.merge_sort_keys)
class CoopWarpMergeSortDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.merge_sort_keys
    impl_key = _make_merge_sort_keys_rewrite
    primitive_name = "coop.warp.merge_sort_keys"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        keys: types.Array,
        items_per_thread: int = None,
        compare_op: Optional[Callable] = None,
        threads_in_warp: int = 32,
        values: types.Array = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpMergeSortDecl.signature).bind(
            keys,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
            values=values,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        keys: types.Array,
        values: types.Array = None,
        *,
        items_per_thread: int = None,
        compare_op: Optional[Callable] = None,
        threads_in_warp: int = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpMergeSortDecl.signature_instance).bind(
            keys,
            values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
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

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

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

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


@register_global(coop.warp.merge_sort_pairs)
class CoopWarpMergeSortPairsDecl(CoopWarpMergeSortDecl):
    key = coop.warp.merge_sort_pairs
    impl_key = _make_merge_sort_pairs_rewrite
    primitive_name = "coop.warp.merge_sort_pairs"

    @staticmethod
    def signature(
        keys: types.Array,
        values: types.Array,
        items_per_thread: int = None,
        compare_op: Optional[Callable] = None,
        threads_in_warp: int = 32,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpMergeSortPairsDecl.signature).bind(
            keys,
            values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        keys: types.Array,
        values: types.Array,
        *,
        items_per_thread: int = None,
        compare_op: Optional[Callable] = None,
        threads_in_warp: int = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpMergeSortPairsDecl.signature_instance).bind(
            keys,
            values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
            temp_storage=temp_storage,
        )


# =============================================================================
# Instance-related Merge Sort Scaffolding
# =============================================================================
class CoopWarpMergeSortInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpMergeSortDecl


warp_merge_sort_instance_type = CoopWarpMergeSortInstanceType()


@typeof_impl.register(coop.warp.merge_sort_keys)
def typeof_warp_merge_sort_instance(*args, **kwargs):
    return warp_merge_sort_instance_type


@register
class CoopWarpMergeSortInstanceDecl(CoopInstanceTemplate):
    key = warp_merge_sort_instance_type
    instance_type = warp_merge_sort_instance_type
    primitive_name = "coop.warp.merge_sort_keys"


@register_model(CoopWarpMergeSortInstanceType)
class CoopWarpMergeSortInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpMergeSortInstanceType)
def lower_constant_warp_merge_sort_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpMergeSortPairsInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpMergeSortPairsDecl


warp_merge_sort_pairs_instance_type = CoopWarpMergeSortPairsInstanceType()


@typeof_impl.register(coop.warp.merge_sort_pairs)
def typeof_warp_merge_sort_pairs_instance(*args, **kwargs):
    return warp_merge_sort_pairs_instance_type


@register
class CoopWarpMergeSortPairsInstanceDecl(CoopInstanceTemplate):
    key = warp_merge_sort_pairs_instance_type
    instance_type = warp_merge_sort_pairs_instance_type
    primitive_name = "coop.warp.merge_sort_pairs"


@register_model(CoopWarpMergeSortPairsInstanceType)
class CoopWarpMergeSortPairsInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpMergeSortPairsInstanceType)
def lower_constant_warp_merge_sort_pairs_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
