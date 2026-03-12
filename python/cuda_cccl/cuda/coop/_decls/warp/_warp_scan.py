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

from ..._scan_op import ScanOp
from ..._typing import ScanOpType
from ...warp._warp_scan import (
    _make_exclusive_scan_rewrite,
    _make_exclusive_sum_rewrite,
    _make_inclusive_scan_rewrite,
    _make_inclusive_sum_rewrite,
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
# Scan/Sum
# =============================================================================
@register_global(coop.warp.inclusive_sum)
class CoopWarpInclusiveSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.inclusive_sum
    impl_key = _make_inclusive_sum_rewrite
    primitive_name = "coop.warp.inclusive_sum"
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: types.Number,
        threads_in_warp: int = 32,
        warp_aggregate: types.Array = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpInclusiveSumDecl.signature).bind(
            src,
            threads_in_warp=threads_in_warp,
            warp_aggregate=warp_aggregate,
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

        warp_aggregate = bound.arguments.get("warp_aggregate")
        warp_aggregate_is_none_type = isinstance(warp_aggregate, types.NoneType)
        if warp_aggregate_is_none_type:
            arglist.append(warp_aggregate)
            warp_aggregate = None
        if not warp_aggregate_is_none_type and warp_aggregate is not None:
            if not isinstance(warp_aggregate, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to be a device array"
                )
            if warp_aggregate.dtype != src:
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to have the same "
                    "dtype as the input"
                )
            arglist.append(warp_aggregate)

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


@register_global(coop.warp.exclusive_sum)
class CoopWarpExclusiveSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.exclusive_sum
    impl_key = _make_exclusive_sum_rewrite
    primitive_name = "coop.warp.exclusive_sum"
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: types.Number,
        threads_in_warp: int = 32,
        warp_aggregate: types.Array = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpExclusiveSumDecl.signature).bind(
            src,
            threads_in_warp=threads_in_warp,
            warp_aggregate=warp_aggregate,
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

        warp_aggregate = bound.arguments.get("warp_aggregate")
        warp_aggregate_is_none_type = isinstance(warp_aggregate, types.NoneType)
        if warp_aggregate_is_none_type:
            arglist.append(warp_aggregate)
            warp_aggregate = None
        if not warp_aggregate_is_none_type and warp_aggregate is not None:
            if not isinstance(warp_aggregate, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to be a device array"
                )
            if warp_aggregate.dtype != src:
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to have the same "
                    "dtype as the input"
                )
            arglist.append(warp_aggregate)

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


@register_global(coop.warp.exclusive_scan)
class CoopWarpExclusiveScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.exclusive_scan
    impl_key = _make_exclusive_scan_rewrite
    primitive_name = "coop.warp.exclusive_scan"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: types.Number,
        scan_op: ScanOpType,
        initial_value: Optional[types.Number] = None,
        threads_in_warp: int = 32,
        valid_items: Optional[int] = None,
        warp_aggregate: types.Array = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpExclusiveScanDecl.signature).bind(
            src,
            scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: types.Number,
        initial_value: Optional[types.Number] = None,
        *,
        scan_op: ScanOpType = None,
        threads_in_warp: int = None,
        valid_items: Optional[int] = None,
        warp_aggregate: types.Array = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpExclusiveScanDecl.signature_instance).bind(
            src,
            initial_value=initial_value,
            scan_op=scan_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        if isinstance(src, types.Array):
            raise errors.TypingError(f"{self.primitive_name} requires a scalar input")
        if not isinstance(src, types.Number):
            raise errors.TypingError(f"{self.primitive_name} requires a numeric input")
        arglist = [src]

        scan_op = bound.arguments.get("scan_op")
        scan_op_is_none_type = isinstance(scan_op, types.NoneType)
        if scan_op is None or scan_op_is_none_type:
            if not two_phase:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'scan_op' to be specified"
                )
            scan_op = None
        if scan_op is not None:
            if isinstance(scan_op, types.StringLiteral):
                scan_op = scan_op.literal_value
            try:
                scan_op = ScanOp(scan_op)
            except ValueError as e:
                raise errors.TypingError(
                    f"Invalid scan_op '{scan_op}' for {self.primitive_name}: {e}"
                )
            arglist.append(scan_op)

        initial_value = bound.arguments.get("initial_value")
        if isinstance(initial_value, types.NoneType):
            arglist.append(initial_value)
            initial_value = None
        if isinstance(initial_value, types.IntegerLiteral):
            initial_value = initial_value.literal_value
        if initial_value is not None:
            arglist.append(initial_value)

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

        valid_items = bound.arguments.get("valid_items")
        valid_items_is_none_type = isinstance(valid_items, types.NoneType)
        if valid_items_is_none_type:
            arglist.append(valid_items)
            valid_items = None
        if not valid_items_is_none_type and valid_items is not None:
            if not isinstance(valid_items, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            arglist.append(valid_items)

        warp_aggregate = bound.arguments.get("warp_aggregate")
        warp_aggregate_is_none_type = isinstance(warp_aggregate, types.NoneType)
        if warp_aggregate_is_none_type:
            arglist.append(warp_aggregate)
            warp_aggregate = None
        if not warp_aggregate_is_none_type and warp_aggregate is not None:
            if not isinstance(warp_aggregate, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to be a device array"
                )
            if warp_aggregate.dtype != src:
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to have the same "
                    "dtype as the input"
                )
            arglist.append(warp_aggregate)

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


@register_global(coop.warp.inclusive_scan)
class CoopWarpInclusiveScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.inclusive_scan
    impl_key = _make_inclusive_scan_rewrite
    primitive_name = "coop.warp.inclusive_scan"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: types.Number,
        scan_op: ScanOpType,
        initial_value: Optional[types.Number] = None,
        threads_in_warp: int = 32,
        valid_items: Optional[int] = None,
        warp_aggregate: types.Array = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpInclusiveScanDecl.signature).bind(
            src,
            scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: types.Number,
        initial_value: Optional[types.Number] = None,
        *,
        scan_op: ScanOpType = None,
        threads_in_warp: int = None,
        valid_items: Optional[int] = None,
        warp_aggregate: types.Array = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpInclusiveScanDecl.signature_instance).bind(
            src,
            initial_value=initial_value,
            scan_op=scan_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        if isinstance(src, types.Array):
            raise errors.TypingError(f"{self.primitive_name} requires a scalar input")
        if not isinstance(src, types.Number):
            raise errors.TypingError(f"{self.primitive_name} requires a numeric input")
        arglist = [src]

        scan_op = bound.arguments.get("scan_op")
        scan_op_is_none_type = isinstance(scan_op, types.NoneType)
        if scan_op is None or scan_op_is_none_type:
            if not two_phase:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'scan_op' to be specified"
                )
            scan_op = None
        if scan_op is not None:
            if isinstance(scan_op, types.StringLiteral):
                scan_op = scan_op.literal_value
            try:
                scan_op = ScanOp(scan_op)
            except ValueError as e:
                raise errors.TypingError(
                    f"Invalid scan_op '{scan_op}' for {self.primitive_name}: {e}"
                )
            arglist.append(scan_op)

        initial_value = bound.arguments.get("initial_value")
        if isinstance(initial_value, types.NoneType):
            arglist.append(initial_value)
            initial_value = None
        if isinstance(initial_value, types.IntegerLiteral):
            initial_value = initial_value.literal_value
        if initial_value is not None:
            arglist.append(initial_value)

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

        valid_items = bound.arguments.get("valid_items")
        valid_items_is_none_type = isinstance(valid_items, types.NoneType)
        if valid_items_is_none_type:
            arglist.append(valid_items)
            valid_items = None
        if not valid_items_is_none_type and valid_items is not None:
            if not isinstance(valid_items, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            arglist.append(valid_items)

        warp_aggregate = bound.arguments.get("warp_aggregate")
        warp_aggregate_is_none_type = isinstance(warp_aggregate, types.NoneType)
        if warp_aggregate_is_none_type:
            arglist.append(warp_aggregate)
            warp_aggregate = None
        if not warp_aggregate_is_none_type and warp_aggregate is not None:
            if not isinstance(warp_aggregate, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to be a device array"
                )
            if warp_aggregate.dtype != src:
                raise errors.TypingError(
                    f"{self.primitive_name} requires warp_aggregate to have the same "
                    "dtype as the input"
                )
            arglist.append(warp_aggregate)

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


# =============================================================================
# Instance-related Scan/Sum Scaffolding
# =============================================================================
class CoopWarpInclusiveSumInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpInclusiveSumDecl


warp_inclusive_sum_instance_type = CoopWarpInclusiveSumInstanceType()


@typeof_impl.register(coop.warp.inclusive_sum)
def typeof_warp_inclusive_sum_instance(*args, **kwargs):
    return warp_inclusive_sum_instance_type


@register
class CoopWarpInclusiveSumInstanceDecl(CoopInstanceTemplate):
    key = warp_inclusive_sum_instance_type
    instance_type = warp_inclusive_sum_instance_type
    primitive_name = "coop.warp.inclusive_sum"


@register_model(CoopWarpInclusiveSumInstanceType)
class CoopWarpInclusiveSumInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpInclusiveSumInstanceType)
def lower_constant_warp_inclusive_sum_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpExclusiveSumInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpExclusiveSumDecl


warp_exclusive_sum_instance_type = CoopWarpExclusiveSumInstanceType()


@typeof_impl.register(coop.warp.exclusive_sum)
def typeof_warp_exclusive_sum_instance(*args, **kwargs):
    return warp_exclusive_sum_instance_type


@register
class CoopWarpExclusiveSumInstanceDecl(CoopInstanceTemplate):
    key = warp_exclusive_sum_instance_type
    instance_type = warp_exclusive_sum_instance_type
    primitive_name = "coop.warp.exclusive_sum"


@register_model(CoopWarpExclusiveSumInstanceType)
class CoopWarpExclusiveSumInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpExclusiveSumInstanceType)
def lower_constant_warp_exclusive_sum_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpExclusiveScanInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpExclusiveScanDecl


warp_exclusive_scan_instance_type = CoopWarpExclusiveScanInstanceType()


@typeof_impl.register(coop.warp.exclusive_scan)
def typeof_warp_exclusive_scan_instance(*args, **kwargs):
    return warp_exclusive_scan_instance_type


@register
class CoopWarpExclusiveScanInstanceDecl(CoopInstanceTemplate):
    key = warp_exclusive_scan_instance_type
    instance_type = warp_exclusive_scan_instance_type
    primitive_name = "coop.warp.exclusive_scan"


@register_model(CoopWarpExclusiveScanInstanceType)
class CoopWarpExclusiveScanInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpExclusiveScanInstanceType)
def lower_constant_warp_exclusive_scan_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpInclusiveScanInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpInclusiveScanDecl


warp_inclusive_scan_instance_type = CoopWarpInclusiveScanInstanceType()


@typeof_impl.register(coop.warp.inclusive_scan)
def typeof_warp_inclusive_scan_instance(*args, **kwargs):
    return warp_inclusive_scan_instance_type


@register
class CoopWarpInclusiveScanInstanceDecl(CoopInstanceTemplate):
    key = warp_inclusive_scan_instance_type
    instance_type = warp_inclusive_scan_instance_type
    primitive_name = "coop.warp.inclusive_scan"


@register_model(CoopWarpInclusiveScanInstanceType)
class CoopWarpInclusiveScanInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpInclusiveScanInstanceType)
def lower_constant_warp_inclusive_scan_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
