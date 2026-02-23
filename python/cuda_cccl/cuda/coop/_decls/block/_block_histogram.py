# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from typing import Optional, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import AttributeTemplate, signature
from numba.extending import lower_builtin, models, register_model, typeof_impl

import cuda.coop as coop

from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopInstanceTypeMixin,
    TempStorageType,
    process_algorithm,
    register,
    register_attr,
    validate_temp_storage,
)


# =============================================================================
# Histogram
# =============================================================================
@register
class CoopBlockHistogramInitDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.histogram.init
    primitive_name = "coop.block.histogram.init"
    minimum_num_args = 0

    @staticmethod
    def signature(histogram: types.Array = None):
        return inspect.signature(
            CoopBlockHistogramInitDecl.signature,
        ).bind(histogram=histogram)

    @staticmethod
    def get_instance_type():
        return block_histogram_instance_type

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        histogram = bound.arguments.get("histogram")
        if histogram is None:
            return signature(types.void)
        if not isinstance(histogram, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'histogram' to be a device array, "
                f"got {type(histogram).__name__}"
            )
        return signature(types.void, histogram)


@register
class CoopBlockHistogramCompositeDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.histogram.composite
    primitive_name = "coop.block.histogram.composite"
    minimum_num_args = 1

    @staticmethod
    def signature(items: types.Array, histogram: types.Array = None):
        return inspect.signature(
            CoopBlockHistogramCompositeDecl.signature,
        ).bind(items, histogram=histogram)

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        if not isinstance(items, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device array, "
                f"got {type(items).__name__}"
            )

        histogram = bound.arguments.get("histogram")
        if histogram is None:
            return signature(types.void, items)
        if not isinstance(histogram, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'histogram' to be a device array, "
                f"got {type(histogram).__name__}"
            )
        sig = signature(types.void, items, histogram)

        return sig


@register
class CoopBlockHistogramDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.histogram
    primitive_name = "coop.block.histogram"
    algorithm_enum = coop.BlockHistogramAlgorithm
    default_algorithm = coop.BlockHistogramAlgorithm.ATOMIC
    minimum_num_args = 1

    @staticmethod
    def get_instance_type():
        return block_histogram_instance_type

    @classmethod
    def signature(
        cls: type,
        items: types.Array,
        histogram: types.Array,
        algorithm: Optional[coop.BlockHistogramAlgorithm] = None,
        temp_storage: Optional[Union[types.Array, TempStorageType]] = None,
    ):
        return inspect.signature(cls.signature).bind(
            items,
            histogram,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_two_phase(
        item_dtype: types.Type,
        counter_dtype: types.Type,
        items_per_thread: Union[types.Integer, types.IntegerLiteral],
        bins: Union[types.Integer, types.IntegerLiteral],
        algorithm: Optional[coop.BlockHistogramAlgorithm] = None,
        temp_storage: Optional[Union[types.Array, TempStorageType]] = None,
    ):
        return inspect.signature(
            CoopBlockHistogramDecl.signature,
        ).bind(
            item_dtype,
            counter_dtype,
            items_per_thread,
            bins,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        if not isinstance(items, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device array, "
                f"got {type(items).__name__}"
            )

        histogram = bound.arguments["histogram"]
        if not isinstance(histogram, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'histogram' to be a device array, "
                f"got {type(histogram).__name__}"
            )

        arglist = [
            items,
            histogram,
        ]

        # Validate algorithm next.  If it's of type ATOMIC, we need to ensure
        # the counter_dtype is a 32-bit or 64-bit integer, as other types
        # won't compile.
        algorithm = bound.arguments.get("algorithm")
        algorithm_is_none_type = isinstance(algorithm, types.NoneType)
        if algorithm_is_none_type:
            arglist.append(algorithm)
            algorithm = None
        if not algorithm_is_none_type:
            if algorithm is None and two_phase:
                # Use the algorithm baked into the two-phase instance.
                pass
            else:
                algorithm = process_algorithm(self, bound, arglist)
                if algorithm == coop.BlockHistogramAlgorithm.ATOMIC:
                    valid_atomic_dtypes = (
                        types.int32,
                        types.int64,
                        types.uint32,
                        types.uint64,
                    )
                    if histogram.dtype not in valid_atomic_dtypes:
                        raise errors.TypingError(
                            "histogram array type must be a 32-bit or 64-bit integer "
                            f"when using the ATOMIC algorithm: got: {histogram.dtype}"
                        )

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            arglist.append(temp_storage)
            temp_storage = None
        if not temp_storage_is_none_type:
            validate_temp_storage(self, temp_storage)
            if temp_storage is not None:
                arglist.append(temp_storage)

        sig = signature(
            block_histogram_instance_type,
            *arglist,
        )

        return sig


# =============================================================================
# Instance-related Histogram Scaffolding
# =============================================================================
class CoopBlockHistogramInstanceType(types.Type, CoopInstanceTypeMixin):
    """
    This type represents an instance of a cooperative block histogram.
    It is used to create a two-phase cooperative block histogram instance.
    """

    decl_class = CoopBlockHistogramDecl

    def __init__(self):
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(self, *args, **kwds):
        if not args and not kwds:
            return signature(block_histogram_instance_type)

        if "items" in kwds or "histogram" in kwds or len(args) >= 2:
            bound = self._bind_instance_signature(*args, **kwds)
            return self.decl._validate_args_and_create_signature(bound, two_phase=True)

        if kwds and set(kwds) - {"temp_storage"}:
            raise errors.TypingError(
                f"{self.decl.primitive_name} only supports 'temp_storage' "
                "without items/histogram arguments"
            )
        if len(args) > 1:
            raise errors.TypingError(
                f"{self.decl.primitive_name} accepts at most one positional argument "
                "when no items/histogram are provided"
            )

        temp_storage = args[0] if args else kwds.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if not temp_storage_is_none_type:
            validate_temp_storage(self.decl, temp_storage)

        arglist = []
        if temp_storage is not None or temp_storage_is_none_type:
            arglist.append(temp_storage)

        return signature(block_histogram_instance_type, *arglist)


block_histogram_instance_type = CoopBlockHistogramInstanceType()


@register_model(CoopBlockHistogramInstanceType)
class CoopBlockHistogramInstanceModel(models.OpaqueModel):
    pass


@typeof_impl.register(coop.block.histogram)
def typeof_block_histogram_instance(*args, **kwargs):
    return block_histogram_instance_type


@register
class CoopBlockHistogramInstanceDecl(CoopInstanceTemplate):
    key = block_histogram_instance_type
    instance_type = block_histogram_instance_type
    primitive_name = "coop.block.histogram"


class CoopBlockHistogramAttrsTemplate(AttributeTemplate):
    key = block_histogram_instance_type

    def resolve_init(self, instance):
        return types.BoundFunction(CoopBlockHistogramInitDecl, instance)

    def resolve_composite(self, instance):
        return types.BoundFunction(CoopBlockHistogramCompositeDecl, instance)


register_attr(CoopBlockHistogramAttrsTemplate)

block_histogram_attrs_template = CoopBlockHistogramAttrsTemplate(None)


@lower_constant(CoopBlockHistogramInstanceType)
def lower_constant_block_histogram_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


@lower_builtin(CoopBlockHistogramInstanceType, types.VarArg(types.Any))
def codegen_block_histogram(context, builder, sig, args):
    # This isn't actually ever called, but it needs to exist.
    return context.get_dummy_value()


@lower_builtin("call", CoopBlockHistogramInstanceType, types.NoneType)
@lower_builtin("call", CoopBlockHistogramInstanceType, types.Array)
def codegen_block_histogram_call(context, builder, sig, args):
    return context.get_dummy_value()
