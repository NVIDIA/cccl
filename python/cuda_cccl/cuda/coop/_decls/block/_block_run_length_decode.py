# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from typing import Any, Optional, Union

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
    register,
    register_attr,
    validate_positive_integer_literal,
    validate_temp_storage,
)

# =============================================================================
# RunLengthDecode
# =============================================================================

# N.B. Because `RunLengthDecodeRunLengthDecode` is an awfully-ugly name in
#      Python, we use `RunLength` to represent the `cub::BlockRunLengthDecode`
#      `cub::BlockRunLengthDecode` primitive, and simply `Decode` to represent
#      the `cub::BlockRunLengthDecode::RunLengthDecode` method instance.


@register
class CoopBlockRunLengthDecodeDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.run_length.decode
    primitive_name = "coop.block.run_length.decode"
    minimum_num_args = 2

    @staticmethod
    def signature(
        decoded_items: types.Array,
        decoded_window_offset: types.Integer,
        relative_offsets: types.Array = None,
    ):
        return inspect.signature(CoopBlockRunLengthDecodeDecl.signature).bind(
            decoded_items,
            decoded_window_offset,
            relative_offsets=relative_offsets,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        decoded_items = bound.arguments["decoded_items"]
        if not isinstance(decoded_items, types.Array):
            raise errors.TypingError(
                "decoded_items must be a device array, "
                f"got {type(decoded_items).__name__}"
            )

        arglist = [
            decoded_items,
        ]

        decoded_window_offset = bound.arguments.get("decoded_window_offset")
        decoded_window_offset_is_none_type = isinstance(
            decoded_window_offset, types.NoneType
        )
        if decoded_window_offset_is_none_type:
            arglist.append(decoded_window_offset)
            decoded_window_offset = None
        if not decoded_window_offset_is_none_type and decoded_window_offset is not None:
            if not isinstance(decoded_window_offset, types.Integer):
                raise errors.TypingError(
                    "decoded_window_offset must be an integer value"
                )
            arglist.append(decoded_window_offset)

        relative_offsets = bound.arguments.get("relative_offsets")
        relative_offsets_is_none_type = isinstance(relative_offsets, types.NoneType)
        if relative_offsets_is_none_type:
            arglist.append(relative_offsets)
            relative_offsets = None
        if not relative_offsets_is_none_type and relative_offsets is not None:
            if not isinstance(relative_offsets, types.Array):
                raise errors.TypingError(
                    "relative_offsets must be a device array, "
                    f"got {type(relative_offsets).__name__}"
                )
            arglist.append(relative_offsets)

        sig = signature(
            types.void,
            *arglist,
        )

        return sig


@register
class CoopBlockRunLengthDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.run_length
    primitive_name = "coop.block.run_length"
    algorithm_enum = coop.NoAlgorithm
    default_algorithm = coop.NoAlgorithm.NO_ALGORITHM
    decode_decl = CoopBlockRunLengthDecodeDecl
    is_constructor = True
    minimum_num_args = 5

    exact_match_required = True
    prefer_literal = True

    def __init__(self, context=None):
        super().__init__(context=context)

    @staticmethod
    def get_instance_type():
        return block_run_length_instance_type

    @staticmethod
    def signature(
        run_values: types.Array,
        run_lengths: types.Array,
        runs_per_thread: Union[types.Integer, types.IntegerLiteral],
        decoded_items_per_thread: Union[types.Integer, types.IntegerLiteral],
        total_decoded_size: types.Array,
        decoded_offset_dtype: Optional[Any] = None,
        temp_storage: Optional[Union[types.Array, TempStorageType]] = None,
    ):
        return inspect.signature(CoopBlockRunLengthDecl.signature).bind(
            run_values,
            run_lengths,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=decoded_offset_dtype,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        # error_class = errors.TypingError
        error_class = RuntimeError

        run_values = bound.arguments["run_values"]
        run_lengths = bound.arguments["run_lengths"]

        # Verify run_values and run_lengths are device arrays.
        if not isinstance(run_values, types.Array):
            raise error_class(
                f"run_values must be a device array, got {type(run_values).__name__}"
            )

        if not isinstance(run_lengths, types.Array):
            raise error_class(
                f"run_lengths must be a device array, got {type(run_lengths).__name__}"
            )

        runs_per_thread = bound.arguments.get("runs_per_thread")
        runs_per_thread_is_none_type = isinstance(runs_per_thread, types.NoneType)
        if runs_per_thread_is_none_type:
            runs_per_thread = None
        if not (two_phase and runs_per_thread is None):
            validate_positive_integer_literal(
                self,
                runs_per_thread,
                "runs_per_thread",
            )

        decoded_items_per_thread = bound.arguments.get("decoded_items_per_thread")
        decoded_items_is_none_type = isinstance(
            decoded_items_per_thread, types.NoneType
        )
        if decoded_items_is_none_type:
            decoded_items_per_thread = None
        if not (two_phase and decoded_items_per_thread is None):
            validate_positive_integer_literal(
                self,
                decoded_items_per_thread,
                "decoded_items_per_thread",
            )

        decoded_offset_dtype = bound.arguments.get("decoded_offset_dtype")
        decoded_offset_is_none_type = isinstance(decoded_offset_dtype, types.NoneType)
        if decoded_offset_is_none_type:
            decoded_offset_dtype = None
        if decoded_offset_dtype is not None:
            from ..._common import normalize_dtype_param

            decoded_offset_dtype = normalize_dtype_param(decoded_offset_dtype)

        total_decoded_size = bound.arguments.get("total_decoded_size")
        if total_decoded_size is None:
            raise error_class("total_decoded_size must be a device array")
        if not isinstance(total_decoded_size, types.Array):
            raise error_class(
                "total_decoded_size must be a device array, "
                f"got {type(total_decoded_size).__name__}"
            )
        if total_decoded_size.ndim != 1:
            raise error_class(
                "total_decoded_size must be a 1D device array, "
                f"got ndim={total_decoded_size.ndim}"
            )
        if not isinstance(total_decoded_size.dtype, types.Integer):
            raise error_class("total_decoded_size array must use an integer dtype")

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            temp_storage = None
        validate_temp_storage(self, temp_storage)

        arglist = [
            run_values,
            run_lengths,
        ]

        if runs_per_thread is not None or runs_per_thread_is_none_type:
            arglist.append(runs_per_thread)

        if decoded_items_per_thread is not None or decoded_items_is_none_type:
            arglist.append(decoded_items_per_thread)

        arglist.append(total_decoded_size)

        if decoded_offset_dtype is not None or decoded_offset_is_none_type:
            arglist.append(decoded_offset_dtype)

        if temp_storage is not None:
            arglist.append(temp_storage)

        sig = signature(
            block_run_length_instance_type,
            *arglist,
        )

        return sig


# =============================================================================
# Instance-related RunLength Scaffolding
# =============================================================================


class CoopBlockRunLengthInstanceType(types.Type, CoopInstanceTypeMixin):
    """
    This type represents an instance of a cooperative block run_length.
    It is used to create a two-phase cooperative block run_length instance.
    """

    decl_class = CoopBlockRunLengthDecl

    def __init__(self):
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(self, *args, **kwds):
        if not args and not kwds:
            return signature(block_run_length_instance_type)

        bound = self._bind_instance_signature(*args, **kwds)
        return self.decl._validate_args_and_create_signature(bound, two_phase=True)


block_run_length_instance_type = CoopBlockRunLengthInstanceType()


@register_model(CoopBlockRunLengthInstanceType)
class CoopBlockRunLengthInstanceModel(models.OpaqueModel):
    pass


@typeof_impl.register(coop.block.run_length)
def typeof_block_run_length_instance(*args, **kwargs):
    return block_run_length_instance_type


@register
class CoopBlockRunLengthInstanceDecl(CoopInstanceTemplate):
    key = block_run_length_instance_type
    instance_type = block_run_length_instance_type
    primitive_name = "coop.block.run_length"


class CoopBlockRunLengthAttrsTemplate(AttributeTemplate):
    key = block_run_length_instance_type

    def resolve_decode(self, instance):
        return types.BoundFunction(CoopBlockRunLengthDecodeDecl, instance)


register_attr(CoopBlockRunLengthAttrsTemplate)

block_run_length_attrs_template = CoopBlockRunLengthAttrsTemplate(None)


@lower_constant(CoopBlockRunLengthInstanceType)
def lower_constant_block_run_length_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


@lower_builtin(CoopBlockRunLengthInstanceType, types.VarArg(types.Any))
def codegen_block_run_length(context, builder, sig, args):
    # This isn't actually ever called, but it needs to exist.
    return context.get_dummy_value()


@lower_builtin("call", CoopBlockRunLengthInstanceType, types.VarArg(types.Any))
def codegen_block_run_length_call(context, builder, sig, args):
    return context.get_dummy_value()
