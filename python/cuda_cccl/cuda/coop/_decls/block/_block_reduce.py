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

from ..._common import CUB_BLOCK_REDUCE_ALGOS
from ...block._block_reduce import _make_reduce_rewrite as _make_block_reduce_rewrite
from ...block._block_reduce import _make_sum_rewrite as _make_block_sum_rewrite
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopInstanceTypeMixin,
    TempStorageType,
    process_items_per_thread,
    register,
    validate_temp_storage,
)


# =============================================================================
# Reduce
# =============================================================================
class CoopBlockReduceDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.reduce
    impl_key = _make_block_reduce_rewrite
    primitive_name = "coop.block.reduce"
    is_constructor = False
    minimum_num_args = 1
    default_algorithm = "warp_reductions"

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        items_per_thread: int = None,
        binary_op: Optional[Callable] = None,
        num_valid: Optional[int] = None,
        algorithm: Optional[str] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockReduceDecl.signature).bind(
            src,
            items_per_thread=items_per_thread,
            binary_op=binary_op,
            num_valid=num_valid,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: Union[types.Array, types.Number],
        num_valid: Optional[int] = None,
        *,
        items_per_thread: int = None,
        binary_op: Optional[Callable] = None,
        algorithm: Optional[str] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockReduceDecl.signature_instance).bind(
            src,
            num_valid=num_valid,
            items_per_thread=items_per_thread,
            binary_op=binary_op,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def get_instance_type():
        return block_reduce_instance_type

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        if not isinstance(src, (types.Array, types.Type)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' to be an array or scalar"
            )

        return_type = src.dtype if isinstance(src, types.Array) else src
        arglist = [src]

        process_items_per_thread(
            self,
            bound,
            arglist,
            two_phase,
            target_array=src if isinstance(src, types.Array) else None,
        )

        binary_op = bound.arguments.get("binary_op")
        binary_op_is_none_type = isinstance(binary_op, types.NoneType)
        if binary_op is None or binary_op_is_none_type:
            if not two_phase:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'binary_op' to be specified"
                )
        else:
            arglist.append(binary_op)

        num_valid = bound.arguments.get("num_valid")
        if num_valid is not None:
            if isinstance(src, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} does not support 'num_valid' for array inputs"
                )
            if not isinstance(num_valid, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'num_valid' to be an integer"
                )
            arglist.append(num_valid)

        algorithm = bound.arguments.get("algorithm")
        algorithm_is_none_type = isinstance(algorithm, types.NoneType)
        if algorithm is None or algorithm_is_none_type:
            if not two_phase:
                algorithm = self.default_algorithm
            else:
                algorithm = None
        if algorithm is not None:
            if isinstance(algorithm, types.StringLiteral):
                algorithm = algorithm.literal_value
            if algorithm not in CUB_BLOCK_REDUCE_ALGOS:
                raise errors.TypingError(
                    f"Invalid algorithm '{algorithm}' for {self.primitive_name}"
                )
            arglist.append(algorithm)

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            arglist.append(temp_storage)
            temp_storage = None
        if not temp_storage_is_none_type:
            validate_temp_storage(self, temp_storage)
            if temp_storage is not None:
                arglist.append(temp_storage)

        sig = signature(return_type, *arglist)

        return sig


class CoopBlockReduceInstanceType(types.Type, CoopInstanceTypeMixin):
    decl_class = CoopBlockReduceDecl

    def __init__(self):
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(self, *args, **kwargs):
        bound = self._bind_instance_signature(*args, **kwargs)
        return self.decl._validate_args_and_create_signature(bound, two_phase=True)


block_reduce_instance_type = CoopBlockReduceInstanceType()


@typeof_impl.register(coop.block.reduce)
def typeof_block_reduce_instance(*args, **kwargs):
    return block_reduce_instance_type


@register
class CoopBlockReduceInstanceDecl(CoopInstanceTemplate):
    key = block_reduce_instance_type
    instance_type = block_reduce_instance_type
    primitive_name = "coop.block.reduce"


@register_model(CoopBlockReduceInstanceType)
class CoopBlockReduceInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockReduceInstanceType)
def lower_constant_block_reduce_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


@register
class CoopBlockSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.sum
    impl_key = _make_block_sum_rewrite
    primitive_name = "coop.block.sum"
    is_constructor = False
    minimum_num_args = 1
    default_algorithm = "warp_reductions"

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        items_per_thread: int = None,
        num_valid: Optional[int] = None,
        algorithm: Optional[str] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockSumDecl.signature).bind(
            src,
            items_per_thread=items_per_thread,
            num_valid=num_valid,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: Union[types.Array, types.Number],
        num_valid: Optional[int] = None,
        *,
        items_per_thread: int = None,
        algorithm: Optional[str] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockSumDecl.signature_instance).bind(
            src,
            num_valid=num_valid,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def get_instance_type():
        return block_sum_instance_type

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        if not isinstance(src, (types.Array, types.Type)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' to be an array or scalar"
            )

        return_type = src.dtype if isinstance(src, types.Array) else src
        arglist = [src]

        process_items_per_thread(
            self,
            bound,
            arglist,
            two_phase,
            target_array=src if isinstance(src, types.Array) else None,
        )

        num_valid = bound.arguments.get("num_valid")
        if num_valid is not None:
            if isinstance(src, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} does not support 'num_valid' for array inputs"
                )
            if not isinstance(num_valid, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'num_valid' to be an integer"
                )
            arglist.append(num_valid)

        algorithm = bound.arguments.get("algorithm")
        algorithm_is_none_type = isinstance(algorithm, types.NoneType)
        if algorithm is None or algorithm_is_none_type:
            if not two_phase:
                algorithm = self.default_algorithm
            else:
                algorithm = None
        if algorithm is not None:
            if isinstance(algorithm, types.StringLiteral):
                algorithm = algorithm.literal_value
            if algorithm not in CUB_BLOCK_REDUCE_ALGOS:
                raise errors.TypingError(
                    f"Invalid algorithm '{algorithm}' for {self.primitive_name}"
                )
            arglist.append(algorithm)

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            arglist.append(temp_storage)
            temp_storage = None
        if not temp_storage_is_none_type:
            validate_temp_storage(self, temp_storage)
            if temp_storage is not None:
                arglist.append(temp_storage)

        sig = signature(return_type, *arglist)

        return sig


class CoopBlockSumInstanceType(types.Type, CoopInstanceTypeMixin):
    decl_class = CoopBlockSumDecl

    def __init__(self):
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(self, *args, **kwargs):
        bound = self._bind_instance_signature(*args, **kwargs)
        return self.decl._validate_args_and_create_signature(bound, two_phase=True)


block_sum_instance_type = CoopBlockSumInstanceType()


@typeof_impl.register(coop.block.sum)
def typeof_block_sum_instance(*args, **kwargs):
    return block_sum_instance_type


@register
class CoopBlockSumInstanceDecl(CoopInstanceTemplate):
    key = block_sum_instance_type
    instance_type = block_sum_instance_type
    primitive_name = "coop.block.sum"


@register_model(CoopBlockSumInstanceType)
class CoopBlockSumInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockSumInstanceType)
def lower_constant_block_sum_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
