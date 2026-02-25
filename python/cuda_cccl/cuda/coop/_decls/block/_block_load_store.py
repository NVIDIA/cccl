# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
import operator
from typing import Any, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    lower_builtin,
    models,
    register_model,
    typeof_impl,
)

import cuda.coop as coop

from ...block._block_load_store import (
    _make_load_rewrite as _make_block_load_rewrite,
)
from ...block._block_load_store import (
    _make_store_rewrite as _make_block_store_rewrite,
)
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopInstanceTypeMixin,
    CoopTempStorageGetItemDecl,
    TempStorageType,
    ThreadDataType,
    register,
    register_global,
    validate_algorithm,
    validate_items_per_thread,
    validate_src_dst,
    validate_temp_storage,
)

# =============================================================================
# Load & Store (Single-phase)
# =============================================================================


class CoopLoadStoreBaseTemplate(AbstractTemplate):
    """
    Base class for all cooperative load and store functions.  Subclasses must
    define the following attributes:
      - key: the function name (e.g. cuda.block.load)
      - primitive_name: the name of the primitive (e.g. "cuda.block.load")
      - algorithm_enum: the enum class for the algorithm (e.g.
        BlockLoadAlgorithm)
      - default_algorithm: the default algorithm to use if not specified
    """

    unsafe_casting = False
    exact_match_required = True
    prefer_literal = True

    def __init__(self, context=None):
        super().__init__(context=context)

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        dst = bound.arguments["dst"]
        validate_src_dst(self, src, dst)

        items_per_thread = bound.arguments.get("items_per_thread")
        using_thread_data = isinstance(src, ThreadDataType) or isinstance(
            dst, ThreadDataType
        )
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        algorithm = bound.arguments.get("algorithm")
        validate_algorithm(self, algorithm)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        # If we reach here, all arguments are valid.
        if self.src_first:
            array_args = (src, dst)
        else:
            array_args = (dst, src)

        arglist = [*array_args]

        if items_per_thread is not None:
            arglist.append(items_per_thread)

        if algorithm is not None:
            arglist.append(algorithm)

        num_valid_items = bound.arguments.get("num_valid_items")
        if num_valid_items is not None:
            if not isinstance(num_valid_items, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'num_valid_items' "
                    f"to be an integer, got: {num_valid_items}"
                )
            arglist.append(num_valid_items)

        oob_default = bound.arguments.get("oob_default")
        if oob_default is not None:
            if not self.src_first:
                raise errors.TypingError(
                    f"{self.primitive_name} does not support 'oob_default'"
                )
            if num_valid_items is None:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'num_valid_items' when "
                    "using 'oob_default'"
                )
            if isinstance(oob_default, (types.Array, ThreadDataType)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'oob_default' to be a scalar"
                )
            if not isinstance(oob_default, types.Number):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'oob_default' to be a scalar"
                )
            arglist.append(oob_default)

        if temp_storage is not None:
            arglist.append(temp_storage)

        sig = signature(
            types.void,
            *arglist,
        )

        return sig

    def _prevalidate_args(self, args):
        if len(args) < 2:
            raise errors.TypingError(
                f"{self.primitive_name} requires at least two positional arguments"
            )

    def generic(self, args, kwds):
        kwds = CoopAbstractTemplate._normalize_kernel_launch_dim_kwds(self, kwds)
        self._prevalidate_args(args)
        bound = self.signature(*args, **kwds)
        return self._validate_args_and_create_signature(bound)


class LoadMixin:
    src_first = True

    @staticmethod
    def signature(
        src: types.Array,
        dst: types.Array,
        items_per_thread: int = None,
        algorithm: coop.BlockLoadAlgorithm = None,
        num_valid_items: int = None,
        oob_default: Any = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(LoadMixin.signature).bind(
            src,
            dst,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: types.Array,
        dst: types.Array,
        num_valid_items: int = None,
        oob_default: Any = None,
        *,
        items_per_thread: int = None,
        algorithm: coop.BlockLoadAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(LoadMixin.signature_instance).bind(
            src,
            dst,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )


class StoreMixin:
    src_first = False

    @staticmethod
    def signature(
        dst: types.Array,
        src: types.Array,
        items_per_thread: int = None,
        algorithm: coop.BlockStoreAlgorithm = None,
        num_valid_items: int = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(LoadMixin.signature).bind(
            dst,
            src,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        dst: types.Array,
        src: types.Array,
        num_valid_items: int = None,
        *,
        items_per_thread: int = None,
        algorithm: coop.BlockStoreAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(StoreMixin.signature_instance).bind(
            dst,
            src,
            num_valid_items=num_valid_items,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )


# Load
@register_global(coop.block.load)
class CoopBlockLoadDecl(CoopLoadStoreBaseTemplate, LoadMixin, CoopDeclMixin):
    key = coop.block.load
    impl_key = _make_block_load_rewrite
    primitive_name = "coop.block.load"
    algorithm_enum = coop.BlockLoadAlgorithm
    default_algorithm = coop.BlockLoadAlgorithm.DIRECT


# register(CoopBlockLoadDecl)


@infer_global(operator.getitem)
class CoopBlockLoadTempStorageGetItemDecl(CoopTempStorageGetItemDecl):
    target_key = coop.block.load
    target_template = CoopBlockLoadDecl


# Store
@register_global(coop.block.store)
class CoopBlockStoreDecl(CoopLoadStoreBaseTemplate, StoreMixin, CoopDeclMixin):
    key = coop.block.store
    impl_key = _make_block_store_rewrite
    primitive_name = "coop.block.store"
    algorithm_enum = coop.BlockStoreAlgorithm
    default_algorithm = coop.BlockStoreAlgorithm.DIRECT


# register(CoopBlockStoreDecl)


@infer_global(operator.getitem)
class CoopBlockStoreTempStorageGetItemDecl(CoopTempStorageGetItemDecl):
    target_key = coop.block.store
    target_template = CoopBlockStoreDecl


@infer_global(operator.getitem)
class CoopGenericTempStorageGetItemDecl(CoopTempStorageGetItemDecl):
    target_key = None


# =============================================================================
# Instance-related Load & Store Scaffolding (Two-Phase)
# =============================================================================


class CoopLoadStoreInstanceBaseType(types.Type, CoopInstanceTypeMixin):
    """
    Base class for cooperative load/store instance types.  Subclasses must
    define the following class attributes:
      - decl_class: the declaration class for the load/store
        (e.g. CoopBlockLoadDecl)
      - src_first: a boolean indicating whether the source array comes first
        or second in the invocation.
    """

    decl_class = None

    def __init__(self):
        if self.decl_class is None:
            msg = "Subclasses must define 'decl_class' attribute"
            raise ValueError(msg)
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(self, *args, **kwargs):
        bound = self._bind_instance_signature(*args, **kwargs)
        return self.decl._validate_args_and_create_signature(bound, two_phase=True)


# Load
class CoopBlockLoadInstanceType(CoopLoadStoreInstanceBaseType, LoadMixin):
    decl_class = CoopBlockLoadDecl


block_load_instance_type = CoopBlockLoadInstanceType()


@typeof_impl.register(coop.block.load)
def typeof_block_load_instance(*args, **kwargs):
    return block_load_instance_type


@register
class CoopBlockLoadInstanceDecl(CoopInstanceTemplate):
    key = block_load_instance_type
    instance_type = block_load_instance_type
    primitive_name = "coop.block.load"


@register_model(CoopBlockLoadInstanceType)
class CoopBlockLoadInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockLoadInstanceType)
def lower_constant_block_load_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


@lower_builtin(CoopBlockLoadInstanceType, types.VarArg(types.Any))
def codegen_block_load(context, builder, sig, args):
    # This isn't actually ever called, but it needs to exist.
    return context.get_dummy_value()


@lower_builtin("call", CoopBlockLoadInstanceType, types.Array, types.Array)
def codegen_block_load_call(context, builder, sig, args):
    return context.get_dummy_value()


# Store
class CoopBlockStoreInstanceType(CoopLoadStoreInstanceBaseType, StoreMixin):
    decl_class = CoopBlockStoreDecl


block_store_instance_type = CoopBlockStoreInstanceType()


@typeof_impl.register(coop.block.store)
def typeof_block_store_instance(*args, **kwargs):
    return block_store_instance_type


@register
class CoopBlockStoreInstanceDecl(CoopInstanceTemplate):
    key = block_store_instance_type
    instance_type = block_store_instance_type
    primitive_name = "coop.block.store"


@register_model(CoopBlockStoreInstanceType)
class CoopBlockStoreInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockStoreInstanceType)
def lower_constant_block_store_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


@lower_builtin(CoopBlockStoreInstanceType, types.VarArg(types.Any))
def codegen_block_store(context, builder, sig, args):
    # This isn't actually ever called, but it needs to exist.
    return context.get_dummy_value()


@lower_builtin("call", CoopBlockStoreInstanceType, types.Array, types.Array)
def codegen_block_store_call(context, builder, sig, args):
    return context.get_dummy_value()
