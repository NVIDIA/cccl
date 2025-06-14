# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is semantically equivalent to the numba.cuda.cudadecl module.
# It is responsible for defining the Numba templates for cuda.cooperative
# primitives.

import operator

from numba.core import errors, types
from numba.core.typing.npydecl import (
    parse_dtype,
    parse_shape,
)
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    CallableTemplate,
    Registry,
    infer_global,
    signature,
)
from numba.extending import (
    type_callable,
    typeof_impl,
)

import cuda.cccl.cooperative.experimental as coop

registry = Registry()
register = registry.register
register_attr = registry.register_attr
register_global = registry.register_global


# =============================================================================
# Utils/Helpers
# =================================================================================
class CoopDeclMixin:
    """
    This is a dummy class that must be inherited by all cooperative methods
    in order for them to be recognized during the stage 5a rewriting pass of
    Numba.
    """


def get_coop_decl_class_map():
    return {
        subclass: getattr(subclass, "key")
        for subclass in CoopDeclMixin.__subclasses__()
    }


# =============================================================================
# Temp Storage
# =================================================================================
class TempStorageType(types.Type):
    def __init__(self):
        super().__init__(name="coop.TempStorage")


temp_storage_type = TempStorageType()


@typeof_impl.register(coop.TempStorage)
def typeof_temp_storage(*args, **kwargs):
    return temp_storage_type


@type_callable(coop.TempStorage)
def type_temp_storage(context):
    def typer(size_in_bytes=None, alignment=None, auto_sync=True):
        if size_in_bytes is not None:
            permitted = (types.Integer, types.IntegerLiteral)
            if not isinstance(size_in_bytes, permitted):
                msg = "size_in_bytes must be an integer value"
                raise errors.TypingError(msg)

        if alignment is not None:
            permitted = (types.Integer, types.IntegerLiteral)
            if not isinstance(alignment, permitted):
                msg = "alignment must be an integer value"
                raise errors.TypingError(msg)

        if auto_sync is not None:
            permitted = (bool, types.Boolean, types.BooleanLiteral)
            if not isinstance(auto_sync, permitted):
                msg = f"auto_sync must be a boolean value, got {auto_sync}"
                raise errors.TypingError(msg)

        return temp_storage_type

    return typer


class CoopTempStorageGetItemDecl(AbstractTemplate):
    # Allows for coop primitives to be called with a temp_storage argument
    # via the getitem syntax, e.g. `coop.block.load[temp_storage](...)`.
    def generic(self, args, kwds):
        assert not kwds, "No keyword arguments expected"
        assert len(args) == 2, "Expected two arguments"
        (func_obj, temp_storage) = args
        if not isinstance(func_obj, types.Function):
            return None

        try:
            typing_key = func_obj.typing_key
        except AttributeError:
            return None

        if typing_key != self.target_key:
            return None

        if not isinstance(temp_storage, TempStorageType):
            msg = f"temp_storage must be a {TempStorageType}, " f"got {temp_storage}"
            raise errors.TypingError(msg)

        return signature(
            func_obj,
            (func_obj, temp_storage),
        )


# =============================================================================
# Thread Data
# =================================================================================

# Our ThreadData type simplifies this code:
#
#    @cuda.jit
#    def kernel(d_in, items_per_thread):
#      thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
#      coop.block.load(d_in, thread_data, items_per_thread)
#      ...
#      coop.block.store(d_in, thread_data, items_per_thread)
#
# To this:
#
#    @cuda.jit
#    def kernel(d_in, d_out, items_per_thread):
#      thread_data = coop.ThreadData(items_per_thread)
#      coop.block.load(d_in, thread_data)
#      ...
#      coop.block.store(d_out, thread_data)
#
# We are able to infer the dtype during rewriting, obviating the need to
# specify it explicitly to the ThreadData constructor.  Additionally, we
# can obtain the `items_per_thread` value from the ThreadData object,
# obviating need to pass it explicitly to the load/store functions.


class ThreadDataType(types.Type):
    def __init__(self):
        super().__init__(name="coop.ThreadData")


thread_data_type = ThreadDataType()


@typeof_impl.register(coop.ThreadData)
def typeof_thread_data(*args, **kwargs):
    return thread_data_type


@type_callable(coop.ThreadData)
def type_thread_data(context):
    def typer(items_per_thread):
        permitted = (types.Integer, types.IntegerLiteral)
        if not isinstance(items_per_thread, permitted):
            msg = (
                "items_per_thread must be an integer or "
                f"integer literal, got {items_per_thread}"
            )
            raise errors.TypingError(msg)

        return thread_data_type

    return typer


# =============================================================================
# Arrays
# =================================================================================

# N.B. The upstream cuda.(local|shared).array() functions in numba-cuda don't
#      support being called with non-IntegerLiteral shapes, so we provide our
#      own versions for now that are more flexible.


class CoopArrayBaseTemplate(CallableTemplate):
    def generic(self):
        def typer(shape, dtype, alignment=None):
            # Allow the shape to be a tuple of integers or a single integer.
            valid_shape = isinstance(shape, types.Integer) or isinstance(
                shape, (types.Tuple, types.UniTuple)
            )
            if not valid_shape:
                return None

            if alignment is not None:
                permitted = (
                    types.Integer,
                    types.IntegerLiteral,
                    types.NoneType,
                )
                if not isinstance(alignment, permitted):
                    msg = "alignment must be an integer value"
                    raise errors.TypingError(msg)

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout="C")

        return typer


@register
class CoopSharedArrayDecl(CoopArrayBaseTemplate, CoopDeclMixin):
    key = coop.shared.array
    primitive_name = "coop.shared.array"


@register
class CoopLocalArrayDecl(CoopArrayBaseTemplate, CoopDeclMixin):
    key = coop.local.array
    primitive_name = "coop.local.array"


# =============================================================================
# Load & Store
# =============================================================================


class CoopLoadStoreBaseTemplate(CallableTemplate):
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

    def _validate_src_dst(self, src, dst):
        """
        Validate that *src* and *dst* are both provided, are device arrays,
        and have compatible types (dtype, ndim, layout).  Raise TypingError
        if any of the checks fail.  Return None if all checks pass.
        """
        if src is None or dst is None:
            raise errors.TypingError(
                f"{self.primitive_name} needs both 'src' and 'dst' arrays"
            )

        permitted = (types.Array, ThreadDataType)
        invalid_types = not isinstance(src, permitted) or not isinstance(dst, permitted)
        if invalid_types:
            raise errors.TypingError(
                f"{self.primitive_name} requires both 'src' and 'dst' to be "
                "device or thread-data arrays"
            )

        if isinstance(src, ThreadDataType) or isinstance(dst, ThreadDataType):
            # No more validation required if one of the types is thread data.
            return

        # Mismatched types.
        if src.dtype != dst.dtype:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' and 'dst' to have the "
                f"same dtype (got {src.dtype} vs {dst.dtype})"
            )

        # Mismatched dimensions.
        if src.ndim != dst.ndim:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' and 'dst' to have the "
                f"same number of dimensions (got {src.ndim} vs {dst.ndim})"
            )

        # Mismatched layout if neither is 'A'.
        invalid_layout = (
            src.layout != "A" and dst.layout != "A" and src.layout != dst.layout
        )
        if invalid_layout:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'src' and 'dst' to have the "
                f"same layout (got {src.layout!r} vs {dst.layout!r})"
            )

    def _validate_positive_integer_literal(self, value, param_name):
        """
        Validate that *value* is a positive integer literal and return it as
        an IntegerLiteral type.  If the underlying literal value is less than
        or equal to zero, raise a TypingError.  Otherwise, return None,
        indicating that this type is not supported.
        """
        if isinstance(value, types.IntegerLiteral):
            if value.literal_value <= 0:
                raise errors.TypingError(
                    f"'{param_name}' must be a positive integer; "
                    f"got {value.literal_value}"
                )
            return value
        return None

    def _validate_items_per_thread(self, items_per_thread):
        return self._validate_positive_integer_literal(
            items_per_thread,
            "items_per_thread",
        )

    def _validate_algorithm(self, algorithm):
        if algorithm is None:
            return

        enum_cls = self.algorithm_enum
        enum_name = enum_cls.__name__
        user_facing_name = f"cuda.{enum_name}"

        if not isinstance(algorithm, types.EnumMember):
            msg = (
                f"algorithm for {self.primitive_name} must be a member "
                f"of {user_facing_name}, got {algorithm}"
            )
            raise errors.TypingError(msg)

        if algorithm.instance_class is not enum_cls:
            name = algorithm.instance_class.__name__
            msg = (
                f"algorithm for {self.primitive_name} must be a member "
                f"of {user_facing_name}, got {name} "
            )
            raise errors.TypingError(msg)

    def _validate_temp_storage(self, temp_storage):
        """
        Validate that *temp_storage* is either None or a device array.
        Raise TypingError if it is not.  Return None if the checks pass.
        """
        if temp_storage is not None and not isinstance(temp_storage, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'temp_storage' to be a "
                "device array or None"
            )

    def _validate_args_and_create_signature(
        self, src, dst, items_per_thread=None, algorithm=None, temp_storage=None
    ):
        self._validate_src_dst(src, dst)

        using_thread_data = isinstance(src, ThreadDataType) or isinstance(
            dst, ThreadDataType
        )
        if not using_thread_data:
            # items_per_thread is mandatory if not using thread data.
            self._validate_items_per_thread(items_per_thread)

        self._validate_algorithm(algorithm)

        self._validate_temp_storage(temp_storage)

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

        if temp_storage is not None:
            arglist.append(temp_storage)

        sig = signature(
            types.void,
            *arglist,
        )

        return sig


class LoadMixin:
    src_first = True

    def generic(self):
        def typer(
            src,
            dst,
            items_per_thread=None,
            algorithm=None,
            temp_storage=None,
        ):
            return self._validate_args_and_create_signature(
                src,
                dst,
                items_per_thread,
                algorithm,
                temp_storage,
            )

        return typer


class StoreMixin:
    src_first = False

    def generic(self):
        def typer(
            dst,
            src,
            items_per_thread=None,
            algorithm=None,
            temp_storage=None,
        ):
            return self._validate_args_and_create_signature(
                src,
                dst,
                items_per_thread,
                algorithm,
                temp_storage,
            )

        return typer


# Load
@register
class CoopBlockLoadDecl(CoopLoadStoreBaseTemplate, LoadMixin, CoopDeclMixin):
    key = coop.block.load
    primitive_name = "coop.block.load"
    algorithm_enum = coop.BlockLoadAlgorithm
    default_algorithm = coop.BlockLoadAlgorithm.DIRECT


@infer_global(operator.getitem)
class CoopBlockLoadTempStorageGetItemDecl(CoopTempStorageGetItemDecl):
    target_key = coop.block.load
    target_template = CoopBlockLoadDecl


# Store
@register
class CoopBlockStoreDecl(CoopLoadStoreBaseTemplate, StoreMixin, CoopDeclMixin):
    key = coop.block.store
    primitive_name = "coop.block.store"
    algorithm_enum = coop.BlockStoreAlgorithm
    default_algorithm = coop.BlockStoreAlgorithm.DIRECT


@infer_global(operator.getitem)
class CoopBlockStoreTempStorageGetItemDecl(CoopTempStorageGetItemDecl):
    target_key = coop.block.store
    target_template = CoopBlockStoreDecl


# =============================================================================
# Module Template
# =============================================================================


@register_attr
class CoopSharedModuleTemplate(AttributeTemplate):
    key = types.Module(coop.shared)

    def resolve_array(self, mod):
        return types.Function(CoopSharedArrayDecl)


@register_attr
class CoopLocalModuleTemplate(AttributeTemplate):
    key = types.Module(coop.local)

    def resolve_array(self, mod):
        return types.Function(CoopLocalArrayDecl)


@register_attr
class CoopBlockModuleTemplate(AttributeTemplate):
    key = types.Module(coop.block)

    def resolve_load(self, mod):
        return types.Function(CoopBlockLoadDecl)

    def resolve_store(self, mod):
        return types.Function(CoopBlockStoreDecl)


@register_attr
class CoopModuleTemplate(AttributeTemplate):
    key = types.Module(coop)

    def resolve_block(self, mod):
        return types.Module(coop.block)

    def resolve_local(self, mod):
        return types.Module(coop.local)

    def resolve_shared(self, mod):
        return types.Module(coop.shared)

    def resolve_BlockLoadAlgorithm(self, mod):
        return types.Module(coop.BlockLoadAlgorithm)

    def resolve_BlockStoreAlgorithm(self, mod):
        return types.Module(coop.BlockStoreAlgorithm)

    def resolve_WarpLoadAlgorithm(self, mod):
        return types.Module(coop.WarpLoadAlgorithm)

    def resolve_WarpStoreAlgorithm(self, mod):
        return types.Module(coop.WarpStoreAlgorithm)


register_global(coop, types.Module(coop))
