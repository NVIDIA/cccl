# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is semantically equivalent to the numba.cuda.cudadecl module.
# It is responsible for defining the Numba templates for cuda.coop
# primitives.

import enum
import inspect
from typing import (
    Any,
    Literal,
    Optional,
    Union,
)

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.npydecl import (
    parse_dtype,
    parse_shape,
)
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Registry,
    Signature,
    signature,
)
from numba.cuda.cudaimpl import (
    cuda_local_array_integer,
    cuda_local_array_tuple,
    cuda_shared_array_integer,
    cuda_shared_array_tuple,
)
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
)
from numba.extending import (
    models,
    register_model,
    typeof_impl,
)

import cuda.coop as coop

from .._types import Invocable

# These rewrite helpers are intentionally imported directly and bound as
# `impl_key` on decl classes below. They are private implementation hooks (not
# public API symbols), so `__all__` exports from `cuda.coop.block/warp` do not
# cover this use case.
from .block import (
    import_side_effect_modules as _import_block_decl_side_effect_modules,
)
from .warp import (
    import_side_effect_modules as _import_warp_decl_side_effect_modules,
)

registry = Registry()
register = registry.register
register_attr = registry.register_attr
register_global = registry.register_global


# =============================================================================
# Utils/Helpers
# =============================================================================
class CoopDeclMixin:
    """
    This is a dummy class that must be inherited by all cooperative methods
    in order for them to be recognized during Numba's rewriting pass.  This
    applies to the single-phase primitives referenced within a kernel.  For
    instances of primitives created outside the kernel (i.e. via two-phase),
    we use `CoopInstanceTypeMixin`.
    """

    is_constructor = False
    """
    When true, indicates that this is a constructor for a cooperative
    primitive that needs an explicit instance type to be created, such
    that other methods can be invoked on it, e.g.:
        run_length = coop.block.run_length(...)
        run_length.decode(...)
    The `run_length()` call is a constructor in this instance, and should
    have `is_constructor = True`
    """


def get_coop_decl_class_map():
    _import_block_decl_side_effect_modules()
    _import_warp_decl_side_effect_modules()

    def _iter_decl_subclasses():
        stack = list(CoopDeclMixin.__subclasses__())
        seen = set()
        while stack:
            subclass = stack.pop()
            if subclass in seen:
                continue
            seen.add(subclass)
            yield subclass
            stack.extend(subclass.__subclasses__())

    decl_map = {}
    for subclass in _iter_decl_subclasses():
        impl_key = getattr(subclass, "impl_key", None)
        if impl_key is None:
            impl_key = getattr(subclass, "key")
        decl_map[subclass] = impl_key
    return decl_map


# Unlike the dict returned by get_coop_decl_class_map() on the fly, we use
# a persistent global set here that each instance of an instance type will
# register itself to.
#
# N.B. "Instance of instance type" is unfortunately wordy, however, it's the
#      clearest way to describe what's being tracked without any subtle
#      ambiguity sneaking in.
__INSTANCE_OF_INSTANCE_TYPES_MAP = {}


def get_coop_instance_of_instance_types_map():
    global __INSTANCE_OF_INSTANCE_TYPES_MAP
    return __INSTANCE_OF_INSTANCE_TYPES_MAP


def _register_coop_instance_of_instance_type(instance: types.Type) -> None:
    """
    Register an instance of an instance type.
    """
    name = instance.name
    if not name.startswith("coop"):
        raise ValueError(f"Instance type {name} must start with 'coop'")

    global __INSTANCE_OF_INSTANCE_TYPES_MAP
    if name in __INSTANCE_OF_INSTANCE_TYPES_MAP:
        raise ValueError(f"Instance type {name} is already registered")
    __INSTANCE_OF_INSTANCE_TYPES_MAP[name] = instance


class CoopInstanceTypeMixin:
    """
    This is a dummy class that must be inherited by all instance types of
    two-phase cooperative methods.
    """

    def __init__(self):
        name = self.name
        # Ensure our type.Type's __init__() has already been called.  This
        # isn't strictly necessary, as there's a "startswith('coop')" check
        # in the _register_coop_instance_of_instance_type() function, but
        # we do it here as well as we can raise a more specific error message.
        if not name.startswith("coop"):
            msg = (
                "CoopInstanceTypeMixin.__init__() called before other subclass "
                " __init__() methods."
            )
            raise ValueError(msg)

        # Register this instance type in the global map.
        _register_coop_instance_of_instance_type(self)

    def _bind_instance_signature(self, *args, **kwargs):
        """
        Bind runtime call arguments to the appropriate signature.

        If a decl provides a specialized instance-call signature, prefer it.
        This allows two-phase instances to use a runtime-friendly argument
        ordering while still sharing validation logic.
        """
        sig_fn = getattr(self.decl, "signature_instance", None)
        if sig_fn is None:
            sig_fn = self.decl.signature
        return sig_fn(*args, **kwargs)


class CoopAbstractTemplate(AbstractTemplate):
    """
    A base class for cooperative templates that provides a common interface
    for validating arguments and creating signatures.
    """

    unsafe_casting = False
    exact_match_required = True
    prefer_literal = True

    def __init__(self, context=None):
        super().__init__(context=context)

    def _prevalidate_args(self, args, kwds):
        if len(args) + len(kwds) < self.minimum_num_args:
            suffix = "s" if self.minimum_num_args >= 2 else ""
            msg = (
                f"{self.primitive_name} requires at least "
                f"{self.minimum_num_args} positional argument{suffix}"
            )
            raise errors.TypingError(msg)

    def generic(self, args, kwds):
        """
        Validate the arguments and create a signature for the cooperative
        primitive.
        """
        self._prevalidate_args(args, kwds)
        bound = self.signature(*args, **kwds)
        return self._validate_args_and_create_signature(bound)


class CoopInstanceTemplate(AbstractTemplate):
    """
    Base class for cooperative instance call templates.  Subclasses must set
    `instance_type` to the corresponding CoopInstanceType.
    """

    unsafe_casting = False
    exact_match_required = True
    prefer_literal = True

    instance_type = None
    primitive_name = None

    def generic(self, args, kwds):
        instance_type = self.instance_type
        if instance_type is None:
            msg = "CoopInstanceTemplate requires `instance_type` to be set"
            raise errors.TypingError(msg)
        try:
            return instance_type._validate_args_and_create_signature(*args, **kwds)
        except TypeError as exc:
            raise errors.TypingError(str(exc)) from exc


# =============================================================================
# Temp Storage
# =============================================================================


# TempStorage is a compile-time placeholder that lets kernels request explicit
# cooperative temporary storage without forcing users to hand-write shared-array
# declarations. Rewrite inserts the concrete shared-memory allocation and wires
# it to primitive calls.
class TempStorageType(types.Type):
    def __init__(self):
        super().__init__(name="coop.TempStorage")


temp_storage_type = TempStorageType()
TempStorageSharing = Literal["shared", "exclusive"]
_TEMP_STORAGE_SHARING_VALUES = frozenset(("shared", "exclusive"))


@typeof_impl.register(coop.TempStorage)
def typeof_temp_storage(*args, **kwargs):
    return temp_storage_type


@register
class CoopTempStorageDecl(CoopAbstractTemplate):
    key = coop.TempStorage
    primitive_name = "coop.TempStorage"
    minimum_num_args = 0

    @staticmethod
    def signature(
        size_in_bytes: Optional[int] = None,
        alignment: Optional[int] = None,
        auto_sync: Optional[bool] = None,
        sharing: Optional[TempStorageSharing] = "shared",
    ):
        return inspect.signature(CoopTempStorageDecl.signature).bind(
            size_in_bytes,
            alignment=alignment,
            auto_sync=auto_sync,
            sharing=sharing,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        arglist = []

        size_in_bytes = bound.arguments.get("size_in_bytes")
        if isinstance(size_in_bytes, types.NoneType):
            arglist.append(size_in_bytes)
            size_in_bytes = None
        if size_in_bytes is not None:
            permitted = (types.Integer, types.IntegerLiteral)
            if not isinstance(size_in_bytes, permitted):
                msg = "size_in_bytes must be an integer value"
                raise errors.TypingError(msg)
            arglist.append(size_in_bytes)

        alignment = bound.arguments.get("alignment")
        if isinstance(alignment, types.NoneType):
            arglist.append(alignment)
            alignment = None
        if alignment is not None:
            permitted = (types.Integer, types.IntegerLiteral)
            if not isinstance(alignment, permitted):
                msg = "alignment must be an integer value"
                raise errors.TypingError(msg)
            arglist.append(alignment)

        auto_sync = bound.arguments.get("auto_sync")
        if isinstance(auto_sync, types.NoneType):
            arglist.append(auto_sync)
            auto_sync = None
        if auto_sync is not None:
            permitted = (bool, types.Boolean, types.BooleanLiteral)
            if not isinstance(auto_sync, permitted):
                msg = f"auto_sync must be a boolean value, got {auto_sync}"
                raise errors.TypingError(msg)
            arglist.append(auto_sync)

        sharing = bound.arguments.get("sharing")
        if isinstance(sharing, types.NoneType):
            arglist.append(sharing)
            sharing = None
        if sharing is not None:
            permitted = (str, types.StringLiteral)
            if not isinstance(sharing, permitted):
                msg = f"sharing must be a string literal value, got {sharing}"
                raise errors.TypingError(msg)
            sharing_value = (
                sharing.literal_value
                if isinstance(sharing, types.StringLiteral)
                else sharing
            )
            if not isinstance(sharing_value, str):
                msg = f"sharing must be a string literal value, got {sharing}"
                raise errors.TypingError(msg)
            sharing_value = sharing_value.strip().lower()
            if sharing_value not in _TEMP_STORAGE_SHARING_VALUES:
                msg = f"sharing must be 'shared' or 'exclusive', got {sharing!r}"
                raise errors.TypingError(msg)
            arglist.append(sharing)

        return signature(temp_storage_type, *arglist)


class CoopTempStorageGetItemDecl(AbstractTemplate):
    # Allows for coop primitives to be called with a temp_storage argument
    # via the getitem syntax, e.g. `coop.block.load[temp_storage](...)`.
    target_key = None

    def _supports_getitem_temp_storage(self, func_obj):
        try:
            templates = func_obj.templates
        except AttributeError:
            return False

        if len(templates) != 1:
            return False

        template = templates[0]
        primitive_name = getattr(template, "primitive_name", "")
        if not primitive_name.startswith("coop."):
            return False

        for signature_attr in ("signature_instance", "signature"):
            signature_fn = getattr(template, signature_attr, None)
            if signature_fn is None:
                continue
            try:
                params = inspect.signature(signature_fn).parameters
            except (TypeError, ValueError):
                continue
            if "temp_storage" in params:
                return True
        return False

    def generic(self, args, kwds):
        assert not kwds, "No keyword arguments expected"
        assert len(args) == 2, "Expected two arguments"
        (func_obj, temp_storage) = args
        if not isinstance(func_obj, types.Function):
            return None

        target_key = getattr(self, "target_key", None)
        try:
            typing_key = func_obj.typing_key
        except AttributeError:
            return None

        if target_key is not None:
            if typing_key != target_key:
                return None
        else:
            # Avoid overlap with specialized getitem templates that target
            # block load/store explicitly.
            if typing_key in (coop.block.load, coop.block.store):
                return None
            if not self._supports_getitem_temp_storage(func_obj):
                return None

        if not isinstance(temp_storage, TempStorageType):
            msg = f"temp_storage must be a {TempStorageType}, got {temp_storage}"
            raise errors.TypingError(msg)

        return signature(
            func_obj,
            (func_obj, temp_storage),
        )


# =============================================================================
# Decomposer
# =============================================================================


# Decomposer support is currently placeholder-only in coop typing. We model it
# as an opaque compile-time token so public APIs can accept it while full UDT
# decomposer lowering support is still under development.
class DecomposerType(types.Type):
    def __init__(self):
        super().__init__(name="coop.Decomposer")


decomposer_type = DecomposerType()


@typeof_impl.register(coop.Decomposer)
def typeof_decomposer(*args, **kwargs):
    return decomposer_type


@register_model(DecomposerType)
class DecomposerModel(models.OpaqueModel):
    pass


@lower_constant(DecomposerType)
def lower_constant_decomposer(context, builder, typ, value):
    return context.get_dummy_value()


# =============================================================================
# Thread Data
# =============================================================================

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
# We are able to infer the dtype during rewriting in many cases, obviating
# the need to specify it explicitly to the ThreadData constructor. If
# inference is ambiguous, users can pass `dtype` directly. Additionally, we
# can obtain the `items_per_thread` value from the ThreadData object,
# obviating the need to pass it explicitly to the load/store functions.


class ThreadDataType(types.Array):
    def __init__(self, dtype, items_per_thread=None):
        super().__init__(dtype=dtype, ndim=1, layout="A")
        self.items_per_thread = items_per_thread
        self.name = "coop.ThreadData"

    def is_precise(self):
        # ThreadData may be created without an explicit dtype; we rely on
        # rewrite-time inference from usage to determine the actual dtype.
        return True


thread_data_type = ThreadDataType(types.undefined)


@typeof_impl.register(coop.ThreadData)
def typeof_thread_data(*args, **kwargs):
    return thread_data_type


@register_model(ThreadDataType)
class ThreadDataModel(models.ArrayModel):
    pass


@register
class CoopThreadDataDecl(CoopAbstractTemplate):
    key = coop.ThreadData
    primitive_name = "coop.ThreadData"
    minimum_num_args = 1

    @staticmethod
    def signature(items_per_thread: int, dtype: Optional[Any] = None):
        return inspect.signature(CoopThreadDataDecl.signature).bind(
            items_per_thread,
            dtype=dtype,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        arglist = []

        items_per_thread = bound.arguments.get("items_per_thread")
        permitted = (types.Integer, types.IntegerLiteral)
        if not isinstance(items_per_thread, permitted):
            msg = (
                "items_per_thread must be an integer or "
                f"integer literal, got {items_per_thread}"
            )
            raise errors.TypingError(msg)
        arglist.append(items_per_thread)

        dtype = bound.arguments.get("dtype")
        if isinstance(dtype, types.NoneType):
            arglist.append(dtype)
            dtype = None
        dtype_type = types.undefined
        if dtype is not None:
            try:
                dtype_type = parse_dtype(dtype)
            except Exception as exc:
                msg = f"Invalid dtype for coop.ThreadData: {dtype}"
                raise errors.TypingError(msg) from exc
            if dtype_type is None:
                if isinstance(dtype, types.Type):
                    dtype_type = dtype
                else:
                    msg = f"Invalid dtype for coop.ThreadData: {dtype}"
                    raise errors.TypingError(msg)
            arglist.append(dtype)

        return signature(
            ThreadDataType(dtype_type, items_per_thread=items_per_thread),
            *arglist,
        )


# =============================================================================
# Arrays
# =============================================================================

# N.B. The upstream cuda.(local|shared).array() functions in numba-cuda don't
#      support being called with non-IntegerLiteral shapes, so we provide our
#      own versions for now that are more flexible.


class CoopArrayBaseTemplate(CoopAbstractTemplate):
    minimum_num_args = 1

    def _prevalidate_args(self, args, kwds):
        if len(args) >= self.minimum_num_args:
            return

        has_shape = len(args) >= 1 or "shape" in kwds
        has_dtype = len(args) >= 2 or "dtype" in kwds
        if has_shape and has_dtype:
            return

        suffix = "s" if self.minimum_num_args >= 2 else ""
        msg = (
            f"{self.primitive_name} requires at least "
            f"{self.minimum_num_args} positional argument{suffix}"
        )
        raise errors.TypingError(msg)

    def generic(self, args, kwds):
        """
        Validate the arguments and create a signature for the array primitive.
        """
        self._prevalidate_args(args, kwds)
        bound = self.signature(*args, **kwds)
        return self._validate_args_and_create_signature(bound)

    @classmethod
    def signature(
        cls: type,  # pylint: disable=unused-argument
        shape: Union[types.Integer, types.Tuple, types.UniTuple],
        dtype: types.DType,
        alignment: Optional[types.Integer] = None,
    ):
        return inspect.signature(cls.signature).bind(
            shape,
            dtype,
            alignment=alignment,
        )

    def _validate_args_and_create_signature(self, bound, two_phase: bool = False):
        shape = bound.arguments.get("shape")
        if not shape:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'shape' to be specified"
            )

        dtype = bound.arguments.get("dtype")
        if not dtype:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'dtype' to be specified"
            )

        ndim = parse_shape(shape)
        if not ndim:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'shape' to be a valid Numba "
                f"shape, got: {shape}"
            )

        dtype = bound.arguments["dtype"]
        nb_dtype = parse_dtype(dtype)
        if not nb_dtype:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'dtype' to be a valid Numba "
                f"dtype, got: {dtype}"
            )

        alignment = bound.arguments.get("alignment")

        arglist = [shape, dtype]

        alignment = bound.arguments.get("alignment")
        if alignment is not None:
            if not isinstance(alignment, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'alignment' to be an "
                    f"integer or integer literal, got: {alignment}"
                )
            arglist.append(alignment)

        # Create a Python signature so kw args are accepted in lowering.
        params = [
            inspect.Parameter("shape", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("dtype", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
        if alignment is not None:
            params.append(
                inspect.Parameter(
                    "alignment",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
        pysig = inspect.Signature(params)

        # Create the signature with the validated arguments.
        sig = Signature(
            types.Array(dtype=nb_dtype, ndim=ndim, layout="C"),
            tuple(arglist),
            recvr=None,
            pysig=pysig,
        )

        return sig


@register
class CoopSharedArrayDecl(CoopArrayBaseTemplate, CoopDeclMixin):
    key = coop.shared.array
    primitive_name = "coop.shared.array"


@register
class CoopLocalArrayDecl(CoopArrayBaseTemplate, CoopDeclMixin):
    key = coop.local.array
    primitive_name = "coop.local.array"


@cuda_lower(coop.shared.array, types.IntegerLiteral, types.Any)
@cuda_lower(coop.shared.array, types.IntegerLiteral, types.Any, types.IntegerLiteral)
@cuda_lower(coop.shared.array, types.IntegerLiteral, types.Any, types.NoneType)
def coop_shared_array_integer(context, builder, sig, args):
    return cuda_shared_array_integer(context, builder, sig, args)


@cuda_lower(coop.shared.array, types.BaseTuple, types.Any)
@cuda_lower(coop.shared.array, types.BaseTuple, types.Any, types.IntegerLiteral)
@cuda_lower(coop.shared.array, types.BaseTuple, types.Any, types.NoneType)
def coop_shared_array_tuple(context, builder, sig, args):
    return cuda_shared_array_tuple(context, builder, sig, args)


@cuda_lower(coop.local.array, types.IntegerLiteral, types.Any)
@cuda_lower(coop.local.array, types.IntegerLiteral, types.Any, types.IntegerLiteral)
@cuda_lower(coop.local.array, types.IntegerLiteral, types.Any, types.NoneType)
def coop_local_array_integer(context, builder, sig, args):
    return cuda_local_array_integer(context, builder, sig, args)


@cuda_lower(coop.local.array, types.BaseTuple, types.Any)
@cuda_lower(coop.local.array, types.BaseTuple, types.Any, types.IntegerLiteral)
@cuda_lower(coop.local.array, types.BaseTuple, types.Any, types.NoneType)
def coop_local_array_tuple(context, builder, sig, args):
    return cuda_local_array_tuple(context, builder, sig, args)


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_src_dst(obj, src, dst):
    """
    Validate that *src* and *dst* are both provided, are device arrays,
    and have compatible types (dtype, ndim, layout).  Raise TypingError
    if any of the checks fail.  Return None if all checks pass.
    """
    if src is None or dst is None:
        raise errors.TypingError(
            f"{obj.primitive_name} needs both 'src' and 'dst' arrays"
        )

    permitted = (types.Array, ThreadDataType)
    invalid_types = not isinstance(src, permitted) or not isinstance(dst, permitted)
    if invalid_types:
        raise errors.TypingError(
            f"{obj.primitive_name} requires both 'src' and 'dst' to be "
            "device or thread-data arrays"
        )

    if isinstance(src, ThreadDataType) or isinstance(dst, ThreadDataType):
        # No more validation required if one of the types is thread data.
        return

    # Mismatched types.
    if src.dtype != dst.dtype:
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'src' and 'dst' to have the "
            f"same dtype (got {src.dtype} vs {dst.dtype})"
        )

    # Mismatched dimensions.
    if src.ndim != dst.ndim:
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'src' and 'dst' to have the "
            f"same number of dimensions (got {src.ndim} vs {dst.ndim})"
        )

    # Mismatched layout if neither is 'A'.
    invalid_layout = (
        src.layout != "A" and dst.layout != "A" and src.layout != dst.layout
    )
    if invalid_layout:
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'src' and 'dst' to have the "
            f"same layout (got {src.layout!r} vs {dst.layout!r})"
        )


def validate_positive_integer_literal(obj, value, param_name):
    """
    Validate that *value* is a positive integer literal and return it as
    an IntegerLiteral type.  If the underlying literal value is less than
    or equal to zero, raise a TypingError.  Otherwise, return None,
    indicating that this type is not supported.
    """
    if isinstance(value, types.IntegerLiteral):
        if value.literal_value <= 0:
            raise errors.TypingError(
                f"{obj.primitive_name} parameter '{param_name}' must be a "
                f"positive integer; got {value.literal_value}"
            )
        return value
    return None


def validate_items_per_thread(obj, items_per_thread):
    return validate_positive_integer_literal(
        obj,
        items_per_thread,
        "items_per_thread",
    )


def validate_threads_in_warp(obj, threads_in_warp):
    return validate_positive_integer_literal(
        obj,
        threads_in_warp,
        "threads_in_warp",
    )


def process_items_per_thread(obj, bound, arglist, two_phase, target_array=None):
    items_per_thread = bound.arguments.get("items_per_thread")
    if target_array is not None:
        if isinstance(target_array, (tuple, list)):
            using_thread_data = any(
                isinstance(array, ThreadDataType) for array in target_array
            )
        else:
            using_thread_data = isinstance(target_array, ThreadDataType)
    else:
        using_thread_data = False

    if items_per_thread is None:
        if using_thread_data:
            return items_per_thread
        if two_phase:
            return items_per_thread
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'items_per_thread' to be specified"
        )

    if not using_thread_data:
        if not two_phase or items_per_thread is not None:
            maybe_literal = validate_items_per_thread(obj, items_per_thread)
            if maybe_literal is not None:
                items_per_thread = maybe_literal
            if items_per_thread is not None:
                arglist.append(items_per_thread)
                return items_per_thread

    return items_per_thread


def validate_algorithm(obj, algorithm):
    if algorithm is None:
        return

    enum_cls = obj.algorithm_enum
    enum_name = enum_cls.__name__
    user_facing_name = f"cuda.{enum_name}"

    if isinstance(algorithm, enum.IntEnum):
        # If the algorithm is an IntEnum, we can directly check its value.
        if algorithm not in enum_cls:
            msg = (
                f"algorithm for {obj.primitive_name} must be a member "
                f"of {user_facing_name}, got {algorithm}"
            )
            raise errors.TypingError(msg)
        return

    if not isinstance(algorithm, types.EnumMember):
        msg = (
            f"algorithm for {obj.primitive_name} must be a member "
            f"of {user_facing_name}, got {algorithm}"
        )
        raise errors.TypingError(msg)

    if algorithm.instance_class is not enum_cls:
        name = algorithm.instance_class.__name__
        msg = (
            f"algorithm for {obj.primitive_name} must be a member "
            f"of {user_facing_name}, got {name} "
        )
        raise errors.TypingError(msg)


def process_algorithm(obj, bound, arglist):
    algorithm = bound.arguments.get("algorithm")
    if algorithm is None:
        # If no algorithm is specified, use the default.
        algorithm = obj.default_algorithm
    if algorithm is None:
        raise RuntimeError(
            f"{obj.primitive_name} requires an algorithm to be specified, "
            "either via the 'algorithm' argument or by setting a default "
            "algorithm in the class definition."
        )
    validate_algorithm(obj, algorithm)
    arglist.append(algorithm)

    return algorithm


def validate_temp_storage(obj, temp_storage):
    """
    Validate that *temp_storage* is either None or a device array.
    Raise TypingError if it is not.  Return None if the checks pass.
    """
    if temp_storage is None:
        return

    if isinstance(temp_storage, TempStorageType):
        return

    if not isinstance(temp_storage, types.Array):
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'temp_storage' to be a device array or None"
        )

    dtype = temp_storage.dtype
    if not (isinstance(dtype, types.Integer) and dtype.bitwidth == 8):
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'temp_storage' to be a uint8 array"
        )


# =============================================================================
# Load & Store (Single-phase)
# =============================================================================

_decls_block_load_store = __import__(
    "cuda.coop._decls.block._block_load_store",
    fromlist=[
        "CoopBlockLoadDecl",
        "CoopBlockLoadInstanceType",
        "CoopBlockStoreDecl",
        "CoopBlockStoreInstanceType",
        "block_load_instance_type",
        "block_store_instance_type",
    ],
)
CoopBlockLoadDecl = _decls_block_load_store.CoopBlockLoadDecl
CoopBlockLoadInstanceType = _decls_block_load_store.CoopBlockLoadInstanceType
CoopBlockStoreDecl = _decls_block_load_store.CoopBlockStoreDecl
CoopBlockStoreInstanceType = _decls_block_load_store.CoopBlockStoreInstanceType
block_load_instance_type = _decls_block_load_store.block_load_instance_type
block_store_instance_type = _decls_block_load_store.block_store_instance_type


# =============================================================================
# Warp Load & Store (Single-phase)
# =============================================================================


class CoopWarpLoadStoreBaseTemplate(AbstractTemplate):
    """
    Base class for warp load/store functions. Subclasses must define:
      - key
      - primitive_name
      - algorithm_enum
      - default_algorithm
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

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal

        algorithm = bound.arguments.get("algorithm")
        validate_algorithm(self, algorithm)

        num_valid_items = bound.arguments.get("num_valid_items")
        if num_valid_items is not None and not isinstance(
            num_valid_items, types.Integer
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'num_valid_items' "
                f"to be an integer, got: {num_valid_items}"
            )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        if self.src_first:
            array_args = (src, dst)
        else:
            array_args = (dst, src)

        arglist = [*array_args]

        if items_per_thread is not None:
            arglist.append(items_per_thread)

        if threads_in_warp is not None:
            arglist.append(threads_in_warp)

        if algorithm is not None:
            arglist.append(algorithm)

        if num_valid_items is not None:
            arglist.append(num_valid_items)

        oob_default = bound.arguments.get("oob_default")
        if oob_default is not None:
            if num_valid_items is None:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'num_valid_items' when "
                    "'oob_default' is specified"
                )
            if not isinstance(oob_default, (types.Number, types.Boolean)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'oob_default' to be a scalar"
                )
            arglist.append(oob_default)
        if temp_storage is not None:
            arglist.append(temp_storage)

        sig = signature(types.void, *arglist)
        return sig

    def _prevalidate_args(self, args):
        if len(args) < 2:
            raise errors.TypingError(
                f"{self.primitive_name} requires at least two positional arguments"
            )

    def generic(self, args, kwds):
        self._prevalidate_args(args)
        bound = self.signature(*args, **kwds)
        return self._validate_args_and_create_signature(bound)


class WarpLoadMixin:
    src_first = True

    @staticmethod
    def signature(
        src: types.Array,
        dst: types.Array,
        items_per_thread: int = None,
        threads_in_warp: int = 32,
        algorithm: coop.WarpLoadAlgorithm = None,
        num_valid_items: int = None,
        oob_default=None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(WarpLoadMixin.signature).bind(
            src,
            dst,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
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
        oob_default=None,
        *,
        items_per_thread: int = None,
        threads_in_warp: int = None,
        algorithm: coop.WarpLoadAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(WarpLoadMixin.signature_instance).bind(
            src,
            dst,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )


class WarpStoreMixin:
    src_first = False

    @staticmethod
    def signature(
        dst: types.Array,
        src: types.Array,
        items_per_thread: int = None,
        threads_in_warp: int = 32,
        algorithm: coop.WarpStoreAlgorithm = None,
        num_valid_items: int = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(WarpStoreMixin.signature).bind(
            dst,
            src,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
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
        threads_in_warp: int = None,
        algorithm: coop.WarpStoreAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(WarpStoreMixin.signature_instance).bind(
            dst,
            src,
            num_valid_items=num_valid_items,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )


CoopWarpLoadDecl = None
CoopWarpStoreDecl = None


# =============================================================================
# Exchange
# =============================================================================

CoopBlockExchangeDecl = None


# =============================================================================
# Adjacent Difference
# =============================================================================

CoopBlockAdjacentDifferenceDecl = None


# =============================================================================
# Shuffle
# =============================================================================

CoopBlockShuffleDecl = None


# =============================================================================
# Discontinuity
# =============================================================================

CoopBlockDiscontinuityDecl = None


# =============================================================================
# Radix Sort
# =============================================================================

CoopBlockRadixSortDecl = None
CoopBlockRadixSortDescendingDecl = None


# =============================================================================
# Radix Rank
# =============================================================================

CoopBlockRadixRankDecl = None


# =============================================================================
# Instance-related Load & Store Scaffolding (Two-Phase)
# =============================================================================

# The following scaffolding allows us to seamlessly handle calling invocables
# within a CUDA kernel that were created outside the kernel via the two-phase
# approach, e.g.:
#   block_load = coop.block.load(dtype, dim, items_per_thread)
#   @cuda.jit
#   def kernel(d_in, d_out, items_per_thread):
#       thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
#       block_load(d_in, thread_data, items_per_thread)


# Generic instance type for two-phase primitives with no special handling.
class CoopSimpleInstanceType(types.Type, CoopInstanceTypeMixin):
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


# =============================================================================
# Histogram
# =============================================================================

_decls_block_histogram = __import__(
    "cuda.coop._decls.block._block_histogram",
    fromlist=[
        "CoopBlockHistogramAttrsTemplate",
        "CoopBlockHistogramCompositeDecl",
        "CoopBlockHistogramDecl",
        "CoopBlockHistogramInitDecl",
        "CoopBlockHistogramInstanceType",
        "block_histogram_attrs_template",
        "block_histogram_instance_type",
    ],
)
CoopBlockHistogramAttrsTemplate = _decls_block_histogram.CoopBlockHistogramAttrsTemplate
CoopBlockHistogramCompositeDecl = _decls_block_histogram.CoopBlockHistogramCompositeDecl
CoopBlockHistogramDecl = _decls_block_histogram.CoopBlockHistogramDecl
CoopBlockHistogramInitDecl = _decls_block_histogram.CoopBlockHistogramInitDecl
CoopBlockHistogramInstanceType = _decls_block_histogram.CoopBlockHistogramInstanceType
block_histogram_attrs_template = _decls_block_histogram.block_histogram_attrs_template
block_histogram_instance_type = _decls_block_histogram.block_histogram_instance_type

# =============================================================================
# RunLengthDecode
# =============================================================================

_decls_block_run_length_decode = __import__(
    "cuda.coop._decls.block._block_run_length_decode",
    fromlist=[
        "CoopBlockRunLengthAttrsTemplate",
        "CoopBlockRunLengthDecodeDecl",
        "CoopBlockRunLengthDecl",
        "CoopBlockRunLengthInstanceType",
        "block_run_length_attrs_template",
        "block_run_length_instance_type",
    ],
)
CoopBlockRunLengthAttrsTemplate = (
    _decls_block_run_length_decode.CoopBlockRunLengthAttrsTemplate
)
CoopBlockRunLengthDecodeDecl = (
    _decls_block_run_length_decode.CoopBlockRunLengthDecodeDecl
)
CoopBlockRunLengthDecl = _decls_block_run_length_decode.CoopBlockRunLengthDecl
CoopBlockRunLengthInstanceType = (
    _decls_block_run_length_decode.CoopBlockRunLengthInstanceType
)
block_run_length_attrs_template = (
    _decls_block_run_length_decode.block_run_length_attrs_template
)
block_run_length_instance_type = (
    _decls_block_run_length_decode.block_run_length_instance_type
)

# =============================================================================
# Scan
# =============================================================================


_decls_block_scan = __import__(
    "cuda.coop._decls.block._block_scan",
    fromlist=[
        "CoopBlockExclusiveScanDecl",
        "CoopBlockExclusiveSumDecl",
        "CoopBlockInclusiveScanDecl",
        "CoopBlockInclusiveSumDecl",
        "CoopBlockScanDecl",
        "block_scan_instance_type",
    ],
)
CoopBlockExclusiveScanDecl = _decls_block_scan.CoopBlockExclusiveScanDecl
CoopBlockExclusiveSumDecl = _decls_block_scan.CoopBlockExclusiveSumDecl
CoopBlockInclusiveScanDecl = _decls_block_scan.CoopBlockInclusiveScanDecl
CoopBlockInclusiveSumDecl = _decls_block_scan.CoopBlockInclusiveSumDecl
CoopBlockScanDecl = _decls_block_scan.CoopBlockScanDecl
block_scan_instance_type = _decls_block_scan.block_scan_instance_type


_decls_block_reduce = __import__(
    "cuda.coop._decls.block._block_reduce",
    fromlist=[
        "CoopBlockReduceDecl",
        "CoopBlockReduceInstanceType",
        "CoopBlockSumDecl",
        "CoopBlockSumInstanceType",
        "block_reduce_instance_type",
        "block_sum_instance_type",
    ],
)
CoopBlockReduceDecl = _decls_block_reduce.CoopBlockReduceDecl
CoopBlockReduceInstanceType = _decls_block_reduce.CoopBlockReduceInstanceType
CoopBlockSumDecl = _decls_block_reduce.CoopBlockSumDecl
CoopBlockSumInstanceType = _decls_block_reduce.CoopBlockSumInstanceType
block_reduce_instance_type = _decls_block_reduce.block_reduce_instance_type
block_sum_instance_type = _decls_block_reduce.block_sum_instance_type

# Block Primitives (Two-phase Instances)
# =============================================================================


_decls_block_exchange = __import__(
    "cuda.coop._decls.block._block_exchange",
    fromlist=[
        "CoopBlockExchangeDecl",
        "CoopBlockExchangeInstanceType",
        "block_exchange_instance_type",
    ],
)
CoopBlockExchangeDecl = _decls_block_exchange.CoopBlockExchangeDecl
CoopBlockExchangeInstanceType = _decls_block_exchange.CoopBlockExchangeInstanceType
block_exchange_instance_type = _decls_block_exchange.block_exchange_instance_type


_decls_block_merge_sort = __import__(
    "cuda.coop._decls.block._block_merge_sort",
    fromlist=[
        "CoopBlockMergeSortDecl",
        "CoopBlockMergeSortPairsDecl",
        "CoopBlockMergeSortInstanceType",
        "CoopBlockMergeSortPairsInstanceType",
        "block_merge_sort_instance_type",
        "block_merge_sort_pairs_instance_type",
    ],
)
CoopBlockMergeSortDecl = _decls_block_merge_sort.CoopBlockMergeSortDecl
CoopBlockMergeSortPairsDecl = _decls_block_merge_sort.CoopBlockMergeSortPairsDecl
CoopBlockMergeSortInstanceType = _decls_block_merge_sort.CoopBlockMergeSortInstanceType
CoopBlockMergeSortPairsInstanceType = (
    _decls_block_merge_sort.CoopBlockMergeSortPairsInstanceType
)
block_merge_sort_instance_type = _decls_block_merge_sort.block_merge_sort_instance_type
block_merge_sort_pairs_instance_type = (
    _decls_block_merge_sort.block_merge_sort_pairs_instance_type
)


_decls_block_adjacent_difference = __import__(
    "cuda.coop._decls.block._block_adjacent_difference",
    fromlist=[
        "CoopBlockAdjacentDifferenceDecl",
        "CoopBlockAdjacentDifferenceInstanceType",
        "block_adjacent_difference_instance_type",
    ],
)
CoopBlockAdjacentDifferenceDecl = (
    _decls_block_adjacent_difference.CoopBlockAdjacentDifferenceDecl
)
CoopBlockAdjacentDifferenceInstanceType = (
    _decls_block_adjacent_difference.CoopBlockAdjacentDifferenceInstanceType
)
block_adjacent_difference_instance_type = (
    _decls_block_adjacent_difference.block_adjacent_difference_instance_type
)


_decls_block_shuffle = __import__(
    "cuda.coop._decls.block._block_shuffle",
    fromlist=[
        "CoopBlockShuffleDecl",
        "CoopBlockShuffleInstanceType",
        "block_shuffle_instance_type",
    ],
)
CoopBlockShuffleDecl = _decls_block_shuffle.CoopBlockShuffleDecl
CoopBlockShuffleInstanceType = _decls_block_shuffle.CoopBlockShuffleInstanceType
block_shuffle_instance_type = _decls_block_shuffle.block_shuffle_instance_type


_decls_block_discontinuity = __import__(
    "cuda.coop._decls.block._block_discontinuity",
    fromlist=[
        "CoopBlockDiscontinuityDecl",
        "CoopBlockDiscontinuityInstanceType",
        "block_discontinuity_instance_type",
    ],
)
CoopBlockDiscontinuityDecl = _decls_block_discontinuity.CoopBlockDiscontinuityDecl
CoopBlockDiscontinuityInstanceType = (
    _decls_block_discontinuity.CoopBlockDiscontinuityInstanceType
)
block_discontinuity_instance_type = (
    _decls_block_discontinuity.block_discontinuity_instance_type
)


_decls_block_radix_sort = __import__(
    "cuda.coop._decls.block._block_radix_sort",
    fromlist=[
        "CoopBlockRadixSortDecl",
        "CoopBlockRadixSortDescendingDecl",
        "CoopBlockRadixSortInstanceType",
        "CoopBlockRadixSortDescendingInstanceType",
        "block_radix_sort_instance_type",
        "block_radix_sort_descending_instance_type",
    ],
)
CoopBlockRadixSortDecl = _decls_block_radix_sort.CoopBlockRadixSortDecl
CoopBlockRadixSortDescendingDecl = (
    _decls_block_radix_sort.CoopBlockRadixSortDescendingDecl
)
CoopBlockRadixSortInstanceType = _decls_block_radix_sort.CoopBlockRadixSortInstanceType
CoopBlockRadixSortDescendingInstanceType = (
    _decls_block_radix_sort.CoopBlockRadixSortDescendingInstanceType
)
block_radix_sort_instance_type = _decls_block_radix_sort.block_radix_sort_instance_type
block_radix_sort_descending_instance_type = (
    _decls_block_radix_sort.block_radix_sort_descending_instance_type
)


_decls_block_radix_rank = __import__(
    "cuda.coop._decls.block._block_radix_rank",
    fromlist=[
        "CoopBlockRadixRankDecl",
        "CoopBlockRadixRankInstanceType",
        "block_radix_rank_instance_type",
    ],
)
CoopBlockRadixRankDecl = _decls_block_radix_rank.CoopBlockRadixRankDecl
CoopBlockRadixRankInstanceType = _decls_block_radix_rank.CoopBlockRadixRankInstanceType
block_radix_rank_instance_type = _decls_block_radix_rank.block_radix_rank_instance_type


# =============================================================================
# Warp Primitives (Single-phase)
# =============================================================================


# =============================================================================
# Warp Primitives (Two-phase Instances)
# =============================================================================


_decls_warp_load_store = __import__(
    "cuda.coop._decls.warp._warp_load_store",
    fromlist=[
        "CoopWarpLoadDecl",
        "CoopWarpLoadInstanceType",
        "CoopWarpStoreDecl",
        "CoopWarpStoreInstanceType",
        "warp_load_instance_type",
        "warp_store_instance_type",
    ],
)
CoopWarpLoadDecl = _decls_warp_load_store.CoopWarpLoadDecl
CoopWarpStoreDecl = _decls_warp_load_store.CoopWarpStoreDecl
CoopWarpLoadInstanceType = _decls_warp_load_store.CoopWarpLoadInstanceType
CoopWarpStoreInstanceType = _decls_warp_load_store.CoopWarpStoreInstanceType
warp_load_instance_type = _decls_warp_load_store.warp_load_instance_type
warp_store_instance_type = _decls_warp_load_store.warp_store_instance_type


_decls_warp_exchange = __import__(
    "cuda.coop._decls.warp._warp_exchange",
    fromlist=[
        "CoopWarpExchangeDecl",
        "CoopWarpExchangeInstanceType",
        "warp_exchange_instance_type",
    ],
)
CoopWarpExchangeDecl = _decls_warp_exchange.CoopWarpExchangeDecl
CoopWarpExchangeInstanceType = _decls_warp_exchange.CoopWarpExchangeInstanceType
warp_exchange_instance_type = _decls_warp_exchange.warp_exchange_instance_type


_decls_warp_reduce = __import__(
    "cuda.coop._decls.warp._warp_reduce",
    fromlist=[
        "CoopWarpReduceDecl",
        "CoopWarpReduceInstanceType",
        "CoopWarpSumDecl",
        "CoopWarpSumInstanceType",
        "warp_reduce_instance_type",
        "warp_sum_instance_type",
    ],
)
CoopWarpReduceDecl = _decls_warp_reduce.CoopWarpReduceDecl
CoopWarpReduceInstanceType = _decls_warp_reduce.CoopWarpReduceInstanceType
CoopWarpSumDecl = _decls_warp_reduce.CoopWarpSumDecl
CoopWarpSumInstanceType = _decls_warp_reduce.CoopWarpSumInstanceType
warp_reduce_instance_type = _decls_warp_reduce.warp_reduce_instance_type
warp_sum_instance_type = _decls_warp_reduce.warp_sum_instance_type


_decls_warp_scan = __import__(
    "cuda.coop._decls.warp._warp_scan",
    fromlist=[
        "CoopWarpExclusiveScanDecl",
        "CoopWarpExclusiveScanInstanceType",
        "CoopWarpExclusiveSumDecl",
        "CoopWarpExclusiveSumInstanceType",
        "CoopWarpInclusiveScanDecl",
        "CoopWarpInclusiveScanInstanceType",
        "CoopWarpInclusiveSumDecl",
        "CoopWarpInclusiveSumInstanceType",
        "warp_exclusive_scan_instance_type",
        "warp_exclusive_sum_instance_type",
        "warp_inclusive_scan_instance_type",
        "warp_inclusive_sum_instance_type",
    ],
)
CoopWarpExclusiveScanDecl = _decls_warp_scan.CoopWarpExclusiveScanDecl
CoopWarpExclusiveScanInstanceType = _decls_warp_scan.CoopWarpExclusiveScanInstanceType
CoopWarpExclusiveSumDecl = _decls_warp_scan.CoopWarpExclusiveSumDecl
CoopWarpExclusiveSumInstanceType = _decls_warp_scan.CoopWarpExclusiveSumInstanceType
CoopWarpInclusiveScanDecl = _decls_warp_scan.CoopWarpInclusiveScanDecl
CoopWarpInclusiveScanInstanceType = _decls_warp_scan.CoopWarpInclusiveScanInstanceType
CoopWarpInclusiveSumDecl = _decls_warp_scan.CoopWarpInclusiveSumDecl
CoopWarpInclusiveSumInstanceType = _decls_warp_scan.CoopWarpInclusiveSumInstanceType
warp_exclusive_scan_instance_type = _decls_warp_scan.warp_exclusive_scan_instance_type
warp_exclusive_sum_instance_type = _decls_warp_scan.warp_exclusive_sum_instance_type
warp_inclusive_scan_instance_type = _decls_warp_scan.warp_inclusive_scan_instance_type
warp_inclusive_sum_instance_type = _decls_warp_scan.warp_inclusive_sum_instance_type


_decls_warp_merge_sort = __import__(
    "cuda.coop._decls.warp._warp_merge_sort",
    fromlist=[
        "CoopWarpMergeSortDecl",
        "CoopWarpMergeSortInstanceType",
        "CoopWarpMergeSortPairsDecl",
        "CoopWarpMergeSortPairsInstanceType",
        "warp_merge_sort_instance_type",
        "warp_merge_sort_pairs_instance_type",
    ],
)
CoopWarpMergeSortDecl = _decls_warp_merge_sort.CoopWarpMergeSortDecl
CoopWarpMergeSortInstanceType = _decls_warp_merge_sort.CoopWarpMergeSortInstanceType
CoopWarpMergeSortPairsDecl = _decls_warp_merge_sort.CoopWarpMergeSortPairsDecl
CoopWarpMergeSortPairsInstanceType = (
    _decls_warp_merge_sort.CoopWarpMergeSortPairsInstanceType
)
warp_merge_sort_instance_type = _decls_warp_merge_sort.warp_merge_sort_instance_type
warp_merge_sort_pairs_instance_type = (
    _decls_warp_merge_sort.warp_merge_sort_pairs_instance_type
)


_INVOCABLE_PRIMITIVE_TYPE_TO_INSTANCE_TYPE = {
    coop.block.load: block_load_instance_type,
    coop.block.store: block_store_instance_type,
    coop.block.exchange: block_exchange_instance_type,
    coop.block.merge_sort_keys: block_merge_sort_instance_type,
    coop.block.merge_sort_pairs: block_merge_sort_pairs_instance_type,
    coop.block.radix_sort_keys: block_radix_sort_instance_type,
    coop.block.radix_sort_pairs: block_radix_sort_instance_type,
    coop.block.radix_sort_keys_descending: block_radix_sort_descending_instance_type,
    coop.block.radix_sort_pairs_descending: block_radix_sort_descending_instance_type,
    coop.block.radix_rank: block_radix_rank_instance_type,
    coop.block.reduce: block_reduce_instance_type,
    coop.block.sum: block_sum_instance_type,
    coop.block.scan: block_scan_instance_type,
    coop.block.exclusive_sum: block_scan_instance_type,
    coop.block.inclusive_sum: block_scan_instance_type,
    coop.block.exclusive_scan: block_scan_instance_type,
    coop.block.inclusive_scan: block_scan_instance_type,
    coop.block.adjacent_difference: block_adjacent_difference_instance_type,
    coop.block.discontinuity: block_discontinuity_instance_type,
    coop.block.shuffle: block_shuffle_instance_type,
    coop.warp.load: warp_load_instance_type,
    coop.warp.store: warp_store_instance_type,
    coop.warp.exchange: warp_exchange_instance_type,
    coop.warp.reduce: warp_reduce_instance_type,
    coop.warp.sum: warp_sum_instance_type,
    coop.warp.exclusive_sum: warp_exclusive_sum_instance_type,
    coop.warp.inclusive_sum: warp_inclusive_sum_instance_type,
    coop.warp.exclusive_scan: warp_exclusive_scan_instance_type,
    coop.warp.inclusive_scan: warp_inclusive_scan_instance_type,
    coop.warp.merge_sort_keys: warp_merge_sort_instance_type,
    coop.warp.merge_sort_pairs: warp_merge_sort_pairs_instance_type,
}


@typeof_impl.register(Invocable)
def typeof_invocable_instance(val, c):
    specialization = getattr(val, "specialization", None)
    if specialization is None:
        return None

    primitive = getattr(specialization, "primitive", None)
    if primitive is None:
        return None

    primitive_type = type(primitive)
    return _INVOCABLE_PRIMITIVE_TYPE_TO_INSTANCE_TYPE.get(primitive_type)


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

    def resolve_exchange(self, mod):
        return types.Function(CoopBlockExchangeDecl)

    def resolve_histogram(self, mod):
        return types.Function(CoopBlockHistogramDecl)

    def resolve_run_length(self, mod):
        return types.Function(CoopBlockRunLengthDecl)

    def resolve_reduce(self, mod):
        return types.Function(CoopBlockReduceDecl)

    def resolve_scan(self, mod):
        return types.Function(CoopBlockScanDecl)

    def resolve_exclusive_sum(self, mod):
        return types.Function(CoopBlockExclusiveSumDecl)

    def resolve_inclusive_sum(self, mod):
        return types.Function(CoopBlockInclusiveSumDecl)

    def resolve_exclusive_scan(self, mod):
        return types.Function(CoopBlockExclusiveScanDecl)

    def resolve_inclusive_scan(self, mod):
        return types.Function(CoopBlockInclusiveScanDecl)

    def resolve_sum(self, mod):
        return types.Function(CoopBlockSumDecl)

    def resolve_merge_sort_keys(self, mod):
        return types.Function(CoopBlockMergeSortDecl)

    def resolve_merge_sort_pairs(self, mod):
        return types.Function(CoopBlockMergeSortPairsDecl)

    def resolve_radix_sort_keys(self, mod):
        return types.Function(CoopBlockRadixSortDecl)

    def resolve_radix_sort_keys_descending(self, mod):
        return types.Function(CoopBlockRadixSortDescendingDecl)

    def resolve_radix_sort_pairs(self, mod):
        return types.Function(CoopBlockRadixSortDecl)

    def resolve_radix_sort_pairs_descending(self, mod):
        return types.Function(CoopBlockRadixSortDescendingDecl)


@register_attr
class CoopWarpModuleTemplate(AttributeTemplate):
    key = types.Module(coop.warp)

    def resolve_load(self, mod):
        return types.Function(CoopWarpLoadDecl)

    def resolve_store(self, mod):
        return types.Function(CoopWarpStoreDecl)

    def resolve_exchange(self, mod):
        return types.Function(CoopWarpExchangeDecl)

    def resolve_reduce(self, mod):
        return types.Function(CoopWarpReduceDecl)

    def resolve_sum(self, mod):
        return types.Function(CoopWarpSumDecl)

    def resolve_inclusive_sum(self, mod):
        return types.Function(CoopWarpInclusiveSumDecl)

    def resolve_exclusive_sum(self, mod):
        return types.Function(CoopWarpExclusiveSumDecl)

    def resolve_exclusive_scan(self, mod):
        return types.Function(CoopWarpExclusiveScanDecl)

    def resolve_inclusive_scan(self, mod):
        return types.Function(CoopWarpInclusiveScanDecl)

    def resolve_merge_sort_keys(self, mod):
        return types.Function(CoopWarpMergeSortDecl)

    def resolve_merge_sort_pairs(self, mod):
        return types.Function(CoopWarpMergeSortPairsDecl)


@register_attr
class CoopModuleTemplate(AttributeTemplate):
    key = types.Module(coop)

    def resolve_block(self, mod):
        return types.Module(coop.block)

    def resolve_warp(self, mod):
        return types.Module(coop.warp)

    def resolve_local(self, mod):
        return types.Module(coop.local)

    def resolve_shared(self, mod):
        return types.Module(coop.shared)

    def resolve_BlockLoadAlgorithm(self, mod):
        return types.Module(coop.BlockLoadAlgorithm)

    def resolve_BlockStoreAlgorithm(self, mod):
        return types.Module(coop.BlockStoreAlgorithm)

    def resolve_BlockScanAlgorithm(self, mod):
        return types.Module(coop.BlockScanAlgorithm)

    def resolve_WarpLoadAlgorithm(self, mod):
        return types.Module(coop.WarpLoadAlgorithm)

    def resolve_WarpStoreAlgorithm(self, mod):
        return types.Module(coop.WarpStoreAlgorithm)

    def resolve_BlockHistogramAlgorithm(self, mod):
        return types.Module(coop.BlockHistogramAlgorithm)

    def resolve_TempStorage(self, mod):
        return types.Function(CoopTempStorageDecl)

    def resolve_ThreadData(self, mod):
        return types.Function(CoopThreadDataDecl)


register_global(coop, types.Module(coop))
