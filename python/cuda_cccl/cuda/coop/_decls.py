# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is semantically equivalent to the numba.cuda.cudadecl module.
# It is responsible for defining the Numba templates for cuda.coop
# primitives.

import enum
import inspect
import operator
from typing import (
    Any,
    Callable,
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
    infer_global,
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
    lower_builtin,
    models,
    register_model,
    typeof_impl,
)

import cuda.coop as coop

from ._common import CUB_BLOCK_REDUCE_ALGOS
from ._scan_op import ScanOp
from ._types import Invocable
from ._typing import (
    ScanOpType,
)
from .block._block_adjacent_difference import (
    _make_adjacent_difference_rewrite as _make_block_adjacent_difference_rewrite,
)
from .block._block_discontinuity import (
    _make_discontinuity_rewrite as _make_block_discontinuity_rewrite,
)
from .block._block_exchange import (
    _make_exchange_rewrite as _make_block_exchange_rewrite,
)
from .block._block_load_store import (
    _make_load_rewrite as _make_block_load_rewrite,
)
from .block._block_load_store import (
    _make_store_rewrite as _make_block_store_rewrite,
)
from .block._block_merge_sort import (
    _make_merge_sort_keys_rewrite as _make_block_merge_sort_keys_rewrite,
)
from .block._block_merge_sort import (
    _make_merge_sort_pairs_rewrite as _make_block_merge_sort_pairs_rewrite,
)
from .block._block_radix_rank import (
    _make_radix_rank_rewrite as _make_block_radix_rank_rewrite,
)
from .block._block_radix_sort import (
    _make_radix_sort_keys_descending_rewrite as _make_block_radix_sort_keys_descending_rewrite,
)
from .block._block_radix_sort import (
    _make_radix_sort_keys_rewrite as _make_block_radix_sort_keys_rewrite,
)
from .block._block_reduce import (
    _make_reduce_rewrite as _make_block_reduce_rewrite,
)
from .block._block_reduce import (
    _make_sum_rewrite as _make_block_sum_rewrite,
)
from .block._block_scan import (
    _make_scan_rewrite as _make_block_scan_rewrite,
)
from .block._block_shuffle import (
    _make_shuffle_rewrite as _make_block_shuffle_rewrite,
)
from .warp._warp_exchange import _make_exchange_rewrite
from .warp._warp_load_store import (
    _make_load_rewrite,
    _make_store_rewrite,
)
from .warp._warp_merge_sort import (
    _make_merge_sort_keys_rewrite,
    _make_merge_sort_pairs_rewrite,
)
from .warp._warp_reduce import (
    _make_reduce_rewrite,
    _make_sum_rewrite,
)
from .warp._warp_scan import (
    _make_exclusive_scan_rewrite,
    _make_exclusive_sum_rewrite,
    _make_inclusive_scan_rewrite,
    _make_inclusive_sum_rewrite,
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
# obviating need to pass it explicitly to the load/store functions.


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


@register_global(coop.warp.load)
class CoopWarpLoadDecl(CoopWarpLoadStoreBaseTemplate, WarpLoadMixin, CoopDeclMixin):
    key = coop.warp.load
    impl_key = _make_load_rewrite
    primitive_name = "coop.warp.load"
    algorithm_enum = coop.WarpLoadAlgorithm
    default_algorithm = coop.WarpLoadAlgorithm.DIRECT


@register_global(coop.warp.store)
class CoopWarpStoreDecl(CoopWarpLoadStoreBaseTemplate, WarpStoreMixin, CoopDeclMixin):
    key = coop.warp.store
    impl_key = _make_store_rewrite
    primitive_name = "coop.warp.store"
    algorithm_enum = coop.WarpStoreAlgorithm
    default_algorithm = coop.WarpStoreAlgorithm.DIRECT


# =============================================================================
# Exchange
# =============================================================================


@register_global(coop.block.exchange)
class CoopBlockExchangeDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.exchange
    impl_key = _make_block_exchange_rewrite
    primitive_name = "coop.block.exchange"
    is_constructor = False
    minimum_num_args = 1
    default_exchange_type = coop.block.BlockExchangeType.StripedToBlocked

    @staticmethod
    def signature(
        items: types.Array,
        output_items: types.Array = None,
        items_per_thread: int = None,
        ranks: types.Array = None,
        valid_flags: types.Array = None,
        block_exchange_type: coop.block.BlockExchangeType = None,
        warp_time_slicing: bool = False,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockExchangeDecl.signature).bind(
            items,
            output_items=output_items,
            items_per_thread=items_per_thread,
            ranks=ranks,
            valid_flags=valid_flags,
            block_exchange_type=block_exchange_type,
            warp_time_slicing=warp_time_slicing,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        items: types.Array,
        output_items: types.Array = None,
        ranks: types.Array = None,
        valid_flags: types.Array = None,
        *,
        items_per_thread: int = None,
        block_exchange_type: coop.block.BlockExchangeType = None,
        warp_time_slicing: bool = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockExchangeDecl.signature_instance).bind(
            items,
            output_items=output_items,
            ranks=ranks,
            valid_flags=valid_flags,
            items_per_thread=items_per_thread,
            block_exchange_type=block_exchange_type,
            warp_time_slicing=warp_time_slicing,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments.get("output_items")
        ranks = bound.arguments.get("ranks")
        valid_flags = bound.arguments.get("valid_flags")
        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )

        using_thread_data = isinstance(items, ThreadDataType) or (
            output_items is not None and isinstance(output_items, ThreadDataType)
        )
        if output_items is not None:
            validate_src_dst(self, items, output_items)

        items_per_thread = bound.arguments.get("items_per_thread")
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        block_exchange_type = bound.arguments.get("block_exchange_type")
        block_exchange_is_none_type = isinstance(block_exchange_type, types.NoneType)
        if block_exchange_type is None or block_exchange_is_none_type:
            if not two_phase:
                block_exchange_type = self.default_exchange_type
            else:
                block_exchange_type = None
        if block_exchange_type is None:
            exchange_type_value = None
        elif isinstance(block_exchange_type, enum.IntEnum):
            if block_exchange_type not in coop.block.BlockExchangeType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_exchange_type' to be "
                    "a BlockExchangeType enum value"
                )
            exchange_type_value = block_exchange_type
        else:
            if not isinstance(block_exchange_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_exchange_type' to be "
                    "a BlockExchangeType enum value"
                )
            if block_exchange_type.instance_class is not coop.block.BlockExchangeType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_exchange_type' to be "
                    "a BlockExchangeType enum value"
                )
            exchange_type_value = None

        if exchange_type_value is None:
            if ranks is not None:
                if not isinstance(ranks, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be a device array"
                    )
                if not isinstance(ranks.dtype, types.Integer):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be an integer array"
                    )
            if valid_flags is not None:
                if not isinstance(valid_flags, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a device "
                        "array"
                    )
                if not isinstance(valid_flags.dtype, (types.Integer, types.Boolean)):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a "
                        "boolean or integer array"
                    )
        else:
            uses_ranks = exchange_type_value in (
                coop.block.BlockExchangeType.ScatterToBlocked,
                coop.block.BlockExchangeType.ScatterToStriped,
                coop.block.BlockExchangeType.ScatterToStripedGuarded,
                coop.block.BlockExchangeType.ScatterToStripedFlagged,
            )
            uses_valid_flags = (
                exchange_type_value
                == coop.block.BlockExchangeType.ScatterToStripedFlagged
            )

            if uses_ranks:
                if ranks is None:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' for scatter exchanges"
                    )
                if not isinstance(ranks, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be a device array"
                    )
                if not isinstance(ranks.dtype, types.Integer):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'ranks' to be an integer array"
                    )
            elif ranks is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'ranks' for "
                    f"{exchange_type_value.name}"
                )

            if uses_valid_flags:
                if valid_flags is None:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' for "
                        "ScatterToStripedFlagged"
                    )
                if not isinstance(valid_flags, types.Array):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a device "
                        "array"
                    )
                if not isinstance(valid_flags.dtype, (types.Integer, types.Boolean)):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'valid_flags' to be a "
                        "boolean or integer array"
                    )
            elif valid_flags is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'valid_flags' for "
                    f"{exchange_type_value.name}"
                )

        warp_time_slicing = bound.arguments.get("warp_time_slicing")
        warp_time_slicing_is_none_type = isinstance(warp_time_slicing, types.NoneType)
        if warp_time_slicing is None or warp_time_slicing_is_none_type:
            if not two_phase:
                warp_time_slicing = False
            else:
                warp_time_slicing = None
        if warp_time_slicing is not None and not isinstance(
            warp_time_slicing, (types.Boolean, types.BooleanLiteral, bool)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'warp_time_slicing' to be a boolean"
            )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        arglist = [items]
        if output_items is not None:
            arglist.append(output_items)
        if items_per_thread is not None:
            arglist.append(items_per_thread)
        if ranks is not None:
            arglist.append(ranks)
        if valid_flags is not None:
            arglist.append(valid_flags)
        if block_exchange_type is not None:
            arglist.append(block_exchange_type)
        if warp_time_slicing is not None:
            arglist.append(warp_time_slicing)
        if temp_storage is not None:
            arglist.append(temp_storage)

        sig = signature(types.void, *arglist)

        return sig


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
# Adjacent Difference
# =============================================================================


@register_global(coop.block.adjacent_difference)
class CoopBlockAdjacentDifferenceDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.adjacent_difference
    impl_key = _make_block_adjacent_difference_rewrite
    primitive_name = "coop.block.adjacent_difference"
    is_constructor = False
    minimum_num_args = 2
    default_difference_type = coop.block.BlockAdjacentDifferenceType.SubtractLeft

    @staticmethod
    def signature(
        items: types.Array,
        output_items: types.Array,
        items_per_thread: int = None,
        difference_op: Optional[Callable] = None,
        block_adjacent_difference_type: coop.block.BlockAdjacentDifferenceType = None,
        valid_items: Optional[int] = None,
        tile_predecessor_item: Optional[Any] = None,
        tile_successor_item: Optional[Any] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockAdjacentDifferenceDecl.signature).bind(
            items,
            output_items,
            items_per_thread=items_per_thread,
            difference_op=difference_op,
            block_adjacent_difference_type=block_adjacent_difference_type,
            valid_items=valid_items,
            tile_predecessor_item=tile_predecessor_item,
            tile_successor_item=tile_successor_item,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments["output_items"]

        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )
        if not isinstance(output_items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'output_items' to be a device or "
                "thread-data array"
            )

        using_thread_data = isinstance(items, ThreadDataType) or isinstance(
            output_items, ThreadDataType
        )
        if not using_thread_data:
            if isinstance(items, types.Array) and isinstance(output_items, types.Array):
                if items.dtype != output_items.dtype:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'items' and "
                        "'output_items' to have matching dtypes"
                    )

        items_per_thread = bound.arguments.get("items_per_thread")
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        block_adjacent_difference_type = bound.arguments.get(
            "block_adjacent_difference_type"
        )
        if block_adjacent_difference_type is None:
            block_adjacent_difference_type = self.default_difference_type
        if isinstance(block_adjacent_difference_type, enum.IntEnum):
            if (
                block_adjacent_difference_type
                not in coop.block.BlockAdjacentDifferenceType
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_adjacent_difference_type' "
                    "to be a BlockAdjacentDifferenceType enum value"
                )
        else:
            if not isinstance(block_adjacent_difference_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_adjacent_difference_type' "
                    "to be a BlockAdjacentDifferenceType enum value"
                )
            if (
                block_adjacent_difference_type.instance_class
                is not coop.block.BlockAdjacentDifferenceType
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_adjacent_difference_type' "
                    "to be a BlockAdjacentDifferenceType enum value"
                )

        difference_op = bound.arguments.get("difference_op")
        if difference_op is None:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'difference_op' to be specified"
            )

        valid_items = bound.arguments.get("valid_items")
        if valid_items is not None and not isinstance(
            valid_items, (types.Integer, types.IntegerLiteral)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'valid_items' to be an integer"
            )

        tile_predecessor_item = bound.arguments.get("tile_predecessor_item")
        tile_successor_item = bound.arguments.get("tile_successor_item")
        if tile_predecessor_item is not None and tile_successor_item is not None:
            raise errors.TypingError(
                f"{self.primitive_name} accepts only one of 'tile_predecessor_item' "
                "or 'tile_successor_item'"
            )
        if (
            block_adjacent_difference_type
            == coop.block.BlockAdjacentDifferenceType.SubtractLeft
            and tile_successor_item is not None
        ):
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'tile_successor_item' for "
                "SubtractLeft"
            )
        if (
            block_adjacent_difference_type
            == coop.block.BlockAdjacentDifferenceType.SubtractRight
            and tile_predecessor_item is not None
        ):
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'tile_predecessor_item' for "
                "SubtractRight"
            )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        arglist = [items, output_items]
        if items_per_thread is not None:
            arglist.append(items_per_thread)
        arglist.append(difference_op)
        if block_adjacent_difference_type is not None:
            arglist.append(block_adjacent_difference_type)
        if valid_items is not None:
            arglist.append(valid_items)
        if tile_predecessor_item is not None:
            arglist.append(tile_predecessor_item)
        if tile_successor_item is not None:
            arglist.append(tile_successor_item)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


# =============================================================================
# Shuffle
# =============================================================================


@register_global(coop.block.shuffle)
class CoopBlockShuffleDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.shuffle
    impl_key = _make_block_shuffle_rewrite
    primitive_name = "coop.block.shuffle"
    is_constructor = False
    minimum_num_args = 1
    default_shuffle_type = coop.block.BlockShuffleType.Up

    @staticmethod
    def signature(
        items: Union[types.Array, types.Number],
        output_items: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        block_shuffle_type: coop.block.BlockShuffleType = None,
        distance: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
        block_prefix: types.Array = None,
        block_suffix: types.Array = None,
    ):
        return inspect.signature(CoopBlockShuffleDecl.signature).bind(
            items,
            output_items=output_items,
            items_per_thread=items_per_thread,
            block_shuffle_type=block_shuffle_type,
            distance=distance,
            temp_storage=temp_storage,
            block_prefix=block_prefix,
            block_suffix=block_suffix,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments.get("output_items")

        items_is_array = isinstance(items, (types.Array, ThreadDataType))
        items_is_scalar = isinstance(items, types.Number)

        if not items_is_array and not items_is_scalar:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a scalar or array"
            )

        if items_is_scalar and output_items is not None:
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'output_items' for scalar "
                "shuffle operations"
            )

        if items_is_array:
            if not isinstance(output_items, (types.Array, ThreadDataType)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'output_items' to be a device "
                    "or thread-data array for Up/Down shuffles"
                )

            if isinstance(items, types.Array) and isinstance(output_items, types.Array):
                if items.dtype != output_items.dtype:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires 'items' and "
                        "'output_items' to have matching dtypes"
                    )

        block_shuffle_type = bound.arguments.get("block_shuffle_type")
        if block_shuffle_type is None:
            if items_is_scalar:
                block_shuffle_type = coop.block.BlockShuffleType.Offset
            else:
                block_shuffle_type = self.default_shuffle_type

        block_shuffle_type_value = None
        if isinstance(block_shuffle_type, enum.IntEnum):
            if block_shuffle_type not in coop.block.BlockShuffleType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_shuffle_type' to be a "
                    "BlockShuffleType enum value"
                )
            block_shuffle_type_value = block_shuffle_type
        else:
            if not isinstance(block_shuffle_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_shuffle_type' to be a "
                    "BlockShuffleType enum value"
                )
            if block_shuffle_type.instance_class is not coop.block.BlockShuffleType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_shuffle_type' to be a "
                    "BlockShuffleType enum value"
                )

        array_shuffle = items_is_array
        scalar_shuffle = items_is_scalar
        if block_shuffle_type_value is not None:
            array_shuffle = block_shuffle_type_value in (
                coop.block.BlockShuffleType.Up,
                coop.block.BlockShuffleType.Down,
            )
            scalar_shuffle = block_shuffle_type_value in (
                coop.block.BlockShuffleType.Offset,
                coop.block.BlockShuffleType.Rotate,
                coop.block.BlockShuffleType.Up,
                coop.block.BlockShuffleType.Down,
            )

            if items_is_scalar and not scalar_shuffle:
                raise errors.TypingError(
                    f"{self.primitive_name} requires a valid BlockShuffleType for "
                    "scalar shuffles"
                )
            if items_is_array and not array_shuffle:
                raise errors.TypingError(
                    f"{self.primitive_name} requires Up or Down for array shuffles"
                )

        items_per_thread = bound.arguments.get("items_per_thread")
        if items_is_array:
            if not isinstance(items, ThreadDataType) and not isinstance(
                output_items, ThreadDataType
            ):
                if not two_phase or items_per_thread is not None:
                    items_per_thread = validate_items_per_thread(self, items_per_thread)

        distance = bound.arguments.get("distance")
        if distance is not None and scalar_shuffle:
            if not isinstance(distance, (types.Integer, types.IntegerLiteral)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'distance' to be an integer"
                )
        if distance is not None and array_shuffle:
            raise errors.TypingError(
                f"{self.primitive_name} does not accept 'distance' for Up/Down shuffles"
            )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        block_prefix = bound.arguments.get("block_prefix")
        block_suffix = bound.arguments.get("block_suffix")

        if block_prefix is not None or block_suffix is not None:
            if not items_is_array:
                raise errors.TypingError(
                    f"{self.primitive_name} only supports block_prefix/block_suffix "
                    "for Up/Down shuffles with array inputs"
                )
            if block_prefix is not None and block_suffix is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not allow block_prefix and "
                    "block_suffix together"
                )
            if block_shuffle_type_value is not None:
                if block_shuffle_type_value == coop.block.BlockShuffleType.Up:
                    if block_prefix is not None:
                        raise errors.TypingError(
                            f"{self.primitive_name} does not allow block_prefix for "
                            "Up shuffles"
                        )
                if block_shuffle_type_value == coop.block.BlockShuffleType.Down:
                    if block_suffix is not None:
                        raise errors.TypingError(
                            f"{self.primitive_name} does not allow block_suffix for "
                            "Down shuffles"
                        )
            if block_prefix is not None and not isinstance(block_prefix, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_prefix to be a device array"
                )
            if block_suffix is not None and not isinstance(block_suffix, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_suffix to be a device array"
                )

            item_dtype = items.dtype if isinstance(items, types.Array) else None
            if ThreadDataType is not None and isinstance(items, ThreadDataType):
                item_dtype = items.dtype

            if item_dtype is not None:
                if (
                    block_prefix is not None
                    and isinstance(block_prefix, types.Array)
                    and block_prefix.dtype != item_dtype
                ):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires block_prefix to have the "
                        "same dtype as items"
                    )
                if (
                    block_suffix is not None
                    and isinstance(block_suffix, types.Array)
                    and block_suffix.dtype != item_dtype
                ):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires block_suffix to have the "
                        "same dtype as items"
                    )

        if items_is_array:
            arglist = [items, output_items]
            if items_per_thread is not None:
                arglist.append(items_per_thread)
            arglist.append(block_shuffle_type)
            if block_prefix is not None:
                arglist.append(block_prefix)
            if block_suffix is not None:
                arglist.append(block_suffix)
            if temp_storage is not None:
                arglist.append(temp_storage)
            return signature(types.void, *arglist)

        arglist = [items]
        if distance is not None:
            arglist.append(distance)
        arglist.append(block_shuffle_type)
        if temp_storage is not None:
            arglist.append(temp_storage)
        return signature(items, *arglist)


# =============================================================================
# Discontinuity
# =============================================================================


@register_global(coop.block.discontinuity)
class CoopBlockDiscontinuityDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.discontinuity
    impl_key = _make_block_discontinuity_rewrite
    primitive_name = "coop.block.discontinuity"
    is_constructor = False
    minimum_num_args = 2
    default_discontinuity_type = coop.block.BlockDiscontinuityType.HEADS

    @staticmethod
    def signature(
        items: types.Array,
        head_flags: types.Array,
        tail_flags: types.Array = None,
        items_per_thread: int = None,
        flag_op: Optional[Callable] = None,
        block_discontinuity_type: coop.block.BlockDiscontinuityType = None,
        tile_predecessor_item: Optional[Any] = None,
        tile_successor_item: Optional[Any] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockDiscontinuityDecl.signature).bind(
            items,
            head_flags,
            tail_flags=tail_flags,
            items_per_thread=items_per_thread,
            flag_op=flag_op,
            block_discontinuity_type=block_discontinuity_type,
            tile_predecessor_item=tile_predecessor_item,
            tile_successor_item=tile_successor_item,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        head_flags = bound.arguments["head_flags"]
        tail_flags = bound.arguments.get("tail_flags")

        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )
        if not isinstance(head_flags, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'head_flags' to be a device or "
                "thread-data array"
            )
        if tail_flags is not None and not isinstance(
            tail_flags, (types.Array, ThreadDataType)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tail_flags' to be a device or "
                "thread-data array"
            )

        using_thread_data = isinstance(items, ThreadDataType) or isinstance(
            head_flags, ThreadDataType
        )
        if tail_flags is not None and isinstance(tail_flags, ThreadDataType):
            using_thread_data = True

        items_per_thread = bound.arguments.get("items_per_thread")
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        block_discontinuity_type = bound.arguments.get("block_discontinuity_type")
        if block_discontinuity_type is None:
            block_discontinuity_type = self.default_discontinuity_type
        discontinuity_value = None
        if isinstance(block_discontinuity_type, enum.IntEnum):
            if block_discontinuity_type not in coop.block.BlockDiscontinuityType:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_discontinuity_type' "
                    "to be a BlockDiscontinuityType enum value"
                )
            discontinuity_value = block_discontinuity_type
        else:
            if not isinstance(block_discontinuity_type, types.EnumMember):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_discontinuity_type' "
                    "to be a BlockDiscontinuityType enum value"
                )
            if (
                block_discontinuity_type.instance_class
                is not coop.block.BlockDiscontinuityType
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'block_discontinuity_type' "
                    "to be a BlockDiscontinuityType enum value"
                )

        if (
            discontinuity_value == coop.block.BlockDiscontinuityType.HEADS_AND_TAILS
            and tail_flags is None
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tail_flags' for HEADS_AND_TAILS"
            )

        flag_op = bound.arguments.get("flag_op")
        if flag_op is None:
            raise errors.TypingError(
                f"{self.primitive_name} requires 'flag_op' to be specified"
            )

        tile_predecessor_item = bound.arguments.get("tile_predecessor_item")
        tile_successor_item = bound.arguments.get("tile_successor_item")
        if tile_predecessor_item is not None and isinstance(
            tile_predecessor_item, (types.Array, ThreadDataType)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tile_predecessor_item' to be a scalar"
            )
        if tile_successor_item is not None and isinstance(
            tile_successor_item, (types.Array, ThreadDataType)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'tile_successor_item' to be a scalar"
            )
        if discontinuity_value == coop.block.BlockDiscontinuityType.HEADS:
            if tile_successor_item is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'tile_successor_item' "
                    "for HEADS"
                )
        if discontinuity_value == coop.block.BlockDiscontinuityType.TAILS:
            if tile_predecessor_item is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not accept 'tile_predecessor_item' "
                    "for TAILS"
                )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)

        arglist = [items, head_flags]

        if tail_flags is not None:
            arglist.append(tail_flags)

        if items_per_thread is not None:
            arglist.append(items_per_thread)

        arglist.append(flag_op)

        if block_discontinuity_type is not None:
            arglist.append(block_discontinuity_type)

        if tile_predecessor_item is not None:
            arglist.append(tile_predecessor_item)
        if tile_successor_item is not None:
            arglist.append(tile_successor_item)

        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


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
# Radix Rank
# =============================================================================


@register_global(coop.block.radix_rank)
class CoopBlockRadixRankDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.radix_rank
    impl_key = _make_block_radix_rank_rewrite
    primitive_name = "coop.block.radix_rank"
    is_constructor = False
    minimum_num_args = 3

    @staticmethod
    def signature(
        items: types.Array,
        ranks: types.Array,
        items_per_thread: int = None,
        begin_bit: Optional[int] = None,
        end_bit: Optional[int] = None,
        descending: Optional[bool] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
        exclusive_digit_prefix: types.Array = None,
    ):
        return inspect.signature(CoopBlockRadixRankDecl.signature).bind(
            items,
            ranks,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            temp_storage=temp_storage,
            exclusive_digit_prefix=exclusive_digit_prefix,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        ranks = bound.arguments["ranks"]

        if not isinstance(items, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device or "
                "thread-data array"
            )
        if not isinstance(ranks, (types.Array, ThreadDataType)):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'ranks' to be a device or "
                "thread-data array"
            )

        if isinstance(items, types.Array) and isinstance(ranks, types.Array):
            if items.dtype.signed:
                raise errors.TypingError(
                    f"{self.primitive_name} requires unsigned integer item types"
                )
            if not isinstance(ranks.dtype, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires integer 'ranks' arrays"
                )
            if ranks.dtype.bitwidth != 32:
                raise errors.TypingError(
                    f"{self.primitive_name} requires int32 ranks arrays"
                )

        items_per_thread = bound.arguments.get("items_per_thread")
        using_thread_data = isinstance(items, ThreadDataType) or isinstance(
            ranks, ThreadDataType
        )
        if not using_thread_data:
            if not two_phase or items_per_thread is not None:
                items_per_thread = validate_items_per_thread(self, items_per_thread)

        begin_bit = bound.arguments.get("begin_bit")
        end_bit = bound.arguments.get("end_bit")
        if begin_bit is None or end_bit is None:
            raise errors.TypingError(
                f"{self.primitive_name} requires begin_bit and end_bit"
            )
        if not isinstance(begin_bit, types.IntegerLiteral) or not isinstance(
            end_bit, types.IntegerLiteral
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires begin_bit and end_bit to be "
                "integer literals"
            )
        if end_bit.literal_value <= begin_bit.literal_value:
            raise errors.TypingError(
                f"{self.primitive_name} requires end_bit > begin_bit"
            )

        descending = bound.arguments.get("descending")
        if descending is not None and not isinstance(
            descending, (types.Boolean, types.BooleanLiteral, bool)
        ):
            raise errors.TypingError(
                f"{self.primitive_name} requires descending to be a boolean"
            )

        arglist = [items, ranks]
        if items_per_thread is not None:
            arglist.append(items_per_thread)
        arglist.extend([begin_bit, end_bit])
        if descending is not None:
            arglist.append(descending)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        exclusive_digit_prefix = bound.arguments.get("exclusive_digit_prefix")
        exclusive_prefix_is_none_type = isinstance(
            exclusive_digit_prefix, types.NoneType
        )
        if exclusive_prefix_is_none_type:
            arglist.append(exclusive_digit_prefix)
            exclusive_digit_prefix = None
        if not exclusive_prefix_is_none_type and exclusive_digit_prefix is not None:
            if not isinstance(exclusive_digit_prefix, (types.Array, ThreadDataType)):
                raise errors.TypingError(
                    f"{self.primitive_name} requires exclusive_digit_prefix to be "
                    "a device or thread-data array"
                )
            prefix_dtype = exclusive_digit_prefix.dtype
            if (
                not isinstance(prefix_dtype, types.Integer)
                or prefix_dtype.bitwidth != 32
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires exclusive_digit_prefix to be "
                    "an int32 array"
                )
            arglist.append(exclusive_digit_prefix)
        return signature(types.void, *arglist)


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
            from ._common import normalize_dtype_param

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


# =============================================================================
# Scan
# =============================================================================


@register
class CoopBlockScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.scan
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        initial_value: Optional[Any] = None,
        mode: Literal["exclusive", "inclusive"] = "exclusive",
        scan_op: ScanOpType = "+",
        block_prefix_callback_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        """
        This method defines the signature of the cooperative block scan
        function. It validates the parameters and returns a signature object.
        """
        return inspect.signature(CoopBlockScanDecl.signature).bind(
            src,
            dst,
            items_per_thread=items_per_thread,
            mode=mode,
            scan_op=scan_op,
            initial_value=initial_value,
            block_prefix_callback_op=block_prefix_callback_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        initial_value: Optional[Any] = None,
        *,
        items_per_thread: int = None,
        mode: Optional[Literal["exclusive", "inclusive"]] = None,
        scan_op: ScanOpType = None,
        block_prefix_callback_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockScanDecl.signature_instance).bind(
            src,
            dst,
            initial_value=initial_value,
            items_per_thread=items_per_thread,
            mode=mode,
            scan_op=scan_op,
            block_prefix_callback_op=block_prefix_callback_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def get_instance_type():
        return block_scan_instance_type

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        dst = bound.arguments.get("dst")

        src_is_array = isinstance(src, (types.Array, ThreadDataType))
        dst_is_array = isinstance(dst, (types.Array, ThreadDataType))
        src_is_scalar = isinstance(src, types.Number)
        dst_is_scalar = isinstance(dst, types.Number)

        scalar_return = dst is None and src_is_scalar

        if scalar_return:
            arglist = [src]
            pysig_params = [
                inspect.Parameter("src", inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            items_per_thread = bound.arguments.get("items_per_thread")
            items_per_thread_is_none_type = isinstance(items_per_thread, types.NoneType)
            if items_per_thread_is_none_type:
                arglist.append(items_per_thread)
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
                items_per_thread = None
            if not items_per_thread_is_none_type and items_per_thread is not None:
                maybe_literal = validate_items_per_thread(self, items_per_thread)
                if maybe_literal is not None:
                    items_per_thread = maybe_literal
                if isinstance(items_per_thread, types.IntegerLiteral):
                    if items_per_thread.literal_value != 1:
                        raise errors.TypingError(
                            f"{self.primitive_name} requires items_per_thread == 1 "
                            "for scalar inputs"
                        )
                elif isinstance(items_per_thread, types.Integer):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires items_per_thread to be a "
                        "compile-time literal for scalar inputs"
                    )
                arglist.append(items_per_thread)
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
        else:
            if src_is_scalar or dst_is_scalar:
                raise errors.TypingError(
                    f"{self.primitive_name} requires scalar inputs to omit dst"
                )
            if not (src_is_array and dst_is_array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires src and dst to be both arrays "
                    "or src-only scalar input"
                )
            validate_src_dst(self, src, dst)
            arglist = [src, dst]
            pysig_params = [
                inspect.Parameter("src", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("dst", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]

        if not scalar_return:
            items_per_thread = bound.arguments.get("items_per_thread")
            items_per_thread_is_none_type = isinstance(items_per_thread, types.NoneType)
            if items_per_thread_is_none_type:
                arglist.append(items_per_thread)
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
                items_per_thread = None
            if not items_per_thread_is_none_type and not (
                two_phase and items_per_thread is None
            ):
                process_items_per_thread(
                    self,
                    bound,
                    arglist,
                    two_phase,
                    target_array=(src, dst),
                )
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        initial_value = bound.arguments.get("initial_value")
        initial_value_is_none_type = isinstance(initial_value, types.NoneType)
        if initial_value_is_none_type:
            arglist.append(initial_value)
            pysig_params.append(
                inspect.Parameter(
                    "initial_value",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            initial_value = None
        if not initial_value_is_none_type:
            if isinstance(initial_value, types.IntegerLiteral):
                # If initial_value is an IntegerLiteral, we can use it directly.
                initial_value = initial_value.literal_value
            if initial_value is not None:
                arglist.append(initial_value)
                pysig_params.append(
                    inspect.Parameter(
                        "initial_value",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        mode = bound.arguments.get("mode")
        mode_is_none_type = isinstance(mode, types.NoneType)
        if mode_is_none_type:
            arglist.append(mode)
            pysig_params.append(
                inspect.Parameter(
                    "mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                )
            )
            mode = None
        if not mode_is_none_type:
            if mode is None:
                if not two_phase:
                    mode = "exclusive"
            elif isinstance(mode, types.StringLiteral):
                mode = mode.literal_value
            if mode is not None:
                if mode not in ("inclusive", "exclusive"):
                    raise errors.TypingError(
                        f"Invalid mode '{mode}' for {self.primitive_name}; expected "
                        "'inclusive' or 'exclusive'"
                    )
                arglist.append(mode)
                pysig_params.append(
                    inspect.Parameter(
                        "mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    )
                )

        scan_op = bound.arguments.get("scan_op")
        scan_op_is_none_type = isinstance(scan_op, types.NoneType)
        if scan_op_is_none_type:
            arglist.append(scan_op)
            pysig_params.append(
                inspect.Parameter(
                    "scan_op", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                )
            )
            scan_op = None
        if not scan_op_is_none_type:
            if scan_op is None:
                if not two_phase:
                    scan_op = "+"
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
                pysig_params.append(
                    inspect.Parameter(
                        "scan_op",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        block_prefix_callback_op = bound.arguments.get("block_prefix_callback_op")
        block_prefix_is_none_type = isinstance(block_prefix_callback_op, types.NoneType)
        if block_prefix_is_none_type:
            arglist.append(block_prefix_callback_op)
            pysig_params.append(
                inspect.Parameter(
                    "block_prefix_callback_op",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            block_prefix_callback_op = None
        if not block_prefix_is_none_type and block_prefix_callback_op is not None:
            # We can't do much validation here.
            arglist.append(block_prefix_callback_op)
            pysig_params.append(
                inspect.Parameter(
                    "block_prefix_callback_op",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )

        block_aggregate = bound.arguments.get("block_aggregate")
        block_aggregate_is_none_type = isinstance(block_aggregate, types.NoneType)
        if block_aggregate_is_none_type:
            arglist.append(block_aggregate)
            pysig_params.append(
                inspect.Parameter(
                    "block_aggregate",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            block_aggregate = None
        if not block_aggregate_is_none_type and block_aggregate is not None:
            if block_prefix_callback_op is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not support block_aggregate when "
                    "block_prefix_callback_op is provided"
                )
            if not isinstance(block_aggregate, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_aggregate to be a device "
                    "array"
                )
            if scalar_return:
                expected_dtype = src
            else:
                expected_dtype = src.dtype if isinstance(src, types.Array) else src
                if ThreadDataType is not None and isinstance(src, ThreadDataType):
                    expected_dtype = src.dtype
            if block_aggregate.dtype != expected_dtype:
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_aggregate to have the same "
                    "dtype as the input"
                )
            arglist.append(block_aggregate)
            pysig_params.append(
                inspect.Parameter(
                    "block_aggregate",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )

        algorithm = bound.arguments.get("algorithm")
        algorithm_is_none_type = isinstance(algorithm, types.NoneType)
        if algorithm_is_none_type:
            arglist.append(algorithm)
            pysig_params.append(
                inspect.Parameter(
                    "algorithm",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            algorithm = None
        if not algorithm_is_none_type:
            if algorithm is None and two_phase:
                # Use the algorithm baked into the two-phase instance.
                pass
            else:
                process_algorithm(self, bound, arglist)
                pysig_params.append(
                    inspect.Parameter(
                        "algorithm",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            arglist.append(temp_storage)
            pysig_params.append(
                inspect.Parameter(
                    "temp_storage",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            temp_storage = None
        if not temp_storage_is_none_type:
            validate_temp_storage(self, temp_storage)
            if temp_storage is not None:
                arglist.append(temp_storage)
                pysig_params.append(
                    inspect.Parameter(
                        "temp_storage",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        return_type = src if scalar_return else types.void
        pysig = inspect.Signature(pysig_params)
        sig = Signature(return_type, tuple(arglist), recvr=None, pysig=pysig)

        return sig


@register
class CoopBlockExclusiveSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.exclusive_sum
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "exclusive"
    forced_scan_op = "+"
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=None,
            mode="exclusive",
            scan_op="+",
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


@register
class CoopBlockInclusiveSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.inclusive_sum
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "inclusive"
    forced_scan_op = "+"
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=None,
            mode="inclusive",
            scan_op="+",
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


@register
class CoopBlockExclusiveScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.exclusive_scan
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "exclusive"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        scan_op: ScanOpType = None,
        initial_value: Optional[Any] = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode="exclusive",
            scan_op=scan_op,
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


@register
class CoopBlockInclusiveScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.inclusive_scan
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "inclusive"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        scan_op: ScanOpType = None,
        initial_value: Optional[Any] = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode="inclusive",
            scan_op=scan_op,
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


class CoopBlockScanInstanceType(types.Type, CoopInstanceTypeMixin):
    decl_class = CoopBlockScanDecl

    def __init__(self):
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(self, *args, **kwargs):
        bound = self._bind_instance_signature(*args, **kwargs)
        return self.decl._validate_args_and_create_signature(bound, two_phase=True)


block_scan_instance_type = CoopBlockScanInstanceType()


@typeof_impl.register(coop.block.scan)
def typeof_block_scan_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.exclusive_sum)
def typeof_block_exclusive_sum_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.inclusive_sum)
def typeof_block_inclusive_sum_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.exclusive_scan)
def typeof_block_exclusive_scan_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.inclusive_scan)
def typeof_block_inclusive_scan_instance(*args, **kwargs):
    return block_scan_instance_type


@register
class CoopBlockScanInstanceDecl(CoopInstanceTemplate):
    key = block_scan_instance_type
    instance_type = block_scan_instance_type
    primitive_name = "coop.block.scan"


@register_model(CoopBlockScanInstanceType)
class CoopBlockScanInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockScanInstanceType)
def lower_constant_block_scan_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


# =============================================================================
# Reduce
# =============================================================================


@register
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


# =============================================================================
# Block Primitives (Two-phase Instances)
# =============================================================================


class CoopBlockExchangeInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockExchangeDecl


block_exchange_instance_type = CoopBlockExchangeInstanceType()


@typeof_impl.register(coop.block.exchange)
def typeof_block_exchange_instance(*args, **kwargs):
    return block_exchange_instance_type


@register
class CoopBlockExchangeInstanceDecl(CoopInstanceTemplate):
    key = block_exchange_instance_type
    instance_type = block_exchange_instance_type
    primitive_name = "coop.block.exchange"


@register_model(CoopBlockExchangeInstanceType)
class CoopBlockExchangeInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockExchangeInstanceType)
def lower_constant_block_exchange_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


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


class CoopBlockAdjacentDifferenceInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockAdjacentDifferenceDecl


block_adjacent_difference_instance_type = CoopBlockAdjacentDifferenceInstanceType()


@typeof_impl.register(coop.block.adjacent_difference)
def typeof_block_adjacent_difference_instance(*args, **kwargs):
    return block_adjacent_difference_instance_type


@register
class CoopBlockAdjacentDifferenceInstanceDecl(CoopInstanceTemplate):
    key = block_adjacent_difference_instance_type
    instance_type = block_adjacent_difference_instance_type
    primitive_name = "coop.block.adjacent_difference"


@register_model(CoopBlockAdjacentDifferenceInstanceType)
class CoopBlockAdjacentDifferenceInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockAdjacentDifferenceInstanceType)
def lower_constant_block_adjacent_difference_instance_type(
    context, builder, typ, value
):
    return context.get_dummy_value()


class CoopBlockShuffleInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockShuffleDecl


block_shuffle_instance_type = CoopBlockShuffleInstanceType()


@typeof_impl.register(coop.block.shuffle)
def typeof_block_shuffle_instance(*args, **kwargs):
    return block_shuffle_instance_type


@register
class CoopBlockShuffleInstanceDecl(CoopInstanceTemplate):
    key = block_shuffle_instance_type
    instance_type = block_shuffle_instance_type
    primitive_name = "coop.block.shuffle"


@register_model(CoopBlockShuffleInstanceType)
class CoopBlockShuffleInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockShuffleInstanceType)
def lower_constant_block_shuffle_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopBlockDiscontinuityInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockDiscontinuityDecl


block_discontinuity_instance_type = CoopBlockDiscontinuityInstanceType()


@typeof_impl.register(coop.block.discontinuity)
def typeof_block_discontinuity_instance(*args, **kwargs):
    return block_discontinuity_instance_type


@register
class CoopBlockDiscontinuityInstanceDecl(CoopInstanceTemplate):
    key = block_discontinuity_instance_type
    instance_type = block_discontinuity_instance_type
    primitive_name = "coop.block.discontinuity"


@register_model(CoopBlockDiscontinuityInstanceType)
class CoopBlockDiscontinuityInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockDiscontinuityInstanceType)
def lower_constant_block_discontinuity_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


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


class CoopBlockRadixRankInstanceType(CoopSimpleInstanceType):
    decl_class = CoopBlockRadixRankDecl


block_radix_rank_instance_type = CoopBlockRadixRankInstanceType()


@typeof_impl.register(coop.block.radix_rank)
def typeof_block_radix_rank_instance(*args, **kwargs):
    return block_radix_rank_instance_type


@register
class CoopBlockRadixRankInstanceDecl(CoopInstanceTemplate):
    key = block_radix_rank_instance_type
    instance_type = block_radix_rank_instance_type
    primitive_name = "coop.block.radix_rank"


@register_model(CoopBlockRadixRankInstanceType)
class CoopBlockRadixRankInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockRadixRankInstanceType)
def lower_constant_block_radix_rank_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


# =============================================================================
# Warp Primitives (Single-phase)
# =============================================================================


@register_global(coop.warp.exchange)
class CoopWarpExchangeDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.exchange
    impl_key = _make_exchange_rewrite
    primitive_name = "coop.warp.exchange"
    is_constructor = False
    minimum_num_args = 1
    default_exchange_type = coop.warp.WarpExchangeType.StripedToBlocked

    @staticmethod
    def signature(
        items: types.Array,
        output_items: types.Array = None,
        items_per_thread: int = None,
        ranks: types.Array = None,
        warp_exchange_type: coop.warp.WarpExchangeType = None,
        threads_in_warp: int = 32,
        offset_dtype: Optional[types.Type] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpExchangeDecl.signature).bind(
            items,
            output_items=output_items,
            items_per_thread=items_per_thread,
            ranks=ranks,
            warp_exchange_type=warp_exchange_type,
            threads_in_warp=threads_in_warp,
            offset_dtype=offset_dtype,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        items: types.Array,
        output_items: types.Array = None,
        ranks: types.Array = None,
        *,
        items_per_thread: int = None,
        warp_exchange_type: coop.warp.WarpExchangeType = None,
        threads_in_warp: int = None,
        offset_dtype: Optional[types.Type] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpExchangeDecl.signature_instance).bind(
            items,
            output_items=output_items,
            ranks=ranks,
            items_per_thread=items_per_thread,
            warp_exchange_type=warp_exchange_type,
            threads_in_warp=threads_in_warp,
            offset_dtype=offset_dtype,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        output_items = bound.arguments.get("output_items")
        ranks = bound.arguments.get("ranks")
        if not isinstance(items, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device array"
            )
        if output_items is not None:
            validate_src_dst(self, items, output_items)

        arglist = [items]
        if output_items is not None:
            arglist.append(output_items)

        process_items_per_thread(self, bound, arglist, two_phase, target_array=items)

        warp_exchange_type = bound.arguments.get("warp_exchange_type")
        warp_exchange_is_none_type = isinstance(warp_exchange_type, types.NoneType)
        if warp_exchange_type is None or warp_exchange_is_none_type:
            if not two_phase:
                warp_exchange_type = self.default_exchange_type
            else:
                warp_exchange_type = None
        if warp_exchange_type is not None:
            if isinstance(warp_exchange_type, enum.IntEnum):
                if warp_exchange_type not in coop.warp.WarpExchangeType:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires a WarpExchangeType value"
                    )
            elif isinstance(warp_exchange_type, types.EnumMember):
                if warp_exchange_type.instance_class is not coop.warp.WarpExchangeType:
                    raise errors.TypingError(
                        f"{self.primitive_name} requires a WarpExchangeType value"
                    )
            else:
                raise errors.TypingError(
                    f"{self.primitive_name} requires a WarpExchangeType value"
                )
            arglist.append(warp_exchange_type)

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

        if warp_exchange_type == coop.warp.WarpExchangeType.ScatterToStriped:
            if ranks is None:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'ranks' for ScatterToStriped"
                )
        if ranks is not None:
            if not isinstance(ranks, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'ranks' to be an array"
                )
            if not isinstance(ranks.dtype, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'ranks' to be integer array"
                )
            arglist.append(ranks)

            offset_dtype = bound.arguments.get("offset_dtype")
            if offset_dtype is not None and not isinstance(
                offset_dtype, (types.DType, types.Type)
            ):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'offset_dtype' to be a dtype"
                )
            if offset_dtype is not None:
                arglist.append(offset_dtype)
        elif ranks is None and warp_exchange_type is not None:
            if warp_exchange_type != coop.warp.WarpExchangeType.ScatterToStriped:
                offset_dtype = bound.arguments.get("offset_dtype")
                if offset_dtype is not None:
                    raise errors.TypingError(
                        f"{self.primitive_name} only accepts 'offset_dtype' with 'ranks'"
                    )
        elif ranks is not None and warp_exchange_type is None and not two_phase:
            raise errors.TypingError(
                f"{self.primitive_name} only accepts 'ranks' for ScatterToStriped"
            )
        elif ranks is None and warp_exchange_type is None and not two_phase:
            offset_dtype = bound.arguments.get("offset_dtype")
            if offset_dtype is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} only accepts 'offset_dtype' with 'ranks'"
                )

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        return signature(types.void, *arglist)


@register_global(coop.warp.reduce)
class CoopWarpReduceDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.reduce
    impl_key = _make_reduce_rewrite
    primitive_name = "coop.warp.reduce"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: types.Number,
        binary_op: Optional[Callable] = None,
        threads_in_warp: int = 32,
        valid_items: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpReduceDecl.signature).bind(
            src,
            binary_op=binary_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: types.Number,
        *,
        binary_op: Optional[Callable] = None,
        threads_in_warp: int = None,
        valid_items: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpReduceDecl.signature_instance).bind(
            src,
            binary_op=binary_op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        if isinstance(src, types.Array):
            raise errors.TypingError(f"{self.primitive_name} requires a scalar input")
        if not isinstance(src, types.Number):
            raise errors.TypingError(f"{self.primitive_name} requires a numeric input")
        arglist = [src]

        binary_op = bound.arguments.get("binary_op")
        binary_op_is_none_type = isinstance(binary_op, types.NoneType)
        if binary_op is None or binary_op_is_none_type:
            if not two_phase:
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'binary_op' to be specified"
                )
        else:
            arglist.append(binary_op)

        threads_in_warp = bound.arguments.get("threads_in_warp")
        if threads_in_warp is not None:
            maybe_literal = validate_threads_in_warp(self, threads_in_warp)
            if maybe_literal is not None:
                threads_in_warp = maybe_literal
            arglist.append(threads_in_warp)

        valid_items = bound.arguments.get("valid_items")
        if valid_items is not None:
            if not isinstance(valid_items, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            arglist.append(valid_items)

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


@register_global(coop.warp.sum)
class CoopWarpSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.warp.sum
    impl_key = _make_sum_rewrite
    primitive_name = "coop.warp.sum"
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: types.Number,
        threads_in_warp: int = 32,
        valid_items: Optional[int] = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopWarpSumDecl.signature).bind(
            src,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
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

        valid_items = bound.arguments.get("valid_items")
        if valid_items is not None:
            if not isinstance(valid_items, types.Integer):
                raise errors.TypingError(
                    f"{self.primitive_name} requires 'valid_items' to be an integer"
                )
            arglist.append(valid_items)

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
# Warp Primitives (Two-phase Instances)
# =============================================================================


class CoopWarpLoadInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpLoadDecl


warp_load_instance_type = CoopWarpLoadInstanceType()


@typeof_impl.register(coop.warp.load)
def typeof_warp_load_instance(*args, **kwargs):
    return warp_load_instance_type


@register
class CoopWarpLoadInstanceDecl(CoopInstanceTemplate):
    key = warp_load_instance_type
    instance_type = warp_load_instance_type
    primitive_name = "coop.warp.load"


@register_model(CoopWarpLoadInstanceType)
class CoopWarpLoadInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpLoadInstanceType)
def lower_constant_warp_load_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpStoreInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpStoreDecl


warp_store_instance_type = CoopWarpStoreInstanceType()


@typeof_impl.register(coop.warp.store)
def typeof_warp_store_instance(*args, **kwargs):
    return warp_store_instance_type


@register
class CoopWarpStoreInstanceDecl(CoopInstanceTemplate):
    key = warp_store_instance_type
    instance_type = warp_store_instance_type
    primitive_name = "coop.warp.store"


@register_model(CoopWarpStoreInstanceType)
class CoopWarpStoreInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpStoreInstanceType)
def lower_constant_warp_store_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpExchangeInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpExchangeDecl


warp_exchange_instance_type = CoopWarpExchangeInstanceType()


@typeof_impl.register(coop.warp.exchange)
def typeof_warp_exchange_instance(*args, **kwargs):
    return warp_exchange_instance_type


@register
class CoopWarpExchangeInstanceDecl(CoopInstanceTemplate):
    key = warp_exchange_instance_type
    instance_type = warp_exchange_instance_type
    primitive_name = "coop.warp.exchange"


@register_model(CoopWarpExchangeInstanceType)
class CoopWarpExchangeInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpExchangeInstanceType)
def lower_constant_warp_exchange_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpReduceInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpReduceDecl


warp_reduce_instance_type = CoopWarpReduceInstanceType()


@typeof_impl.register(coop.warp.reduce)
def typeof_warp_reduce_instance(*args, **kwargs):
    return warp_reduce_instance_type


@register
class CoopWarpReduceInstanceDecl(CoopInstanceTemplate):
    key = warp_reduce_instance_type
    instance_type = warp_reduce_instance_type
    primitive_name = "coop.warp.reduce"


@register_model(CoopWarpReduceInstanceType)
class CoopWarpReduceInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpReduceInstanceType)
def lower_constant_warp_reduce_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpSumInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpSumDecl


warp_sum_instance_type = CoopWarpSumInstanceType()


@typeof_impl.register(coop.warp.sum)
def typeof_warp_sum_instance(*args, **kwargs):
    return warp_sum_instance_type


@register
class CoopWarpSumInstanceDecl(CoopInstanceTemplate):
    key = warp_sum_instance_type
    instance_type = warp_sum_instance_type
    primitive_name = "coop.warp.sum"


@register_model(CoopWarpSumInstanceType)
class CoopWarpSumInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpSumInstanceType)
def lower_constant_warp_sum_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


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
