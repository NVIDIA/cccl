# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is semantically equivalent to the numba.cuda.cudadecl module.
# It is responsible for defining the Numba templates for cuda.cooperative
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
    CallableTemplate,
    Registry,
    infer_global,
    signature,
)
from numba.extending import (
    lower_builtin,
    models,
    register_model,
    type_callable,
    typeof_impl,
)

import cuda.cccl.cooperative.experimental as coop

from ._scan_op import ScanOp
from ._typing import (
    ScanOpType,
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
    return {
        subclass: getattr(subclass, "key")
        for subclass in CoopDeclMixin.__subclasses__()
    }


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

    def _prevalidate_args(self, args):
        if len(args) < self.minimum_num_args:
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
        self._prevalidate_args(args)
        bound = self.signature(*args, **kwds)
        return self._validate_args_and_create_signature(bound)


# =============================================================================
# Temp Storage
# =============================================================================
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
            msg = f"temp_storage must be a {TempStorageType}, got {temp_storage}"
            raise errors.TypingError(msg)

        return signature(
            func_obj,
            (func_obj, temp_storage),
        )


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
# =============================================================================

# N.B. The upstream cuda.(local|shared).array() functions in numba-cuda don't
#      support being called with non-IntegerLiteral shapes, so we provide our
#      own versions for now that are more flexible.


class CoopArrayBaseTemplate(CoopAbstractTemplate):
    minimum_num_args = 2

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

        # Create the signature with the validated arguments.
        sig = signature(
            types.Array(dtype=nb_dtype, ndim=ndim, layout="C"),
            *arglist,
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


def process_items_per_thread(obj, bound, arglist, two_phase, target_array=None):
    items_per_thread = bound.arguments.get("items_per_thread")
    if items_per_thread is None:
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'items_per_thread' to be specified"
        )
    if target_array is not None:
        using_thread_data = isinstance(target_array, ThreadDataType)
    else:
        using_thread_data = False

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
    if temp_storage is not None and not isinstance(temp_storage, types.Array):
        raise errors.TypingError(
            f"{obj.primitive_name} requires 'temp_storage' to be a device array or None"
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
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(LoadMixin.signature).bind(
            src,
            dst,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
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


# Load
@register_global(coop.block.load)
class CoopBlockLoadDecl(CoopLoadStoreBaseTemplate, LoadMixin, CoopDeclMixin):
    key = coop.block.load
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
    primitive_name = "coop.block.store"
    algorithm_enum = coop.BlockStoreAlgorithm
    default_algorithm = coop.BlockStoreAlgorithm.DIRECT


# register(CoopBlockStoreDecl)


@infer_global(operator.getitem)
class CoopBlockStoreTempStorageGetItemDecl(CoopTempStorageGetItemDecl):
    target_key = coop.block.store
    target_template = CoopBlockStoreDecl


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

    def _validate_args_and_create_signature(
        self,
        src,
        dst,
        items_per_thread=None,
        num_valid_items=None,
        temp_storage=None,
    ):
        bound = inspect.signature(self.decl.signature).bind(
            src,
            dst,
            items_per_thread=items_per_thread,
            algorithm=None,
            num_valid_items=num_valid_items,
            temp_storage=temp_storage,
        )
        return self.decl._validate_args_and_create_signature(bound)


# Load
class CoopBlockLoadInstanceType(CoopLoadStoreInstanceBaseType, LoadMixin):
    decl_class = CoopBlockLoadDecl


block_load_instance_type = CoopBlockLoadInstanceType()


@typeof_impl.register(coop.block.load)
def typeof_block_load_instance(*args, **kwargs):
    return block_load_instance_type


@type_callable(block_load_instance_type)
def type_block_load_instance_call(context):
    instance = block_load_instance_type

    def typer(src, dst, items_per_thread=None, num_valid_items=None, temp_storage=None):
        return instance._validate_args_and_create_signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            num_valid_items=num_valid_items,
            temp_storage=temp_storage,
        )

    return typer


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


@type_callable(block_store_instance_type)
def type_block_store_instance_call(context):
    instance = block_store_instance_type

    def typer(dst, src, items_per_thread=None, num_valid_items=None, temp_storage=None):
        return instance._validate_args_and_create_signature(
            dst,
            src,
            items_per_thread=items_per_thread,
            num_valid_items=num_valid_items,
            temp_storage=temp_storage,
        )

    return typer


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
    def signature(*args, **kwargs):
        return inspect.signature(
            CoopBlockHistogramInitDecl.signature,
        ).bind()

    @staticmethod
    def get_instance_type():
        return block_histogram_instance_type

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        sig = signature(types.void, None)
        return sig


@register
class CoopBlockHistogramCompositeDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.histogram.composite
    primitive_name = "coop.block.histogram.composite"
    minimum_num_args = 1

    @staticmethod
    def signature(items: types.Array):
        return inspect.signature(
            CoopBlockHistogramCompositeDecl.signature,
        ).bind(items)

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        items = bound.arguments["items"]
        if not isinstance(items, types.Array):
            raise errors.TypingError(
                f"{self.primitive_name} requires 'items' to be a device array, "
                f"got {type(items).__name__}"
            )

        sig = signature(types.void, items)

        return sig


@register
class CoopBlockHistogramDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.histogram
    primitive_name = "coop.block.histogram"
    algorithm_enum = coop.BlockHistogramAlgorithm
    default_algorithm = coop.BlockHistogramAlgorithm.ATOMIC
    minimum_num_args = 2

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


block_histogram_instance_type = CoopBlockHistogramInstanceType()


@register_model(CoopBlockHistogramInstanceType)
class CoopBlockHistogramInstanceModel(models.OpaqueModel):
    pass


@typeof_impl.register(coop.block.histogram)
def typeof_block_histogram_instance(*args, **kwargs):
    return block_histogram_instance_type


@type_callable(block_histogram_instance_type)
def type_block_histogram_instance_call(context):
    # decl = block_histogram_instance_type.decl
    # return decl.generic()

    def typer(temp_storage=None):
        """
        This function is called to infer the type of the coop.block.histogram
        instance type. It checks that the parameters are of the expected types.
        """
        obj = block_histogram_instance_type.decl_class
        validate_temp_storage(obj, temp_storage)

        # Return the block histogram instance type.
        return block_histogram_instance_type

    return typer


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
class CoopBlockRunLengthDecodeDecl(CallableTemplate, CoopDeclMixin):
    key = coop.block.run_length.decode
    primitive_name = "coop.block.run_length.decode"

    unsafe_casting = False
    exact_match_required = True
    prefer_literal = True

    def generic(self):
        def typer(decoded_items, decoded_window_offset, relative_offsets=None):
            # Verify decoded_items is a device array.
            if not isinstance(decoded_items, types.Array):
                raise errors.TypingError(
                    "decoded_items must be a device array, "
                    f"got {type(decoded_items).__name__}"
                )

            arglist = [
                decoded_items,
            ]

            if decoded_window_offset is not None:
                if not isinstance(decoded_window_offset, types.Integer):
                    raise errors.TypingError(
                        "decoded_window_offset must be an integer value"
                    )
                arglist.append(decoded_window_offset)

            if relative_offsets is not None:
                # Verify relative_offsets is a device array.
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

        return typer


@register
class CoopBlockRunLengthDecl(CallableTemplate, CoopDeclMixin):
    key = coop.block.run_length
    primitive_name = "coop.block.run_length"
    algorithm_enum = coop.NoAlgorithm
    default_algorithm = coop.NoAlgorithm.NO_ALGORITHM
    decode_decl = CoopBlockRunLengthDecodeDecl
    is_constructor = True

    # unsafe_casting = True
    exact_match_required = True
    prefer_literal = True

    def __init__(self, context=None):
        super().__init__(context=context)

    @staticmethod
    def get_instance_type():
        return block_run_length_instance_type

    def generic(self):
        def typer(
            run_values,
            run_lengths,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=None,
            temp_storage=None,
        ):
            # error_class = errors.TypingError
            error_class = RuntimeError

            # Verify run_values and run_lengths are device arrays.
            if not isinstance(run_values, types.Array):
                raise error_class(
                    "run_values must be a device array, "
                    f"got {type(run_values).__name__}"
                )

            if not isinstance(run_lengths, types.Array):
                raise error_class(
                    "run_lengths must be a device array, "
                    f"got {type(run_lengths).__name__}"
                )

            validate_positive_integer_literal(
                self,
                runs_per_thread,
                "runs_per_thread",
            )

            validate_positive_integer_literal(
                self,
                decoded_items_per_thread,
                "decoded_items_per_thread",
            )

            if decoded_offset_dtype is not None:
                from ._common import normalize_dtype_param

                decoded_offset_dtype = normalize_dtype_param(decoded_offset_dtype)
                # if not isinstance(decoded_offset_dtype, types.Integer):
                #    raise error_class("decoded_offset_dtype must be an integer type")

            invalid_total_decoded_size = total_decoded_size is None or not isinstance(
                total_decoded_size, types.Integer
            )
            if invalid_total_decoded_size:
                raise error_class("total_decoded_size must be an integer type")

            validate_temp_storage(self, temp_storage)

            arglist = [
                run_values,
                run_lengths,
                runs_per_thread,
                decoded_items_per_thread,
                total_decoded_size,
            ]

            if decoded_offset_dtype is not None:
                arglist.append(decoded_offset_dtype)

            if temp_storage is not None:
                arglist.append(temp_storage)

            sig = signature(
                block_run_length_instance_type,
                *arglist,
            )

            return sig

        return typer


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


block_run_length_instance_type = CoopBlockRunLengthInstanceType()


@register_model(CoopBlockRunLengthInstanceType)
class CoopBlockRunLengthInstanceModel(models.OpaqueModel):
    pass


@typeof_impl.register(coop.block.run_length)
def typeof_block_run_length_instance(*args, **kwargs):
    return block_run_length_instance_type


@type_callable(block_run_length_instance_type)
def type_block_run_length_instance_call(context):
    def typer(
        run_values,
        run_lengths,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype,
        temp_storage=None,
    ):
        decl = block_run_length_instance_type.decl
        return decl.generic()

    return typer


class CoopBlockRunLengthAttrsTemplate(AttributeTemplate):
    key = block_run_length_instance_type

    def resolve_decode(self, instance):
        return types.BoundFunction(CoopBlockRunLengthDecodeDecl, instance)


register_attr(CoopBlockRunLengthAttrsTemplate)

block_run_length_attrs_template = CoopBlockRunLengthAttrsTemplate(None)


@lower_constant(CoopBlockRunLengthInstanceType)
def lower_constant_block_run_length_instance_type(context, builder, typ, value):
    raise RuntimeError("Not yet implemented")
    return context.get_dummy_value()


@lower_builtin(CoopBlockRunLengthInstanceType, types.VarArg(types.Any))
def codegen_block_run_length(context, builder, sig, args):
    raise RuntimeError("Not yet implemented")
    return context.get_dummy_value()


@lower_builtin("call", CoopBlockRunLengthInstanceType, types.VarArg(types.Any))
def codegen_block_run_length_call(context, builder, sig, args):
    raise RuntimeError("Not yet implemented")
    return context.get_dummy_value()


# =============================================================================
# Scan
# =============================================================================


@register
class CoopBlockScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.scan
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    is_constructor = False
    minimum_num_args = 3

    @staticmethod
    def signature(
        src: types.Array,
        dst: types.Array,
        items_per_thread: int = None,
        initial_value: Optional[Any] = None,
        mode: Literal["exclusive", "inclusive"] = "exclusive",
        scan_op: ScanOpType = "+",
        block_prefix_callback_op: Optional[Callable] = None,
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
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def get_instance_type():
        return block_scan_instance_type

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        dst = bound.arguments["dst"]
        validate_src_dst(self, src, dst)
        arglist = [src, dst]

        process_items_per_thread(
            self,
            bound,
            arglist,
            two_phase,
            target_array=src,
        )

        mode = bound.arguments.get("mode")
        if mode is None:
            mode = "exclusive"
        if mode not in ("inclusive", "exclusive"):
            raise errors.TypingError(
                f"Invalid mode '{mode}' for {self.primitive_name}; expected "
                "'inclusive' or 'exclusive'"
            )
        arglist.append(mode)

        scan_op = bound.arguments.get("scan_op", "+")
        try:
            scan_op = ScanOp(scan_op)
        except ValueError as e:
            raise errors.TypingError(
                f"Invalid scan_op '{scan_op}' for {self.primitive_name}: {e}"
            )
        arglist.append(scan_op)

        block_prefix_callback_op = bound.arguments.get("block_prefix_callback_op")
        if block_prefix_callback_op is not None:
            # We can't do much validation here.
            arglist.append(block_prefix_callback_op)

        initial_value = bound.arguments.get("initial_value")
        if isinstance(initial_value, types.IntegerLiteral):
            # If initial_value is an IntegerLiteral, we can use it directly.
            initial_value = initial_value.literal_value
        if initial_value is not None:
            arglist.append(initial_value)

        process_algorithm(self, bound, arglist)

        temp_storage = bound.arguments.get("temp_storage")
        validate_temp_storage(self, temp_storage)
        if temp_storage is not None:
            arglist.append(temp_storage)

        sig = signature(
            block_scan_instance_type,
            *arglist,
        )

        return sig


class CoopBlockScanInstanceType(types.Type, CoopInstanceTypeMixin):
    decl_class = CoopBlockScanDecl

    def __init__(self):
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(
        self,
        src,
        dst,
        items_per_thread=None,
        initial_value=None,
        mode=None,
        scan_op=None,
        block_prefix_callback_op=None,
        algorithm=None,
        temp_storage=None,
    ):
        bound = inspect.signature(self.decl.signature).bind(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode=mode,
            scan_op=scan_op,
            block_prefix_callback_op=block_prefix_callback_op,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )
        return self.decl._validate_args_and_create_signature(bound)


block_scan_instance_type = CoopBlockScanInstanceType()


@typeof_impl.register(coop.block.scan)
def typeof_block_scan_instance(*args, **kwargs):
    return block_scan_instance_type


@type_callable(block_scan_instance_type)
def type_block_scan_instance_call(context):
    instance = block_scan_instance_type

    def typer(
        src,
        dst,
        items_per_thread=None,
        initial_value=None,
        mode=None,
        scan_op=None,
        block_prefix_callback_op=None,
        algorithm=None,
        temp_storage=None,
    ):
        """
        This function is called to infer the type of the coop.block.scan
        instance type. It checks that the parameters are of the expected types.
        """
        return instance._validate_args_and_create_signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode=mode,
            scan_op=scan_op,
            block_prefix_callback_op=block_prefix_callback_op,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    return typer


@register_model(CoopBlockScanInstanceType)
class CoopBlockScanInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockScanInstanceType)
def lower_constant_block_scan_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


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

    def resolve_histogram(self, mod):
        return types.Function(CoopBlockHistogramDecl)

    def resolve_run_length(self, mod):
        return types.Function(CoopBlockRunLengthDecl)

    def resolve_scan(self, mod):
        return types.Function(CoopBlockScanDecl)


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

    def resolve_BlockScanAlgorithm(self, mod):
        return types.Module(coop.BlockScanAlgorithm)

    def resolve_WarpLoadAlgorithm(self, mod):
        return types.Module(coop.WarpLoadAlgorithm)

    def resolve_WarpStoreAlgorithm(self, mod):
        return types.Module(coop.WarpStoreAlgorithm)

    def resolve_BlockHistogramAlgorithm(self, mod):
        return types.Module(coop.BlockHistogramAlgorithm)


register_global(coop, types.Module(coop))
