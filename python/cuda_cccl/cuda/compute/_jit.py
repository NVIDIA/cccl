# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import functools
import inspect
import operator
import struct
import textwrap
from types import new_class
from typing import TYPE_CHECKING, Callable, Hashable, List, Tuple

import numpy as np

# numba-cuda-mlir backend: used for op compilation, return-type inference, the
# gpu_struct typing/lowering machinery, and the TypeDescriptor <-> numba type
# conversions (see ._mlir).
from . import _mlir
from . import types as cccl_types
from ._bindings import Op, OpKind
from ._caching import CachableFunction, cache_with_registered_key_functions

try:
    from ._build_info import USING_V2  # type: ignore[import-not-found]
except ImportError:
    USING_V2 = False

from ._odr_helpers import create_stateful_op_void_ptr_wrapper
from ._utils import sanitize_identifier
from ._utils.protocols import (
    get_data_pointer,
    get_dtype,
    is_contiguous,
    is_device_array,
)
from .op import OpAdapter

if TYPE_CHECKING:
    from .typing import DeviceArrayLike


def _compile_op_to_llvm_bitcode(wrapped_op, wrapper_sig) -> bytes:
    """Compile a device op to LLVM bitcode (.bc) bytes via numba-cuda-mlir.

    Used on the v2 (HostJIT) backend, which prefers LLVM bitcode over NVRTC
    LTO-IR — the JIT linker routes "BC"-magic blobs through LLVM's native
    bitcode linker instead of nvJitLink's LTO codegen.

    numba-cuda-mlir's public ``cuda.compile`` only emits PTX or LTO-IR, so we
    extract LLVM IR from its internal MLIR -> LLVM translation (one step before
    libnvvm; see ``_mlir.compile_to_llvm_ir``) and turn that textual IR into
    bitcode with llvmlite.  The C-ABI wrapper is emitted under the exact symbol
    ``wrapped_op.__name__`` that CUB's PTX references by name.
    """
    import os

    import llvmlite.binding as llvm

    target_name = wrapped_op.__name__
    text_ir = _mlir.compile_to_llvm_ir(wrapped_op, wrapper_sig, target_name)

    debug_dir = os.environ.get("CCCL_JIT_DEBUG")
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, f"{target_name}.raw.ll"), "w") as f:
            f.write(text_ir)

    try:
        module = llvm.parse_assembly(text_ir)
        module.verify()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse LLVM IR for '{target_name}': {exc}"
        ) from exc

    if debug_dir:
        with open(os.path.join(debug_dir, f"{target_name}.symbols.txt"), "w") as f:
            f.write(f"target_name={target_name}\n")
            for fn in module.functions:
                f.write(
                    f"  {fn.linkage} {'decl' if fn.is_declaration else 'def '} {fn.name}\n"
                )

    return bytes(module.as_bitcode())


# -----------------------------------------------------------------------------
# Struct registration and casting
# -----------------------------------------------------------------------------


# Base class for all struct types, used for struct-to-struct cast matching.
class _StructBase(_mlir.types.Type):
    """Base class for all CCCL GPU struct types."""

    _field_spec: dict  # Mapping of field names to Numba types


# The struct registration logic is isolated here to avoid polluting other
# modules with Numba-specific type plumbing.
@functools.lru_cache(maxsize=256)
def _make_struct_type(struct_class_or_name, field_names, field_types):
    """
    Core factory function that uses the Numba extension machinery to
    create a struct type with pass-by-value semantics.

    This function is called lazily from _register_struct_with_numba.

    Args:
        struct_class_or_name: Either an existing struct class (from gpu_struct) or a string name
        field_names: Tuple of field names
        field_types: Tuple of TypeDescriptors
    """
    from .struct import _is_struct_type, _Struct

    # If we're given an existing class, use it; otherwise create a new one
    if isinstance(struct_class_or_name, type):
        struct_class = struct_class_or_name
    else:
        struct_class = new_class(struct_class_or_name, bases=(_Struct,))

    # Convert TypeDescriptors to Numba types
    numba_field_types = [type_descriptor_to_numba(ft) for ft in field_types]

    raw_field_spec = dict(zip(field_names, numba_field_types))
    assert all(
        _is_struct_type(tp) or isinstance(tp, _mlir.types.Type)
        for tp in raw_field_spec.values()
    )

    field_spec = {
        name: _mlir.as_numba_type(typ) if _is_struct_type(typ) else typ
        for name, typ in raw_field_spec.items()
    }

    # Only set _field_spec if it doesn't already exist (for new classes created here)
    # For classes from gpu_struct, they already have _field_spec with TypeDescriptors
    if not hasattr(struct_class, "_field_spec"):
        struct_class._field_spec = raw_field_spec

    class StructType(_StructBase):
        def __init__(self):
            super().__init__(name=struct_class.__name__)
            # Store field_spec for struct-to-struct conversion checks
            self._field_spec = field_spec

        def can_convert_from(self, typingctx, other):
            if isinstance(other, _mlir.types.UniTuple):
                tuple_size = other.count
                if tuple_size == len(field_types):
                    return _mlir.Conversion.safe

            elif isinstance(other, _mlir.types.Tuple):
                tuple_size = len(other.types)
                if tuple_size == len(field_types):
                    all_compatible = all(
                        typingctx.can_convert(src_type, tgt_type) is not None
                        for src_type, tgt_type in zip(other.types, field_spec.values())
                    )
                    if all_compatible:
                        return _mlir.Conversion.safe

            # Allow conversion from another StructType with identical field layout
            elif hasattr(other, "_field_spec"):
                other_field_spec = other._field_spec
                # Check if field count matches
                if len(other_field_spec) == len(field_spec):
                    # Check if all field types are compatible (by position, not name)
                    other_field_types = list(other_field_spec.values())
                    self_field_types = list(field_spec.values())
                    all_compatible = all(
                        typingctx.can_convert(src_type, tgt_type) is not None
                        for src_type, tgt_type in zip(
                            other_field_types, self_field_types
                        )
                    )
                    if all_compatible:
                        return _mlir.Conversion.safe

            return None

    numba_type = StructType()
    numba_type.python_type = struct_class

    _mlir.as_numba_type.register(struct_class, numba_type)

    @_mlir.typeof_impl.register(struct_class)
    def typeof_struct(val, c):
        return numba_type  # Must return the SAME instance, not a new StructType()

    # Data model: the struct lowers to an LLVM struct whose members are the MLIR
    # value types of the fields (numba-cuda-mlir builds backend types as MLIR).
    # Use a *literal* (structural) struct rather than new_identified: the same
    # logical gpu_struct is registered more than once (input type, constructed
    # value, h_init, ...), and new_identified mints a fresh uniquely-named type
    # each call, so casts between two registrations of the same struct fail. A
    # literal struct compares equal by body, so all registrations agree.
    @_mlir.register_model(StructType)
    class StructModel(_mlir.PrimitiveModel):
        def __init__(self, dmm, fe_type):
            member_mlir_types = [
                dmm.lookup(typ).get_value_type() for typ in field_spec.values()
            ]
            be_type = _mlir.llvm.StructType.get_literal(member_mlir_types)
            super().__init__(dmm, fe_type, be_type)

    field_names_list = list(field_spec.keys())

    # Field access typing: `struct.field` resolves to the field's type.  This
    # replaces numba-cuda's make_attribute_wrapper, which has no MLIR equivalent;
    # the matching lowering is the lower_getattr_generic below.
    @_mlir.typing_registry.register_attr
    class StructAttributeTemplate(_mlir.AttributeTemplate):
        key = StructType

        def generic_resolve(self, typ, attr):
            return typ._field_spec.get(attr)

    @_mlir.lowering_registry.lower_getattr_generic(StructType)
    def lower_struct_getattr(context, builder, target, value, attr):
        field_index = field_names_list.index(attr)
        struct_value = builder.load_var(value)
        struct_mlir_ty = _mlir.llvm.StructType(struct_value.type)
        field_mlir_ty = struct_mlir_ty.body[field_index]
        field_value = _mlir.llvm.extractvalue(
            res=field_mlir_ty,
            container=struct_value,
            position=_mlir.struct_field_position(field_index),
        )
        target_mlir_ty = builder.get_mlir_type(builder.get_numba_type(target.name))
        builder.store_var(target, _mlir.convert(field_value, target_mlir_ty))

    # Validate that all field names are valid Python identifiers before
    # we exec any generated code that accesses them:
    for name in field_names_list:
        if not name.isidentifier():
            raise ValueError(
                f"Struct field name {name!r} is not a valid Python identifier"
            )

    @_mlir.overload(
        operator.getitem,
        typing_registry=_mlir.typing_registry,
        prefer_literal=True,
    )
    def struct_getitem(struct_val, idx):
        if not isinstance(struct_val, StructType):
            return

        if isinstance(idx, (_mlir.types.IntegerLiteral)):
            idx_val = getattr(idx, "literal_value", getattr(idx, "value", None))

            if idx_val is None or not (0 <= idx_val < len(field_names_list)):

                def error_impl(struct_val, idx):
                    raise IndexError(
                        f"Index out of range for struct with {len(field_names_list)} fields"
                    )

                return error_impl

            field_name = field_names_list[idx_val]
            exec(
                f"def impl(struct_val, idx): return struct_val.{field_name}",
                namespace := {},
            )
            return namespace["impl"]

        conditions = "\n".join(
            f"    {'if' if i == 0 else 'elif'} idx == {i}: return struct_val.{name}"
            for i, name in enumerate(field_names_list)
        )
        exec(
            f"def impl(struct_val, idx):\n{conditions}\n    else: raise IndexError('Index out of range')",
            namespace := {},
        )
        return namespace["impl"]

    # getitem lowering: `struct[i]` with a constant index extracts field i.
    # The overload above supplies the (literal-aware) typing; numba-cuda-mlir's
    # getitem lowering needs a registered builder, which it looks up with the
    # constant index normalized to int64.  Registering this builder also means
    # the overload's generated impl (which would `raise IndexError`, something
    # numba-cuda-mlir cannot lower) is never compiled.
    def lower_struct_getitem(builder, target, args, kwargs):
        struct_var, index = args
        # The index arrives as a plain int (static_getitem) or as an IR Var
        # whose numba type is an IntegerLiteral carrying the constant value.
        if isinstance(index, int):
            field_index = index
        else:
            index_type = builder.get_numba_type(index.name)
            field_index = getattr(index_type, "literal_value", None)
        if field_index is None or not (0 <= field_index < len(field_names_list)):
            raise NotImplementedError(
                "indexing a gpu_struct requires a constant integer index in range"
            )
        struct_value = builder.load_var(struct_var)
        struct_mlir_ty = _mlir.llvm.StructType(struct_value.type)
        field_mlir_ty = struct_mlir_ty.body[field_index]
        field_value = _mlir.llvm.extractvalue(
            res=field_mlir_ty,
            container=struct_value,
            position=_mlir.struct_field_position(field_index),
        )
        target_mlir_ty = builder.get_mlir_type(builder.get_numba_type(target.name))
        builder.store_var(target, _mlir.convert(field_value, target_mlir_ty))

    _mlir.lowering_registry.lower(operator.getitem, StructType, _mlir.types.Integer)(
        lower_struct_getitem
    )

    # Constructor typing: StructClass(field0, field1, ...) -> struct.
    # Use an AbstractTemplate (rather than a ConcreteTemplate keyed on the exact
    # field types) so a call whose argument types merely *convert* to the field
    # types still matches -- numba-cuda-mlir promotes e.g. int32 + int32 to
    # int64, so `Struct(a.x + b.x, ...)` arrives with wider arg types.  The
    # lowering converts each argument to its field type.
    _struct_field_types = list(field_spec.values())

    class StructConstructor(_mlir.AbstractTemplate):
        key = struct_class

        def generic(self, args, kws):
            # Match on arity only and accept the actual argument types: numba
            # promotes arithmetic (int32 + int32 -> int64), so a field built
            # from an expression arrives wider than its declared type, and a
            # narrowing conversion (int64 -> int32) is not an *implicit* numba
            # conversion.  The constructor lowering converts each argument to
            # its field type explicitly.
            if kws or len(args) != len(_struct_field_types):
                return None
            return _mlir.signature(numba_type, *args)

    _mlir.typing_registry.register_global(
        struct_class, _mlir.types.Function(StructConstructor)
    )

    def _pack_fields(builder, struct_mlir_ty, field_mlir_values):
        """Build an LLVM struct value from per-field MLIR values."""
        result = _mlir.llvm.UndefOp(struct_mlir_ty)
        for i, field_value in enumerate(field_mlir_values):
            result = _mlir.llvm.insertvalue(
                container=result,
                value=field_value,
                position=_mlir.struct_field_position(i),
            )
        return result

    def _coerce_to_field(builder, value, field_numba_type):
        """Coerce a constructor argument value to its declared field type.

        Scalars are converted directly.  A struct field may be supplied as a
        tuple of its own field values (tuple-construction syntax, e.g.
        ``Outer(x, (a, b))``); numba-cuda-mlir represents such a tuple as a
        Python sequence of MLIR values, which we pack into the field's struct
        (recursively, so nested tuple-construction works).
        """
        field_mlir_ty = builder.get_mlir_type(field_numba_type)
        if isinstance(value, (tuple, list)):
            sub_field_types = list(field_numba_type._field_spec.values())
            sub_values = [
                _coerce_to_field(builder, v, t) for v, t in zip(value, sub_field_types)
            ]
            return _pack_fields(
                builder, _mlir.llvm.StructType(field_mlir_ty), sub_values
            )
        return _mlir.convert(value, field_mlir_ty)

    # Constructor lowering: coerce each argument to its field type and pack into
    # the LLVM struct (replaces cgutils.create_struct_proxy).
    def struct_constructor(builder, target, args, kwargs):
        struct_mlir_ty = _mlir.llvm.StructType(
            builder.get_mlir_type(builder.get_numba_type(target.name))
        )
        field_values = [
            _coerce_to_field(builder, builder.load_var(arg), field_type)
            for arg, field_type in zip(args, field_spec.values())
        ]
        builder.store_var(target, _pack_fields(builder, struct_mlir_ty, field_values))

    # Register the constructor lowering as a catch-all on the struct class
    # (variadic, any argument types) so it matches calls whose argument types
    # were promoted (e.g. `Struct(a.x + b.x, ...)` arrives as int64 even though
    # the field is int32).  The body converts each argument to its declared
    # field type.  Registering for the exact field types (or for no arguments)
    # would miss those promoted calls and fail with
    # "NotImplemented lowering call to <struct>".
    _mlir.lowering_registry.lower(struct_class, _mlir.types.VarArg(_mlir.types.Any))(
        struct_constructor
    )

    # NOTE: the tuple->struct and struct->struct cast lowerings below mirror the
    # numba-cuda implementation translated to MLIR.  numba-cuda-mlir routes
    # aggregate-unification casts differently than numba-cuda, so these are the
    # part of the migration most in need of validation against the struct test
    # suite.
    @_mlir.lower_cast(_mlir.types.BaseTuple, StructType)
    def tuple_to_struct_cast(context, builder, fromty, toty, val):
        if isinstance(fromty, _mlir.types.UniTuple):
            tuple_size = fromty.count
            element_types = [fromty.dtype] * tuple_size
        else:
            tuple_size = len(fromty.types)
            element_types = list(fromty.types)

        if tuple_size != len(field_spec):
            raise ValueError(
                f"Cannot cast tuple of size {tuple_size} to {struct_class.__name__} "
                f"with {len(field_types)} fields"
            )

        # A numba-cuda-mlir tuple value is a Python sequence of MLIR values when
        # not yet concretized; fall back to extractvalue for aggregate values.
        if isinstance(val, (tuple, list)):
            elements = list(val)
        else:
            elements = [
                _mlir.llvm.extractvalue(
                    res=builder.get_mlir_type(element_types[i]),
                    container=val,
                    position=_mlir.struct_field_position(i),
                )
                for i in range(tuple_size)
            ]

        struct_mlir_ty = _mlir.llvm.StructType(builder.get_mlir_type(toty))
        field_values = [
            _mlir.convert(elements[i], builder.get_mlir_type(field_type))
            for i, field_type in enumerate(field_spec.values())
        ]
        return _pack_fields(builder, struct_mlir_ty, field_values)

    @_mlir.lower_cast(_StructBase, StructType)
    def cast_struct_to_struct(context, builder, fromty, toty, val):
        """Cast from one CCCL struct type to another with identical layout."""
        from_field_types = list(fromty._field_spec.values())
        to_field_types = list(toty._field_spec.values())

        if len(from_field_types) != len(to_field_types):
            return None

        struct_mlir_ty = _mlir.llvm.StructType(builder.get_mlir_type(toty))
        field_values = []
        for i, (from_type, to_type) in enumerate(zip(from_field_types, to_field_types)):
            elem = _mlir.llvm.extractvalue(
                res=builder.get_mlir_type(from_type),
                container=val,
                position=_mlir.struct_field_position(i),
            )
            field_values.append(_mlir.convert(elem, builder.get_mlir_type(to_type)))
        return _pack_fields(builder, struct_mlir_ty, field_values)

    return struct_class


@functools.lru_cache(maxsize=256)
def _register_struct_with_numba(struct_class):
    field_spec = struct_class._type_descriptor.fields

    registered_class = _make_struct_type(
        struct_class,
        tuple(field_spec.keys()),
        tuple(field_spec.values()),
    )

    return _mlir.as_numba_type(registered_class)


# -----------------------------------------------------------------------------
# TypeDescriptor <-> Numba conversions
# -----------------------------------------------------------------------------


@functools.cache
def type_descriptor_to_numba(td):
    """
    Convert a TypeDescriptor to a Numba type.

    Handles:
    - PointerTypeDescriptor: creates CPointer to the pointee's numba type
    - StructTypeDescriptor: registers a struct class for the layout
    - POD TypeDescriptor: uses numba-cuda-mlir's from_dtype
    - Numba types: pass through
    """

    # Pass through if already a numba-cuda-mlir type
    if isinstance(td, _mlir.types.Type):
        return td

    # Handle PointerTypeDescriptor (must check before TypeDescriptor since it's a subclass)
    if isinstance(td, cccl_types.PointerTypeDescriptor):
        return _mlir.types.CPointer(type_descriptor_to_numba(td.pointee))

    # Handle TypeDescriptor (includes StructTypeDescriptor)
    if isinstance(td, cccl_types.TypeDescriptor):
        return _convert_type_descriptor_to_numba(td)

    raise TypeError(f"Expected TypeDescriptor or numba type, got {type(td)}")


def _convert_type_descriptor_to_numba(td):
    """Internal helper to convert TypeDescriptor to Numba type."""

    # For struct types
    if isinstance(td, cccl_types.StructTypeDescriptor):
        from .struct import (
            _get_struct_record_dtype,
            _get_struct_type_descriptor,
            _Struct,
        )

        layout_key = tuple(td.layout_key())
        struct_class = new_class(td.name, bases=(_Struct,))
        struct_class._field_spec = dict(layout_key)
        struct_class._type_descriptor = _get_struct_type_descriptor(struct_class)
        struct_class.dtype = _get_struct_record_dtype(struct_class)
        try:
            return _mlir.as_numba_type(struct_class)
        except _mlir.errors.NumbaError:
            return _register_struct_with_numba(struct_class)

    # For POD types
    return _mlir.from_numpy_dtype(td.dtype)


def _is_gpu_struct_class(obj):
    """Check if an object is a gpu_struct class (not instance)."""
    return (
        isinstance(obj, type)
        and hasattr(obj, "_type_descriptor")
        and hasattr(obj, "_field_spec")
    )


def _iter_function_objects(py_func):
    if hasattr(py_func, "__globals__"):
        yield from py_func.__globals__.values()

    if py_func.__closure__ is not None:
        for cell in py_func.__closure__:
            try:
                yield cell.cell_contents
            except ValueError:
                pass


def _ensure_function_structs_registered(py_func):
    """
    Scan a function's globals and closure for gpu_struct classes and ensure
    they're registered with Numba before compilation.
    """

    def _register_if_needed(struct_class):
        try:
            return _mlir.as_numba_type(struct_class)
        except _mlir.errors.NumbaError:
            return _register_struct_with_numba(struct_class)

    for value in _iter_function_objects(py_func):
        if _is_gpu_struct_class(value):
            _register_if_needed(value)


def _numba_type_to_type_descriptor(numba_type):
    """Convert a Numba type to a TypeDescriptor (internal helper)."""
    from .struct import _is_struct_type

    # Already a TypeDescriptor
    if isinstance(numba_type, cccl_types.TypeDescriptor):
        return numba_type

    # Custom StructType with an associated gpu_struct Python class
    if hasattr(numba_type, "python_type") and _is_struct_type(numba_type.python_type):
        return numba_type.python_type._type_descriptor

    # POD type - convert via numpy dtype
    dtype = _mlir.as_numpy_dtype(numba_type)
    return cccl_types.from_numpy_dtype(dtype)


@cache_with_registered_key_functions
def _infer_return_type(py_func, input_types):
    # Ensure any gpu_struct classes referenced in the function are registered
    _ensure_function_structs_registered(py_func)

    # Compile to infer return type
    from ._utils import sanitize_identifier

    sanitized_name = sanitize_identifier(py_func.__name__)
    unique_suffix = hex(id(py_func))[2:]
    abi_name = f"{sanitized_name}_{unique_suffix}"
    input_numba_types = tuple(type_descriptor_to_numba(t) for t in input_types)
    _, return_type = _mlir.cuda.compile(
        py_func,
        input_numba_types,
        device=True,
        abi_info={"abi_name": abi_name},
        output="ltoir",
    )
    return _numba_type_to_type_descriptor(return_type)


# -----------------------------------------------------------------------------
# Stateless ops
# -----------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _compile_op_impl(cachable_op, input_types_tuple: tuple, output_type):
    """Cached implementation of op compilation.

    Args:
        cachable_op: CachableFunction wrapper around the operator
        input_types_tuple: Tuple of input TypeDescriptors
        output_type: Output TypeDescriptor
    """
    from ._bindings import Op, OpKind
    from ._odr_helpers import create_op_void_ptr_wrapper

    # Extract the actual function from CachableFunction
    op = cachable_op._func

    # Ensure any gpu_struct classes referenced in the op are registered
    _ensure_function_structs_registered(op)

    numba_input_types = tuple(
        type_descriptor_to_numba(t) if isinstance(t, cccl_types.TypeDescriptor) else t
        for t in input_types_tuple
    )

    if isinstance(output_type, cccl_types.TypeDescriptor):
        numba_output_type = type_descriptor_to_numba(output_type)
    else:
        numba_output_type = output_type

    sig = numba_output_type(*numba_input_types)
    wrapped_op, wrapper_sig = create_op_void_ptr_wrapper(op, sig)

    from ._device_code import DeviceCode

    if USING_V2:
        code = DeviceCode(
            op_bytes=_compile_op_to_llvm_bitcode(wrapped_op, wrapper_sig),
            kind="llvm_ir",
        )
    else:
        ltoir, _ = _mlir.cuda.compile(
            wrapped_op,
            sig=wrapper_sig,
            device=True,
            abi="c",
            abi_info={"abi_name": wrapped_op.__name__},
            output="ltoir",
        )
        code = DeviceCode(op_bytes=ltoir, kind="ltoir")

    return Op(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=code,
        state_alignment=1,
        state=None,
    )


def compile_op(op, input_types, output_type=None):
    """Compile a user-provided binary operator for use with CCCL algorithms.

    This function is cached to ensure that identical operators with identical
    types produce the same compiled result (same symbol names and LTOIR),
    which allows proper deduplication during linking.
    """
    from ._caching import CachableFunction

    cachable_op = CachableFunction(op)
    return _compile_op_impl(cachable_op, tuple(input_types), output_type)


class _StatelessOp(OpAdapter):
    """Adapter for stateless callables."""

    __slots__ = ()

    def __init__(self, func):
        self._func = func
        self._cachable = CachableFunction(func)

    def compile(self, input_types, output_type=None) -> Op:
        return compile_op(self._func, input_types, output_type)

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func

    def __eq__(self, other):
        return self._cachable == other._cachable

    def __hash__(self):
        return hash(
            self._cachable,
        )

    def get_return_type(self, input_types):
        return _infer_return_type(self._func, input_types)


# -----------------------------------------------------------------------------
# Stateful ops
# -----------------------------------------------------------------------------
#
# Stateful ops are callables that capture device arrays (state) as globals or
# closure variables.
#
# When numba-cuda encounters a device function that references device
# arrays in its globals/closures, it captures pointers to the
# referenced arrays as constants.  If an object being referenced
# changes (e.g., common in a loop) the device function will be
# recompiled because it sees a new pointer.
#
# We avoid this by relying on support for stateful ops in the CCCL C
# library itself. Here's how this works
#
# 1. Detect any global/closure device arrays referenced by a given callable.
#
# 2. Transform the callable's AST to add parameters for each detected device array.
#
# 3. Compilation: the compiled device function whose LTOIR we
#    eventually pass to the CCCL C library must take a single `void*`
#    argument for is state.  The corresponding declaration of stateful
#    operators in CCCL C looks like:
#
#       extern "C" __device__ void foo(void* state, )`.
#
#    Thus, we have Thus, at compilation time, we
#    define some intrinsics to unpack a `void*` into typed arrays and
#    invoke the transformed callable from step 2.  Much of this is
#    implemented in _odr_helpers.py


def _detect_device_array_globals(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect device arrays referenced as globals in a function.

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    state_arrays = []
    code = func.__code__

    for name in code.co_names:
        val = func.__globals__.get(name)
        if val is not None and is_device_array(val):
            state_arrays.append((name, val))

    return state_arrays


def _detect_device_array_closures(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect device arrays captured in function closures.

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    state_arrays: List[Tuple[str, object]] = []
    code = func.__code__
    closure = func.__closure__

    if closure is None:
        return state_arrays

    # co_freevars contains the names of closure variables
    for name, cell in zip(code.co_freevars, closure):
        try:
            val = cell.cell_contents
            if is_device_array(val):
                state_arrays.append((name, val))
        except ValueError:
            # Cell is empty
            pass

    return state_arrays


def _detect_all_device_arrays(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect all device arrays referenced by a function (globals + closures).

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    globals_arrays = _detect_device_array_globals(func)
    closure_arrays = _detect_device_array_closures(func)
    return globals_arrays + closure_arrays


class _AddStateParameters(ast.NodeTransformer):
    """AST transformer that adds state parameters to a function definition."""

    def __init__(self, state_names: List[str]):
        self.state_names = state_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Prepend state parameters to the function arguments
        # Inner function signature: (state_arrays..., regular_args...)
        new_args = [ast.arg(arg=name, annotation=None) for name in self.state_names]
        node.args.args = new_args + node.args.args
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        # Handle async functions the same way
        new_args = [ast.arg(arg=name, annotation=None) for name in self.state_names]
        node.args.args = new_args + node.args.args
        return node


def _transform_function_ast(func: Callable, state_names: List[str]) -> Callable:
    """
    Transform a function to add state arrays captured as globals or closures
    as explicit parameters.

    For example, if the function is:

        def func(x): return x + state[0]  # state is a global array

    Then the transformed function will be:

        def func(state, x): return x + state[0]

    Args:
        func: The original function
        state_names: Names of device arrays to add as parameters

    Returns:
        A new function with state arrays as explicit parameters,
        appearing after the regular parameters.

    Raises:
        ValueError: If the function source cannot be obtained
    """
    # Get source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Cannot get source code for function '{func.__name__}'. "
        ) from e

    # Dedent source (in case function is defined inside a class/function)
    source = textwrap.dedent(source)

    # Parse to AST
    tree = ast.parse(source)

    # Transform: add state parameters
    transformer = _AddStateParameters(state_names)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Compile and execute to create new function
    # We need to provide the original function's globals for imports
    # AND inject closure variables so they're accessible in the new function
    globals_dict = func.__globals__.copy()

    # Inject closure variables (except state arrays which become parameters)
    if func.__closure__:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            if name not in state_names:  # Don't inject state arrays
                try:
                    globals_dict[name] = cell.cell_contents
                except ValueError:
                    pass  # Cell is empty

    local_ns: dict[str, Callable] = {}
    exec(
        compile(tree, filename=f"<auto_stateful:{func.__name__}>", mode="exec"),
        globals_dict,
        local_ns,
    )

    # Get the transformed function
    transformed_func = local_ns[func.__name__]

    return transformed_func


def _extract_state(func: Callable):
    # Detect device arrays
    state_info = _detect_all_device_arrays(func)

    # Extract names and arrays
    state_names = [name for name, _ in state_info]
    state_arrays: List[DeviceArrayLike] = [arr for _, arr in state_info]  # type: ignore[misc]

    return state_names, state_arrays


def _compile_stateful_op(op, input_types, state_arrays, output_type=None):
    """
    Compile a stateful operator for use with CCCL algorithms.

    Args:
        op: The operator function (already transformed to take state as explicit parameters)
        input_types: Tuple of TypeDescriptors for regular input arguments
        state_arrays: Tuple of device arrays containing state
        output_type: Optional TypeDescriptor for return value (inferred if None)

    Returns:
        Compiled Op object for C++ interop
    """

    # Ensure any gpu_struct classes referenced in the op are registered
    _ensure_function_structs_registered(op)

    # Validate all state arrays are contiguous
    for i, state_array in enumerate(state_arrays):
        if not is_contiguous(state_array):
            raise ValueError(f"state array {i} must be contiguous")

    # Convert input types to numba-cuda-mlir types
    numba_input_types = tuple(type_descriptor_to_numba(t) for t in input_types)

    # State arrays are passed to the (transformed) op as typed pointers; the op
    # body indexes them (``state[i]``), which works on a CPointer.  See
    # _odr_helpers.create_stateful_op_void_ptr_wrapper for how the packed state
    # void* is unpacked into one CPointer per state array.
    state_dtypes = [_mlir.from_numpy_dtype(get_dtype(s)) for s in state_arrays]
    state_ptr_types = [_mlir.types.CPointer(dt) for dt in state_dtypes]

    # Infer output type if needed
    if output_type is None:
        # Compile to infer return type.
        # The transformed function expects (state_arrays..., regular_args...)
        all_numba_input_types = tuple(state_ptr_types) + numba_input_types
        sanitized_name = sanitize_identifier(op.__name__)
        unique_suffix = hex(id(op))[2:]
        abi_name = f"{sanitized_name}_{unique_suffix}"
        _, return_type = _mlir.cuda.compile(
            op,
            all_numba_input_types,
            device=True,
            abi_info={"abi_name": abi_name},
            output="ltoir",
        )
        # Convert return type to TypeDescriptor
        output_type = cccl_types.from_numpy_dtype(_mlir.as_numpy_dtype(return_type))

    # Convert output type to numba-cuda-mlir type
    numba_output_type = type_descriptor_to_numba(output_type)

    # Build full signature: output_type(state_arrays..., regular_args...)
    sig = numba_output_type(*state_ptr_types, *numba_input_types)

    # Get state pointers - pointers to the device array data
    state_ptrs = [get_data_pointer(arr) for arr in state_arrays]

    # All pointers have the same alignment, use pointer-sized int alignment
    state_alignment = np.dtype(np.intp).alignment

    # Create the stateful wrapper (unpacks the packed state pointers).
    wrapped_op, wrapper_sig = create_stateful_op_void_ptr_wrapper(op, sig, state_dtypes)

    # Compile the wrapper — LLVM bitcode for v2 (HostJIT), LTO-IR for v1 (NVRTC).
    from ._device_code import DeviceCode

    if USING_V2:
        code = DeviceCode(
            op_bytes=_compile_op_to_llvm_bitcode(wrapped_op, wrapper_sig),
            kind="llvm_ir",
        )
    else:
        ltoir, _ = _mlir.cuda.compile(
            wrapped_op,
            sig=wrapper_sig,
            device=True,
            abi="c",
            abi_info={"abi_name": wrapped_op.__name__},
            output="ltoir",
        )
        code = DeviceCode(op_bytes=ltoir, kind="ltoir")

    # Pack all data pointers as bytes (sequentially)
    state_bytes = struct.pack(f"{len(state_ptrs)}P", *state_ptrs)

    # Return Op with STATEFUL kind and packed pointers
    return Op(
        operator_type=OpKind.STATEFUL,
        name=wrapped_op.__name__,
        ltoir=code,
        state_alignment=state_alignment,
        state=state_bytes,
    )


class _JitOpState:
    def __init__(self, names: List[str], arrays: List[DeviceArrayLike]):
        self.names = names
        self.arrays = arrays

    def get_cache_key(self) -> Hashable:
        return (tuple(self.names), tuple(get_dtype(s) for s in self.arrays))

    def to_bytes(self):
        state_ptrs = [get_data_pointer(arr) for arr in self.arrays]
        return struct.pack(f"{len(state_ptrs)}P", *state_ptrs)


class _StatefulOp(OpAdapter):
    __slots__ = "_state"

    def __init__(self, func, state):
        self._func = func
        self._cachable = CachableFunction(func)
        self._state = state

    def get_state(self):
        return self._state.to_bytes()

    def compile(self, input_types, output_type=None) -> Op:
        transformed_func = _transform_function_ast(self._func, self._state.names)
        return _compile_stateful_op(
            transformed_func, input_types, self._state.arrays, output_type
        )

    def get_cache_key(self):
        return (self._func.__name__, self._cachable, self._state.get_cache_key())

    @property
    def is_stateful(self) -> bool:
        return True

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func

    def __hash__(self) -> int:
        return hash(self.get_cache_key())

    def __eq__(self, other):
        return (self._cachable == other._cachable) and (self._state == other._state)


def to_jit_op_adapter(op: Callable) -> OpAdapter:
    """
    Convert the Python callable into the appropriate stateful or stateless
    OpAdapter, depending on whether or not it references any global state.
    """
    state_names, state_arrays = _extract_state(op)
    if state_names:
        return _StatefulOp(op, _JitOpState(state_names, state_arrays))
    else:
        return _StatelessOp(op)


cache_with_registered_key_functions.register(_StatelessOp, lambda op: op._cachable)

cache_with_registered_key_functions.register(_StatefulOp, lambda op: op.get_cache_key())


__all__ = [
    "to_jit_op_adapter",
]
