# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler-facing dtype, operator, and generated-invocable machinery.

Primitive routing lives in the semantic lowering modules; this module owns the
shared Numba types and wrapper construction those lowerings build upon.
"""

import hashlib
import os
import re
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from io import StringIO
from numbers import Integral
from textwrap import dedent
from types import FunctionType as PyFunctionType
from typing import BinaryIO, Sequence

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.compiler import ExternFunction
from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.types import signature

from cuda.coop._core import SynchronizationScope

from ._compiler import _nvrtc as nvrtc
from ._compiler._numba_mlir_compat import (
    _get_numba_mlir_compat,
    _get_numba_mlir_datamodel_compat,
)
from ._compiler._operations import StorageABI
from ._semantic import _numba_semantic_token

_numba_mlir_compat = _get_numba_mlir_compat()
_NumbaCudaMlirOverloadFunctionTemplate = _numba_mlir_compat.overload_function_template
make_overload_template = _numba_mlir_compat.make_overload_template

NUMBA_TYPES_TO_CPP = {
    types.boolean: "bool",
    types.int8: "::cuda::std::int8_t",
    types.int16: "::cuda::std::int16_t",
    types.int32: "::cuda::std::int32_t",
    types.int64: "::cuda::std::int64_t",
    types.uint8: "::cuda::std::uint8_t",
    types.uint16: "::cuda::std::uint16_t",
    types.uint32: "::cuda::std::uint32_t",
    types.uint64: "::cuda::std::uint64_t",
    types.float16: "__half",
    types.float32: "float",
    types.float64: "double",
}

_SUPPORTED_LOGICAL_WARP_THREADS = frozenset({1, 2, 4, 8, 16, 32})


_COOP_SPECIALIZATION_COLLECTOR: ContextVar[
    list[tuple[object, int | None, int | tuple[int, ...] | None]] | None
] = ContextVar("cuda_coop_numba_mlir_specialization_collector", default=None)


@contextmanager
def collect_specializations():
    collected: list[tuple[object, int | None, int | tuple[int, ...] | None]] = []
    token = _COOP_SPECIALIZATION_COLLECTOR.set(collected)
    try:
        yield collected
    finally:
        _COOP_SPECIALIZATION_COLLECTOR.reset(token)


def numba_type_to_cpp(numba_type):
    cpp_type = NUMBA_TYPES_TO_CPP.get(numba_type)
    if cpp_type is not None:
        return cpp_type
    return "storage_t"


def _validate_logical_warp_threads(threads):
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads not in _SUPPORTED_LOGICAL_WARP_THREADS
    ):
        supported = ", ".join(
            str(value) for value in sorted(_SUPPORTED_LOGICAL_WARP_THREADS)
        )
        raise ValueError(
            "warp-scoped providers require a logical width in "
            f"{{{supported}}}; got {threads!r}"
        )
    return threads


def _normalize_block_threads(threads_per_block):
    if isinstance(threads_per_block, bool):
        raise ValueError(
            "block_threads must be a positive integer or dimension tuple; "
            f"got {threads_per_block!r}"
        )
    if isinstance(threads_per_block, int):
        block_threads = threads_per_block
    elif isinstance(threads_per_block, (tuple, list)):
        if not 1 <= len(threads_per_block) <= 3:
            raise ValueError(
                "block_threads dimension tuples require one to three entries; "
                f"got {threads_per_block!r}"
            )
        block_threads = 1
        for dimension in threads_per_block:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError(
                    "block_threads dimension tuple entries must be positive "
                    f"integers; got {threads_per_block!r}"
                )
            block_threads *= dimension
    else:
        raise ValueError(
            "block_threads must be a positive integer or dimension tuple; "
            f"got {threads_per_block!r}"
        )
    if not 1 <= block_threads <= 1024:
        raise ValueError(
            "block_threads must describe between 1 and 1024 threads; "
            f"got {threads_per_block!r}"
        )
    return block_threads


def _symbol_component(value):
    component = re.sub(r"\W+", "_", str(value)).strip("_")
    return component or "anon"


def _hash_symbol_value(hasher, value, depth=0):
    del depth
    hasher.update(
        repr(_numba_semantic_token(value)).encode("utf-8", errors="backslashreplace")
    )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _lto_ir_digest(lto_ir):
    if isinstance(lto_ir, str):
        data = lto_ir.encode("utf-8")
    else:
        data = bytes(lto_ir)
    return hashlib.sha1(data).hexdigest()


def _struct_size_alignment(member_types):
    offset = 0
    max_align = 1
    for member_type in member_types:
        member_size, member_align = _size_alignment_from_numba_type(member_type)
        offset = _align_up(offset, member_align)
        offset += member_size
        max_align = max(max_align, member_align)
    return _align_up(offset, max_align), max_align


def _registered_struct_member_types(numba_type):
    """Return matching CUDA/MLIR ``StructModel`` members, or ``None``."""

    from numba_cuda_mlir import models as mlir_models

    compat = _get_numba_mlir_datamodel_compat()
    mlir_ir = compat.mlir_ir

    def get_member_types(manager, model_type):
        try:
            model = manager.lookup(numba_type)
        except (KeyError, NotImplementedError, TypeError, ValueError):
            return None
        if not isinstance(model, model_type):
            return None
        return tuple(model.get_type(index) for index in range(model.field_count))

    cuda_manager = compat.cuda_data_manager.chain(compat.default_manager)
    cuda_members = get_member_types(cuda_manager, compat.cuda_struct_model)
    if cuda_members is None:
        return None

    try:
        current_context = mlir_ir.Context.current
    except ValueError:
        current_context = None

    if current_context is None:
        with mlir_ir.Context(), mlir_ir.Location.unknown():
            mlir_members = get_member_types(
                mlir_models.mlir_data_manager,
                mlir_models.StructModel,
            )
    else:
        try:
            current_location = mlir_ir.Location.current
        except ValueError:
            current_location = None
        if current_location is None:
            with mlir_ir.Location.unknown():
                mlir_members = get_member_types(
                    mlir_models.mlir_data_manager,
                    mlir_models.StructModel,
                )
        else:
            mlir_members = get_member_types(
                mlir_models.mlir_data_manager,
                mlir_models.StructModel,
            )

    if mlir_members != cuda_members:
        return None
    return cuda_members


def _size_alignment_from_numba_type(numba_type):
    from numba_cuda_mlir.type_defs.aggregate_types import AggregateType

    if isinstance(numba_type, (types.Boolean, types.BooleanLiteral)):
        return 1, 1
    if isinstance(numba_type, (types.Integer, types.IntegerLiteral)):
        size = max(1, numba_type.bitwidth // 8)
        return size, size
    if isinstance(numba_type, types.Float):
        size = max(1, numba_type.bitwidth // 8)
        return size, size
    if isinstance(numba_type, types.Complex):
        elem_size = max(1, numba_type.underlying_float.bitwidth // 8)
        return 2 * elem_size, elem_size
    if isinstance(numba_type, types.UniTuple):
        elem_size, elem_align = _size_alignment_from_numba_type(numba_type.dtype)
        return elem_size * numba_type.count, elem_align
    if isinstance(numba_type, AggregateType):
        if numba_type.is_bitfield_struct:
            return _size_alignment_from_numba_type(
                numba_type.get_bitfield_storage_type()
            )
        return _struct_size_alignment(
            field_type
            for field_name, field_type, _bit_width in numba_type.fields
            if field_name is not None
        )

    struct_member_types = _registered_struct_member_types(numba_type)
    if struct_member_types:
        return _struct_size_alignment(struct_member_types)

    dtype_name = getattr(numba_type, "name", str(numba_type))
    raise TypeError(
        "cuda.coop.numba_mlir cannot safely materialize CUB storage for dtype "
        f"{dtype_name}: exact ABI size and alignment are unavailable; use a "
        "compiler-native dtype, Numba-CUDA-MLIR AggregateType, or matching "
        "registered CUDA and MLIR StructModels with inspectable member types"
    )


def _ltoir_to_ptx(ltoir: bytes, *, name: str, cc: int) -> str:
    """Link one LTO-IR image to PTX for compile-time metadata inspection."""

    from cuda.core import Linker, LinkerOptions, ObjectCode

    ltoir_obj = ObjectCode.from_ltoir(ltoir, name=name)
    linker_options = LinkerOptions(
        arch=f"sm_{cc}",
        link_time_optimization=True,
        ptx=True,
    )
    linked_ptx = Linker(ltoir_obj, options=linker_options).link("ptx")
    return linked_ptx.code.decode("utf-8")


class TypeWrapper:
    def __init__(self, numba_type):
        self.lto_irs = []

        if numba_type in NUMBA_TYPES_TO_CPP:
            self.code = ""
            return

        # TypeWrapper needs LLVM ABI layout info (size/alignment) to materialize
        # the storage_t wrapper for user-defined dtypes.
        try:
            abi_context = mlir_target.target_context
            value_type = abi_context.get_value_type(numba_type)
            size = value_type.get_abi_size(abi_context.target_data)
            alignment = value_type.get_abi_alignment(abi_context.target_data)
        except Exception:
            size, alignment = _size_alignment_from_numba_type(numba_type)
        buf = StringIO()
        w = buf.write
        w(f"struct __align__({alignment}) storage_t\n")
        w("{\n")
        w(f"    char data[{size}];\n")
        w("};\n")

        self.code = buf.getvalue()


def numba_type_to_wrapper(numba_type: types.Type):
    return TypeWrapper(numba_type)


class Parameter:
    def __init__(self, is_output=False):
        self.is_output = is_output

    def __repr__(self) -> str:
        return f"Parameter(out={self.is_output})"

    def specialize(self, _):
        return self

    def is_provided_by_user(self):
        return not self.is_output

    def accepts_actual_type(self, actual_type, typing_context):
        return typing_context.can_convert(actual_type, self.dtype()) is not None


class Value(Parameter):
    def __init__(self, value_type, is_output=False):
        self.value_type = value_type
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Value(dtype={self.value_type}, out={self.is_output})"

    def dtype(self):
        return self.value_type

    @property
    def provider_dtype(self):
        """Return the scalar dtype received by the underlying provider."""

        return self.value_type

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_type) + " " + name

    def mangled_name(self):
        return f"{self.value_type}"


class ExactValue(Value):
    """A scalar whose compiler dtype must exactly match the provider ABI."""

    def __repr__(self) -> str:
        return f"ExactValue(dtype={self.value_type}, out={self.is_output})"

    def mangled_name(self):
        return f"Exact{self.value_type}"

    def accepts_actual_type(self, actual_type, typing_context):
        del typing_context
        literal_type = getattr(actual_type, "literal_type", actual_type)
        return literal_type == self.value_type


def _accepts_runtime_control_integer(actual_type) -> bool:
    actual_type = getattr(actual_type, "literal_type", actual_type)
    return (
        isinstance(actual_type, types.Integer)
        and actual_type.bitwidth <= 64
        and (actual_type.signed or actual_type.bitwidth <= 32)
    )


class BoundedInteger(Value):
    """A signed 64-bit integer checked before provider-type narrowing."""

    def __init__(self, provider_dtype, *, minimum, maximum):
        if not isinstance(provider_dtype, types.Integer) or isinstance(
            provider_dtype, types.IntegerLiteral
        ):
            raise TypeError("provider_dtype must be a non-literal integer dtype")
        for name, bound in (("minimum", minimum), ("maximum", maximum)):
            if not isinstance(bound, Integral) or isinstance(bound, bool):
                raise TypeError(f"{name} must be an integer")
        minimum = int(minimum)
        maximum = int(maximum)
        if minimum > maximum:
            raise ValueError("minimum must not exceed maximum")

        if provider_dtype.signed:
            provider_minimum = -(1 << (provider_dtype.bitwidth - 1))
            provider_maximum = (1 << (provider_dtype.bitwidth - 1)) - 1
        else:
            provider_minimum = 0
            provider_maximum = (1 << provider_dtype.bitwidth) - 1
        if not provider_minimum <= minimum <= maximum <= provider_maximum:
            raise ValueError(
                "bounded integer limits must fit the provider integer dtype"
            )
        if not -(1 << 63) <= minimum <= maximum <= (1 << 63) - 1:
            raise ValueError("bounded integer limits must fit the signed 64-bit ABI")

        self._provider_dtype = provider_dtype
        self.minimum = minimum
        self.maximum = maximum
        super().__init__(types.int64)

    def __repr__(self) -> str:
        return (
            "BoundedInteger("
            f"provider_dtype={self.provider_dtype}, minimum={self.minimum}, "
            f"maximum={self.maximum})"
        )

    @property
    def provider_dtype(self):
        return self._provider_dtype

    def mangled_name(self):
        bounds = internal_mangle_cpp(f"{self.minimum}_{self.maximum}")
        return f"Bounded{self.value_type}To{self.provider_dtype}_{bounds}"

    def accepts_actual_type(self, actual_type, typing_context):
        del typing_context
        return _accepts_runtime_control_integer(actual_type)


class PointerOffset(Value):
    def __init__(self, value_type, pointer_arg_index=0, static_value=None):
        if static_value is not None and (
            not isinstance(static_value, Integral) or isinstance(static_value, bool)
        ):
            raise TypeError("static pointer offset must be an integer")
        self.pointer_arg_index = pointer_arg_index
        self.static_value = None if static_value is None else int(static_value)
        super().__init__(value_type)

    def __repr__(self) -> str:
        return (
            f"PointerOffset(dtype={self.value_type}, "
            f"pointer_arg_index={self.pointer_arg_index}, "
            f"static_value={self.static_value})"
        )

    def mangled_name(self):
        static = (
            "Runtime"
            if self.static_value is None
            else f"Static{internal_mangle_cpp(str(self.static_value))}"
        )
        return f"Offset{self.value_type}{static}"

    def is_provided_by_user(self):
        return self.static_value is None

    def accepts_actual_type(self, actual_type, typing_context):
        del typing_context
        return _accepts_runtime_control_integer(actual_type)


class Pointer(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Pointer(dtype={self.value_dtype}, out={self.is_output})"

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_dtype) + "* " + name

    def dtype(self):
        # Provider wrappers receive a raw pointer and therefore cannot preserve
        # an arbitrary array stride. Require contiguous one-dimensional memory
        # so a sliced device view cannot silently produce adjacent-element
        # results.
        return types.Array(self.value_dtype, 1, "C")

    def mangled_name(self):
        return f"P{self.value_dtype}"


class DependentPointer(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentPointer(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Pointer(self.value_dtype.resolve(template_arguments), self.is_output)


class PointerReference(Pointer):
    def __init__(self, value_dtype, is_output=False):
        self.deref_on_call = True
        super().__init__(value_dtype, is_output)

    def __repr__(self) -> str:
        return f"PointerReference(dtype={self.value_dtype}, out={self.is_output})"


class DependentPointerReference(DependentPointer):
    def __init__(self, value_dtype, is_output=False):
        self.deref_on_call = True
        super().__init__(value_dtype, is_output)

    def __repr__(self) -> str:
        return (
            f"DependentPointerReference(dep={self.value_dtype}, out={self.is_output})"
        )

    def specialize(self, template_arguments):
        return PointerReference(
            self.value_dtype.resolve(template_arguments),
            self.is_output,
        )


class Reference(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Reference(dtype={self.value_dtype}, out={self.is_output})"

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_dtype) + "& " + name

    def dtype(self):
        return self.value_dtype

    def mangled_name(self):
        return f"R{self.value_dtype}"


class DependentReference(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentReference(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Reference(self.value_dtype.resolve(template_arguments), self.is_output)


class Array(Pointer):
    def __init__(self, value_dtype, size, is_output=False):
        self.size = size
        super().__init__(value_dtype, is_output)

    def __repr__(self) -> str:
        return (
            f"Array(dtype={self.value_dtype}, size={self.size}, out={self.is_output})"
        )

    def cpp_decl(self, name):
        return f"{numba_type_to_cpp(self.value_dtype)} (&{name})[{self.size}]"

    def dtype(self):
        return types.Array(self.value_dtype, 1, "C")

    def mangled_name(self):
        return f"A{self.size}_{self.value_dtype}"


class TransformedArray(Array):
    """Array input converted elementwise before the native CUB call."""

    def __init__(
        self,
        source_dtype,
        target_dtype,
        size,
        cpp_expression,
    ):
        if "{value}" not in cpp_expression:
            raise ValueError("array input transform must reference {value}")
        self.target_dtype = target_dtype
        self.cpp_expression = cpp_expression
        super().__init__(source_dtype, size, is_output=False)

    def __repr__(self) -> str:
        return (
            "TransformedArray("
            f"source_dtype={self.value_dtype}, target_dtype={self.target_dtype}, "
            f"size={self.size}, cpp_expression={self.cpp_expression!r})"
        )

    def mangled_name(self):
        expression_digest = hashlib.sha1(
            self.cpp_expression.encode("utf-8")
        ).hexdigest()[:12]
        return (
            f"TA{self.size}_{self.value_dtype}_to_{self.target_dtype}_"
            f"{expression_digest}"
        )


class SubstitutionFailure(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Dependency:
    def __init__(self, dep):
        self.dep = dep

    def resolve(self, template_arguments):
        if self.dep not in template_arguments:
            raise SubstitutionFailure(f"Template argument {self.dep} not provided")
        if template_arguments[self.dep] is None:
            raise SubstitutionFailure(f"Template argument {self.dep} is None")
        return template_arguments[self.dep]


class Constant:
    def __init__(self, val):
        self.val = val

    def resolve(self, _):
        return self.val


class CxxFunction(Parameter):
    def __init__(self, cpp, func_dtype):
        super().__init__()
        self.cpp = cpp
        self.func_dtype = func_dtype

    def __repr__(self) -> str:
        return f"CxxFunction(cpp={self.cpp})"

    def mangled_name(self):
        return f"cuda_coop_numba_mlir_F{internal_mangle_cpp(self.cpp)}"

    def dtype(self):
        return self.func_dtype

    def is_provided_by_user(self):
        return False


class DependentCxxOperator:
    def __init__(self, dep: Dependency, cpp: str):
        self.dep = dep
        self.cpp = cpp

    def specialize(self, template_arguments):
        dtype = self.dep.resolve(template_arguments)
        dtype_cpp = numba_type_to_cpp(dtype)
        source = f"<{self.dep.dep}>"
        target = f"<{dtype_cpp}>"
        match_count = self.cpp.count(source)
        if match_count != 1:
            raise ValueError(
                f"Expected exactly one {source!r} placeholder in C++ operator "
                f"{self.cpp!r}; found {match_count}."
            )
        cpp = self.cpp.replace(source, target, 1)
        return CxxFunction(cpp=f"{cpp}{{}}", func_dtype=dtype)


class DependentArray(Parameter):
    def __init__(self, value_dtype, size, is_output=False):
        self.value_dtype = value_dtype
        self.size = size
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentArray(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Array(
            self.value_dtype.resolve(template_arguments),
            self.size.resolve(template_arguments),
            self.is_output,
        )


class TemplateParameter:
    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name}"


def internal_mangle_cpp(cpp_name: str):
    """
    Substitutes non-alphanumeric characters in a C++ name with underscores,
    such that they can be used as valid, unique identifiers in C code.  This
    is for internal use only, and does not comport with C++ ABI name mangling.

    :param cpp_name: Supplies a C++ name to be mangled.
    :type cpp_name: str

    :return: Returns the mangled C++ name with non-alphanumeric characters
    substituted with underscores.
    :rtype: str

    Example
    -------

    .. code-block:: python

        >>> mangle("std::vector<int>")
        'std_vector_int_'
        >>> mangle("::cuda::std::min<::cuda::std::uint32_t>{}")
        '__cuda__std__min__cuda__std__uint32_t__'
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", cpp_name)


def mangle_symbol(name, template_parameters):
    return "_".join(
        [name]
        + [
            template_parameter.mangled_name()
            for template_parameter in template_parameters
        ]
    )


def war_introspection(fn, n):
    arglist = ", ".join(f"param{i}" for i in range(n))
    mod_str = dedent(f"""
    def impl({arglist}):
        return fn({arglist})
    """)
    mod_code = compile(mod_str, "<string>", "exec")
    func_code = mod_code.co_consts[0]
    return PyFunctionType(func_code, {"fn": fn})


def war_introspection_call(fn, n, returns_value):
    arglist = ", ".join(f"param{i}" for i in range(n))
    mod_lines = [f"def impl({arglist}):"]
    if returns_value:
        mod_lines.append(f"    return fn({arglist})")
    else:
        mod_lines.append(f"    fn({arglist})")
        mod_lines.append("    return")
    mod_str = "\n".join(mod_lines) + "\n"
    mod_code = compile(mod_str, "<string>", "exec")
    func_code = mod_code.co_consts[0]
    return PyFunctionType(func_code, {"fn": fn})


def war_introspection_call_with_transforms(fn, transforms, returns_value):
    n = len(transforms)
    arglist = ", ".join(f"param{i}" for i in range(n))
    mod_lines = [f"def impl({arglist}):"]
    for i, transform in enumerate(transforms):
        src_name = f"param{i}"
        dst_name = f"arg{i}"
        if transform == "ptr":
            mod_lines.append(f"    {dst_name} = types.ptr({src_name})")
        elif transform == "value":
            mod_lines.append(f"    {dst_name} = {src_name}")
        else:
            raise ValueError(f"Unexpected transform kind: {transform}")

    call_args = ", ".join(f"arg{i}" for i in range(n))
    if returns_value:
        mod_lines.append(f"    return fn({call_args})")
    else:
        mod_lines.append(f"    fn({call_args})")
        mod_lines.append("    return")

    mod_str = "\n".join(mod_lines) + "\n"
    mod_code = compile(mod_str, "<string>", "exec")
    func_code = mod_code.co_consts[0]
    return PyFunctionType(func_code, {"fn": fn, "types": types})


class Algorithm:
    def __init__(
        self,
        struct_name,
        method_name,
        c_name,
        includes,
        template_parameters,
        parameters,
        *,
        storage_abi,
        execution_scope,
        synchronization_scope,
        type_definitions=None,
        fake_return=False,
        output_by_reference=False,
        compile_context=None,
    ):
        self.struct_name = struct_name
        self.method_name = method_name
        self.c_name = (
            c_name
            if c_name.startswith("cuda_coop_numba_mlir_")
            else f"cuda_coop_numba_mlir_{c_name}"
        )
        self.includes = includes
        self.template_parameters = template_parameters
        self.parameters = parameters
        self.storage_abi = StorageABI(storage_abi)
        self.execution_scope = SynchronizationScope(execution_scope)
        self.synchronization_scope = SynchronizationScope(synchronization_scope)
        if self.synchronization_scope not in {
            SynchronizationScope.NONE,
            self.execution_scope,
        }:
            raise ValueError(
                "synchronization_scope must be NONE or match execution_scope"
            )
        self.type_definitions = type_definitions
        self.fake_return = fake_return
        self.output_by_reference = output_by_reference
        self._compile_context = compile_context
        self._temp_storage_bytes = None
        self._temp_storage_alignment = None
        self.threads = None
        self.block_threads = None
        self._private_symbol_digest = None
        self._private_symbol_key = None
        self._provider_compile_identity = None

    def __repr__(self) -> str:
        return f"{self.struct_name}::{self.method_name}{self.template_parameters}: {self.parameters}"

    def _symbol_base_name(self):
        namespace = (
            ""
            if self._private_symbol_digest is None
            else f"_S{self._private_symbol_digest}"
        )
        return f"{self.c_name}{namespace}_{self.method_name}"

    def _current_provider_compile_identity(self):
        device = cuda.get_current_device()
        cc_major, cc_minor = device.compute_capability
        cc = int(cc_major) * 10 + int(cc_minor)
        return nvrtc.compiler_identity(
            context=self._resolved_compile_context(),
            cc=cc,
            rdc=True,
            code="lto",
        )

    def _bind_provider_compile_identity(self, compile_identity=None):
        observed = (
            self._current_provider_compile_identity()
            if compile_identity is None
            else compile_identity
        )
        existing = self._provider_compile_identity
        if existing is not None and existing != observed:
            raise RuntimeError(
                "Provider artifacts were already qualified for a different "
                "compute capability, output kind, or compiler-option set."
            )
        self._provider_compile_identity = observed
        return observed

    def _qualify_private_symbols(
        self, *, threads=None, block_threads=None, compile_identity=None
    ):
        """Give each emitted provider interface a deterministic private namespace."""

        compile_identity = self._bind_provider_compile_identity(compile_identity)
        key = algo_coalesce_key(
            self,
            threads=threads,
            block_threads=block_threads,
        )
        if self._private_symbol_key is not None:
            if self._private_symbol_key != key:
                raise RuntimeError(
                    "Provider symbols were already qualified for a different "
                    "thread configuration or compilation target."
                )
            return compile_identity

        self._private_symbol_key = key
        self._private_symbol_digest = hashlib.sha1(
            repr(key).encode("utf-8")
        ).hexdigest()[:20]
        return compile_identity

    def _resolved_compile_context(self):
        if self._compile_context is None:
            self._compile_context = nvrtc.resolve_compile_context()
        return self._compile_context

    def mangled_name(self, parameters):
        return mangle_symbol(self._symbol_base_name(), parameters)

    def specialize(self, template_arguments):
        # No partial specializations for now
        template_list = []
        for template_parameter in self.template_parameters:
            if template_parameter.name not in template_arguments:
                raise ValueError(
                    f"Template argument {template_parameter.name} not provided"
                )
            template_argument = template_arguments[template_parameter.name]
            if isinstance(template_argument, int):
                template_list.append(str(template_argument))
            elif isinstance(template_argument, str):
                template_list.append(template_argument)
            else:
                template_list.append(numba_type_to_cpp(template_argument))
        template_list = ", ".join(template_list)

        # '::cuda::std::int32_t, 32' -> __cuda__std__int32_t__32
        mangle = internal_mangle_cpp(template_list)

        specialized_parameters = []
        for method in self.parameters:
            specialized_signature = []
            try:
                for parameter in method:
                    specialized_signature.append(
                        parameter.specialize(template_arguments)
                    )
                specialized_parameters.append(specialized_signature)
            except SubstitutionFailure:
                pass  # Substitution failure is not an error

        specialized_name = f"{self.struct_name}<{template_list}>"
        specialized = Algorithm(
            specialized_name,
            self.method_name,
            self.c_name + mangle,
            self.includes,
            [],
            specialized_parameters,
            storage_abi=self.storage_abi,
            execution_scope=self.execution_scope,
            synchronization_scope=self.synchronization_scope,
            type_definitions=self.type_definitions,
            fake_return=self.fake_return,
            output_by_reference=self.output_by_reference,
            compile_context=self._compile_context,
        )
        specialized.threads = template_arguments.get(
            "LOGICAL_WARP_THREADS",
            template_arguments.get("VIRTUAL_WARP_THREADS"),
        )
        return specialized

    @property
    def temp_storage_bytes(self):
        if self._temp_storage_bytes is None:
            raise RuntimeError(
                "Temporary storage bytes not computed yet.  Call get_lto_ir() first."
            )
        return self._temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        if self._temp_storage_alignment is None:
            raise RuntimeError(
                "Temporary storage alignment not computed yet.  "
                "Call get_lto_ir() first."
            )
        return self._temp_storage_alignment

    @staticmethod
    def _ignore_codegen_param(param):
        return isinstance(param, CxxFunction) or (
            isinstance(param, PointerOffset) and param.static_value is not None
        )

    def _emit_abi_wrapper(
        self,
        w,
        method,
        exported_name,
        internal_name,
        temp_storage_type_name="temp_storage_t",
        temp_storage_param_pid=0,
    ):
        output_param = None
        user_params = []
        for pid, param in enumerate(method):
            if self._ignore_codegen_param(param):
                continue
            if param.is_output:
                if output_param is not None:
                    raise ValueError("Multiple output parameters not supported")
                output_param = (pid, param)
            else:
                user_params.append((pid, param))

        abi_param_decls = ["void *__ret"]
        body_lines = []
        call_args_by_pid = {}

        for user_pid, (pid, param) in enumerate(user_params):
            abi_name = f"abi_param_{user_pid}"
            if isinstance(param, (Pointer, Array, PointerReference)):
                abi_param_decls.append(f"void *{abi_name}")
                cast_name = f"abi_cast_{pid}"
                if (
                    pid == temp_storage_param_pid
                    and isinstance(param, Pointer)
                    and param.value_dtype == types.uint8
                ):
                    # Non-alloc wrappers receive raw temporary storage explicitly.
                    body_lines.append(
                        "    "
                        f"{temp_storage_type_name} *{cast_name} = "
                        f"reinterpret_cast<{temp_storage_type_name} *>({abi_name});"
                    )
                else:
                    pointee_type = numba_type_to_cpp(param.value_dtype)
                    body_lines.append(
                        f"    {pointee_type} *{cast_name} = reinterpret_cast<{pointee_type} *>({abi_name});"
                    )
                call_args_by_pid[pid] = cast_name
            elif isinstance(param, Reference):
                value_type = numba_type_to_cpp(param.value_dtype)
                abi_param_decls.append(f"{value_type} {abi_name}")
                local_name = f"abi_ref_{pid}"
                body_lines.append(f"    {value_type} {local_name} = {abi_name};")
                call_args_by_pid[pid] = local_name
            else:
                value_type = numba_type_to_cpp(param.dtype())
                abi_param_decls.append(f"{value_type} {abi_name}")
                call_args_by_pid[pid] = abi_name

        output_var = None
        output_cpp_type = None
        if output_param is not None:
            output_cpp_type = numba_type_to_cpp(output_param[1].dtype())
            output_var = "abi_out"
            body_lines.append(f"    {output_cpp_type} {output_var};")
            call_args_by_pid[output_param[0]] = output_var

        call_args = [
            call_args_by_pid[pid]
            for pid, param in enumerate(method)
            if not self._ignore_codegen_param(param)
        ]
        args_csv = ", ".join(call_args)
        if args_csv:
            body_lines.append(f"    {internal_name}({args_csv});")
        else:
            body_lines.append(f"    {internal_name}();")

        if output_var is not None:
            body_lines.append(
                f"    *reinterpret_cast<{output_cpp_type} *>(__ret) = {output_var};"
            )
        body_lines.append("    return 0;")

        abi_params_csv = ", ".join(abi_param_decls)
        w(f'extern "C" __device__ int {exported_name}__abi({abi_params_csv}) {{\n')
        for line in body_lines:
            w(f"{line}\n")
        w("}\n\n")

    def _temp_storage_symbol_names(self):
        suffix = internal_mangle_cpp(self._symbol_base_name())
        return (
            f"cuda_coop_numba_mlir_temp_storage_bytes_{suffix}",
            f"cuda_coop_numba_mlir_temp_storage_alignment_{suffix}",
        )

    def _collect_support_ltoirs_and_udf_declarations(self):
        lto_irs = []
        udf_declarations = OrderedDict()

        if self.type_definitions:
            for type_definition in self.type_definitions:
                lto_irs.extend(type_definition.lto_irs)

        return _dedupe_ltoirs(lto_irs), udf_declarations

    def _source_code(self, threads=None, block_threads=None, *, compile_identity=None):
        self._qualify_private_symbols(
            threads=threads,
            block_threads=block_threads,
            compile_identity=compile_identity,
        )
        support_lto_irs, udf_declarations = (
            self._collect_support_ltoirs_and_udf_declarations()
        )

        algorithm_name = self.struct_name
        includes = self.includes or []
        type_definitions = self.type_definitions or []
        method_name = self.method_name

        alias_suffix = internal_mangle_cpp(self._symbol_base_name())
        algorithm_type_name = f"algorithm_t_{alias_suffix}"
        temp_storage_type_name = None
        temp_storage_symbols = ()
        if self.storage_abi is StorageABI.LEADING_POINTER:
            temp_storage_type_name = f"temp_storage_t_{alias_suffix}"
            temp_storage_symbols = self._temp_storage_symbol_names()

        buf = StringIO()
        w = buf.write

        w("#include <cuda/std/cstdint>\n")
        for include in includes:
            w(f"#include <{include}>\n")
        for type_definition in type_definitions:
            w(f"{type_definition.code}\n")
        w("\n")
        for decl in udf_declarations.values():
            w(f"{decl}\n")
        w("\n")

        w(f"using {algorithm_type_name} = cub::{algorithm_name};\n")
        if temp_storage_type_name is not None:
            temp_storage_bytes_symbol, temp_storage_alignment_symbol = (
                temp_storage_symbols
            )
            w(
                f"using {temp_storage_type_name} = typename "
                f"{algorithm_type_name}::TempStorage;\n"
            )
            w(
                "__device__ constexpr unsigned "
                f"{temp_storage_bytes_symbol} = sizeof({temp_storage_type_name});\n"
            )
            w(
                "__device__ constexpr unsigned "
                f"{temp_storage_alignment_symbol} = "
                f"alignof({temp_storage_type_name});\n"
            )

        src = buf.getvalue()

        for method in self.parameters:
            param_decls = []
            func_decls = []
            param_args = []
            out_param = None

            param_arg_positions_by_pid = {}
            provider_parameters = (
                method[1:] if self.storage_abi is StorageABI.LEADING_POINTER else method
            )
            for pid, param in enumerate(provider_parameters):
                if isinstance(param, CxxFunction):
                    param_args.append(param.cpp)
                else:
                    name = f"param_{pid}"
                    if isinstance(param, PointerOffset):
                        offset_expression = name
                        if param.static_value is None:
                            param_decls.append(param.cpp_decl(name))
                        else:
                            offset_expression = str(param.static_value)
                        pointer_arg_pos = param_arg_positions_by_pid.get(
                            param.pointer_arg_index
                        )
                        if pointer_arg_pos is None:
                            raise ValueError(
                                "PointerOffset must reference an earlier pointer "
                                "parameter."
                            )
                        param_args[pointer_arg_pos] = (
                            f"({param_args[pointer_arg_pos]} + {offset_expression})"
                        )
                        continue
                    if isinstance(param, BoundedInteger):
                        checked_name = f"checked_{name}"
                        param_decls.append(param.cpp_decl(name))
                        func_decls.append(
                            f"if ({name} < {param.minimum} || "
                            f"{name} > {param.maximum}) {{"
                        )
                        func_decls.append('  asm volatile("trap;" : : :);')
                        func_decls.append("}")
                        provider_cpp = numba_type_to_cpp(param.provider_dtype)
                        func_decls.append(
                            f"{provider_cpp} {checked_name} = "
                            f"static_cast<{provider_cpp}>({name});"
                        )
                        param_arg = checked_name
                    elif isinstance(param, TransformedArray):
                        source_type = numba_type_to_cpp(param.value_dtype)
                        target_type = numba_type_to_cpp(param.target_dtype)
                        transformed_name = f"transformed_{name}"
                        param_decls.append(f"{source_type} *{name}")
                        func_decls.append(
                            f"{target_type} {transformed_name}[{param.size}];"
                        )
                        for index in range(param.size):
                            expression = param.cpp_expression.format(
                                value=f"{name}[{index}]"
                            )
                            func_decls.append(
                                f"{transformed_name}[{index}] = {expression};"
                            )
                        param_arg = transformed_name
                    elif isinstance(param, Array):
                        value_type = numba_type_to_cpp(param.value_dtype)
                        param_decls.append(f"{value_type} *{name}")
                        param_arg = (
                            f"*reinterpret_cast<{value_type} (*)[{param.size}]>({name})"
                        )
                    else:
                        param_decls.append(param.cpp_decl(name))
                        if getattr(param, "deref_on_call", False):
                            param_arg = f"*{name}"
                        else:
                            param_arg = name
                    if not self.fake_return and param.is_output:
                        if out_param is not None:
                            raise ValueError("Multiple output parameters not supported")
                        out_param = name
                        if self.output_by_reference:
                            param_arg_positions_by_pid[pid] = len(param_args)
                            param_args.append(param_arg)
                    else:
                        param_arg_positions_by_pid[pid] = len(param_args)
                        param_args.append(param_arg)

            provide_alloc_version = self.storage_abi is StorageABI.LEADING_POINTER
            if provide_alloc_version:
                assert temp_storage_type_name is not None
                logical_width = None
                if self.execution_scope is SynchronizationScope.BLOCK:
                    storage = f"__shared__ {temp_storage_type_name} temp_storage;"
                elif self.execution_scope is SynchronizationScope.WARP:
                    logical_width = _validate_logical_warp_threads(
                        threads if threads is not None else self.threads
                    )
                    resolved_block_threads = _normalize_block_threads(
                        block_threads
                        if block_threads is not None
                        else self.block_threads
                    )
                    if resolved_block_threads % logical_width != 0:
                        raise ValueError(
                            "warp-scoped provider width must divide the exact "
                            f"block size; got width={logical_width} and "
                            f"block_threads={resolved_block_threads}"
                        )
                    instances = resolved_block_threads // logical_width
                    storage = (
                        "unsigned __coop_thread_rank = threadIdx.x + blockDim.x * "
                        "(threadIdx.y + blockDim.y * threadIdx.z);\n"
                        "    "
                        f"__shared__ {temp_storage_type_name} temp_storages"
                        f"[{instances}];\n"
                        f"    {temp_storage_type_name} &temp_storage = temp_storages"
                        f"[__coop_thread_rank / {logical_width}];"
                    )
                elif self.execution_scope is SynchronizationScope.NONE:
                    storage = f"{temp_storage_type_name} temp_storage;"
                else:
                    raise NotImplementedError(
                        "cuda.coop.numba_mlir provider execution scope "
                        f"{self.execution_scope.value!r} has no emitter"
                    )
                if self.synchronization_scope is SynchronizationScope.WARP:
                    assert logical_width is not None
                    sync = (
                        "__syncwarp();"
                        if logical_width == 32
                        else (
                            "__syncwarp((((1u << "
                            f"{logical_width}"
                            ") - 1u) << (((__coop_thread_rank & 31) / "
                            f"{logical_width}"
                            f") * {logical_width})));"
                        )
                    )
                else:
                    sync = {
                        SynchronizationScope.NONE: "",
                        SynchronizationScope.BLOCK: "__syncthreads();",
                    }.get(self.synchronization_scope)
                if sync is None:
                    raise NotImplementedError(
                        "cuda.coop.numba_mlir provider synchronization scope "
                        f"{self.synchronization_scope.value!r} has no emitter"
                    )
            else:
                storage = ""
                sync = ""

            buf = StringIO()
            w = buf.write

            mangled_name = self.mangled_name(method)
            param_decls_csv = ", ".join(param_decls)
            param_args_csv = ", ".join(param_args)

            if provide_alloc_version:
                w(f'extern "C" __device__ void {mangled_name}_alloc(')
                w(f"{param_decls_csv}) {{\n")
                w(f"    {storage}\n")
                for decl in func_decls:
                    w(f"    {decl}\n")
                if out_param and not self.output_by_reference:
                    w(f"    {out_param} = ")
                else:
                    w("    ")
                w(f"{algorithm_type_name}(temp_storage).")
                w(f"{method_name}({param_args_csv});\n")
                w(f"    {sync}\n")
                w("}\n")
                self._emit_abi_wrapper(
                    w,
                    method[1:],
                    exported_name=f"{mangled_name}_alloc",
                    internal_name=f"{mangled_name}_alloc",
                    temp_storage_type_name=temp_storage_type_name,
                    temp_storage_param_pid=None,
                )

            w(f'extern "C" __device__ void {mangled_name}(')
            if self.storage_abi is StorageABI.LEADING_POINTER:
                assert temp_storage_type_name is not None
                w(f"{temp_storage_type_name} *temp_storage")
                if param_decls_csv:
                    w(", ")
            w(f"{param_decls_csv}) {{\n")
            for decl in func_decls:
                w(f"    {decl}\n")
            if out_param and not self.output_by_reference:
                w(f"    {out_param} = ")
            else:
                w("    ")
            if self.storage_abi is StorageABI.LEADING_POINTER:
                w(f"{algorithm_type_name}(*temp_storage).")
            else:
                w(f"{algorithm_type_name}().")
            w(f"{method_name}({param_args_csv});\n")
            w("}\n\n")
            self._emit_abi_wrapper(
                w,
                method,
                exported_name=mangled_name,
                internal_name=mangled_name,
                temp_storage_type_name=(
                    "temp_storage_t"
                    if temp_storage_type_name is None
                    else temp_storage_type_name
                ),
                temp_storage_param_pid=(
                    0 if self.storage_abi is StorageABI.LEADING_POINTER else None
                ),
            )

            src += buf.getvalue()

        return (
            src,
            support_lto_irs,
            temp_storage_symbols,
            udf_declarations,
        )

    def _make_lto_ir_cache_key(
        self, threads=None, block_threads=None, *, compile_identity=None
    ):
        resolved_threads = threads if threads is not None else self.threads
        resolved_block_threads = (
            block_threads if block_threads is not None else self.block_threads
        )
        if self.execution_scope is SynchronizationScope.WARP:
            resolved_threads = _validate_logical_warp_threads(resolved_threads)
            resolved_block_threads = _normalize_block_threads(resolved_block_threads)
            if resolved_block_threads % resolved_threads != 0:
                raise ValueError(
                    "warp-scoped provider width must divide the exact block "
                    f"size; got width={resolved_threads} and "
                    f"block_threads={resolved_block_threads}"
                )
        compile_identity = self._bind_provider_compile_identity(compile_identity)
        return (
            self.storage_abi.value,
            self.execution_scope.value,
            self.synchronization_scope.value,
            resolved_threads,
            resolved_block_threads,
            compile_identity,
        )

    def get_lto_ir(self, threads=None, block_threads=None, *, compile_identity=None):
        # With no explicit identity, re-query the current device even when an
        # artifact is already cached. Reusing one Algorithm across devices must
        # fail closed instead of returning LTO compiled for the earlier target.
        compile_identity = self._bind_provider_compile_identity(compile_identity)
        cache_key = self._make_lto_ir_cache_key(
            threads,
            block_threads,
            compile_identity=compile_identity,
        )
        existing = self.__dict__.get("lto_irs")
        if existing is not None:
            existing_key = self.__dict__.get("_lto_ir_cache_key")
            if existing_key != cache_key:
                raise RuntimeError(
                    "LTO IR was already compiled for a different thread "
                    "configuration or compilation target."
                )
            return existing

        src, support_lto_irs, temp_storage_symbols, _ = self._source_code(
            threads=threads,
            block_threads=block_threads,
            compile_identity=compile_identity,
        )

        cc = int(compile_identity[0])
        _, ltoir = nvrtc.compile(
            cpp=src,
            cc=cc,
            rdc=True,
            code="lto",
            context=self._resolved_compile_context(),
        )
        ltoir_blob = bytes(ltoir)
        ptx = _ltoir_to_ptx(ltoir_blob, name=self.c_name, cc=cc)

        lto_irs = list(support_lto_irs)
        lto_irs.append(ltoir_blob)

        from ._compiler._artifacts import find_unsigned

        if temp_storage_symbols:
            abi_globals = {
                symbol: find_unsigned(symbol, ptx) for symbol in temp_storage_symbols
            }
            self._temp_storage_bytes = abi_globals[temp_storage_symbols[0]]
            self._temp_storage_alignment = abi_globals[temp_storage_symbols[1]]
        else:
            self._temp_storage_bytes = 0
            self._temp_storage_alignment = 1
        self.__dict__["_lto_ir_cache_key"] = cache_key
        self.__dict__["_link_input_suffixes"] = [".ltoir"] * len(lto_irs)
        self.__dict__["lto_irs"] = lto_irs
        return lto_irs

    def codegen(self, func_to_overload):
        if len(self.template_parameters):
            raise ValueError("Cannot generate codegen for a template")

        overloads = []
        for method in self.parameters:
            overloads.append(
                self.codegen_method(
                    func_to_overload,
                    method,
                    self.mangled_name(method),
                )
            )
            if self.storage_abi is StorageABI.LEADING_POINTER:
                overloads.append(
                    self.codegen_method(
                        func_to_overload,
                        method[1:],
                        self.mangled_name(method) + "_alloc",
                    )
                )
        return tuple(overloads)

    def codegen_method(self, func_to_overload, method, mangled_name):
        if len(self.template_parameters):
            raise ValueError("Cannot generate codegen for a template")

        def ignore_param(param):
            # C++ functions do not require additional argument handling.
            ignore = isinstance(param, CxxFunction) or (
                isinstance(param, PointerOffset) and param.static_value is not None
            )
            return ignore

        num_user_provided_params = sum(
            [param.is_provided_by_user() for param in method]
        )
        returns_value = any(param.is_output for param in method)
        link_files = getattr(func_to_overload, "files", [])
        expected_input_parameters = []
        abi_input_types = []
        arg_transforms = []
        ret_type = types.void

        for param in method:
            if ignore_param(param) or param.is_output:
                if not ignore_param(param) and param.is_output:
                    if ret_type is not types.void:
                        raise ValueError("Multiple output parameters not supported")
                    ret_type = param.dtype()
                continue

            expected_input_parameters.append(param)
            if isinstance(param, (Pointer, Array, PointerReference)):
                abi_input_types.append(types.CPointer(types.none))
                arg_transforms.append("ptr")
            else:
                abi_input_types.append(param.dtype())
                arg_transforms.append("value")

        abi_name = f"{mangled_name}__abi"
        extern_fn = ExternFunction(
            abi_name,
            signature(ret_type, *abi_input_types),
            link=link_files,
        )

        def algorithm_impl(*args):
            if len(args) != len(expected_input_parameters):
                return None
            typingctx = mlir_target.typing_context
            for actual_type, parameter in zip(args, expected_input_parameters):
                if not parameter.accepts_actual_type(actual_type, typingctx):
                    return None
            impl = war_introspection_call_with_transforms(
                extern_fn, arg_transforms, returns_value=returns_value
            )
            if link_files:
                setattr(impl, "__numba_cuda_mlir_link__", link_files)
            return impl

        wrapped_algorithm_impl = war_introspection(
            algorithm_impl, num_user_provided_params
        )
        if link_files:
            setattr(wrapped_algorithm_impl, "__numba_cuda_mlir_link__", link_files)
        return make_overload_template(
            func_to_overload,
            wrapped_algorithm_impl,
            {"no_cpython_wrapper": True, "nopython": True},
            strict=True,
            inline="always",
            prefer_literal=False,
            base=_NumbaCudaMlirOverloadFunctionTemplate,
        )


def _dedupe_ltoirs(lto_irs):
    seen = set()
    deduped = []
    for lto_ir in lto_irs:
        key = _lto_ir_digest(lto_ir)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(lto_ir)
    return deduped


class _SharedTempFile:
    def __init__(self, temp_file: BinaryIO):
        self._temp_file = temp_file
        self._temp_file_finalizer = weakref.finalize(
            self, _cleanup_temp_files, (temp_file.name,)
        )

    @property
    def name(self):
        return self._temp_file.name


def _collect_udf_decls(algo):
    del algo
    return OrderedDict()


def _collect_extra_ltoirs(algo):
    extras = []
    if algo.type_definitions:
        for type_definition in algo.type_definitions:
            extras.extend(getattr(type_definition, "lto_irs", []))
    for method in algo.parameters:
        for param in method:
            lto_ir = getattr(param, "ltoir", None)
            if lto_ir is not None:
                extras.append(lto_ir)
    return _dedupe_ltoirs(extras)


def _param_coalesce_key(param):
    if isinstance(param, TransformedArray):
        return (
            "TransformedArray",
            str(param.value_dtype),
            str(param.target_dtype),
            param.size,
            param.cpp_expression,
        )
    if isinstance(param, Array):
        return ("Array", str(param.value_dtype), param.size, param.is_output)
    if isinstance(param, Pointer):
        return (
            "Pointer",
            str(param.value_dtype),
            param.is_output,
            getattr(param, "deref_on_call", False),
        )
    if isinstance(param, Reference):
        return ("Reference", str(param.value_dtype), param.is_output)
    if isinstance(param, PointerOffset):
        return (
            "PointerOffset",
            str(param.value_type),
            param.pointer_arg_index,
            param.static_value,
            param.is_output,
        )
    if isinstance(param, BoundedInteger):
        return (
            "BoundedInteger",
            str(param.value_type),
            str(param.provider_dtype),
            param.minimum,
            param.maximum,
            param.is_output,
        )
    if isinstance(param, ExactValue):
        return ("ExactValue", str(param.value_type), param.is_output)
    if isinstance(param, Value):
        return ("Value", str(param.value_type), param.is_output)
    if isinstance(param, CxxFunction):
        return ("CxxFunction", param.cpp, str(param.func_dtype))
    return (type(param).__name__, repr(param))


def algo_coalesce_key(algo, *, threads=None, block_threads=None):
    type_defs = []
    for type_definition in getattr(algo, "type_definitions", None) or []:
        code = getattr(type_definition, "code", None) or ""
        lto_irs = getattr(type_definition, "lto_irs", []) or []
        lto_ir_hashes = tuple(_lto_ir_digest(lto_ir) for lto_ir in lto_irs)
        type_defs.append((code, lto_ir_hashes))

    params_key = tuple(
        tuple(_param_coalesce_key(param) for param in method)
        for method in getattr(algo, "parameters", [])
    )

    resolved_threads = threads
    if resolved_threads is None:
        resolved_threads = getattr(algo, "threads", None)
    resolved_block_threads = block_threads
    if resolved_block_threads is None:
        resolved_block_threads = getattr(algo, "block_threads", None)

    return (
        getattr(algo, "struct_name", None),
        getattr(algo, "method_name", None),
        getattr(algo, "c_name", None),
        tuple(getattr(algo, "includes", None) or []),
        tuple(type_defs),
        params_key,
        resolved_threads,
        resolved_block_threads,
        getattr(algo, "fake_return", None),
        getattr(algo, "output_by_reference", None),
        getattr(algo, "storage_abi", None),
        getattr(algo, "execution_scope", None),
        getattr(algo, "synchronization_scope", None),
        getattr(algo, "_compile_context", None),
        getattr(algo, "_provider_compile_identity", None),
    )


def _strip_source_preamble(src, algo, udf_decls):
    body = src
    body = body.replace("#include <cuda/std/cstdint>\n", "", 1)
    for include in algo.includes or []:
        body = body.replace(f"#include <{include}>\n", "", 1)
    for type_definition in algo.type_definitions or []:
        code = getattr(type_definition, "code", "")
        if code:
            body = body.replace(f"{code}\n", "", 1)
    for decl in udf_decls.values():
        body = body.replace(f"{decl}\n", "", 1)
    return body.lstrip()


def prepare_ltoir_bundle(
    algorithms,
    *,
    bundle_name=None,
    allow_single=False,
    threads_by_algo=None,
    block_threads_by_algo=None,
):
    if not algorithms:
        return None

    threads_by_algo = threads_by_algo or {}
    block_threads_by_algo = block_threads_by_algo or {}

    deduped_by_id = OrderedDict()
    for algo in algorithms:
        deduped_by_id[id(algo)] = algo
    all_algos = list(deduped_by_id.values())

    compile_contexts = {algo._resolved_compile_context() for algo in all_algos}
    if len(compile_contexts) != 1:
        raise RuntimeError("coalesced providers must use one exact compiler context")
    compile_context = next(iter(compile_contexts))
    device = cuda.get_current_device()
    cc_major, cc_minor = device.compute_capability
    cc = int(cc_major) * 10 + int(cc_minor)
    compile_identity = nvrtc.compiler_identity(
        context=compile_context,
        cc=cc,
        rdc=True,
        code="lto",
    )

    key_to_rep = OrderedDict()
    rep_for_algo_id = {}
    for algo in all_algos:
        threads = threads_by_algo.get(id(algo), getattr(algo, "threads", None))
        block_threads = block_threads_by_algo.get(
            id(algo), getattr(algo, "block_threads", None)
        )
        algo._qualify_private_symbols(
            threads=threads,
            block_threads=block_threads,
            compile_identity=compile_identity,
        )
        key = algo_coalesce_key(algo, threads=threads, block_threads=block_threads)
        rep = key_to_rep.setdefault(key, algo)
        rep_for_algo_id[id(algo)] = rep

    reps = list(key_to_rep.values())
    if len(reps) < 2 and not allow_single:
        return None

    rep_src = {}
    rep_symbols = {}
    rep_udf_decls = {}
    includes = OrderedDict()
    type_defs = OrderedDict()
    udf_decls = OrderedDict()

    for rep in reps:
        threads = threads_by_algo.get(id(rep), getattr(rep, "threads", None))
        block_threads = block_threads_by_algo.get(
            id(rep), getattr(rep, "block_threads", None)
        )
        src, _support_lto_irs, symbols, udf = rep._source_code(
            threads=threads,
            block_threads=block_threads,
            compile_identity=compile_identity,
        )
        rep_src[id(rep)] = src
        rep_symbols[id(rep)] = symbols
        rep_udf_decls[id(rep)] = udf

        for include in rep.includes or []:
            includes.setdefault(include, None)
        for type_definition in rep.type_definitions or []:
            code = getattr(type_definition, "code", "")
            if code:
                type_defs.setdefault(code, None)
        for decl in udf.values():
            udf_decls.setdefault(decl, None)

    buf = StringIO()
    w = buf.write
    w("#include <cuda/std/cstdint>\n")
    for include in includes.keys():
        w(f"#include <{include}>\n")
    for code in type_defs.keys():
        w(f"{code}\n")
    w("\n")
    if udf_decls:
        for decl in udf_decls.keys():
            w(f"{decl}\n")
        w("\n")

    bodies = []
    for rep in reps:
        body = _strip_source_preamble(
            rep_src[id(rep)],
            rep,
            rep_udf_decls[id(rep)],
        )
        bodies.append(body.rstrip() + "\n")

    src = buf.getvalue() + "\n".join(bodies)

    if bundle_name is None:
        bundle_name = f"cuda_coop_numba_mlir_bundle_{hashlib.sha1(src.encode('utf-8')).hexdigest()[:16]}"

    _, ltoir = nvrtc.compile(
        cpp=src,
        cc=cc,
        rdc=True,
        code="lto",
        context=compile_context,
    )
    ltoir_blob = bytes(ltoir)
    ptx = _ltoir_to_ptx(ltoir_blob, name=bundle_name, cc=cc)

    symbols = []
    for rep in reps:
        symbols.extend(rep_symbols[id(rep)])

    from ._compiler._artifacts import find_unsigned, make_binary_tempfile

    abi_globals = {symbol: find_unsigned(symbol, ptx) for symbol in symbols}

    bundle_temp_file = _SharedTempFile(make_binary_tempfile(ltoir_blob, ".ltoir"))

    for algo in all_algos:
        rep = rep_for_algo_id[id(algo)]
        storage_symbols = rep_symbols[id(rep)]
        extras = _collect_extra_ltoirs(algo)
        if storage_symbols:
            bytes_symbol, alignment_symbol = storage_symbols
            algo._temp_storage_bytes = abi_globals[bytes_symbol]
            algo._temp_storage_alignment = abi_globals[alignment_symbol]
        else:
            algo._temp_storage_bytes = 0
            algo._temp_storage_alignment = 1
        algo._lto_irs = extras
        algo._precompiled_ltoir_files = (bundle_temp_file,)
        algo.__dict__["_lto_ir_cache_key"] = algo._make_lto_ir_cache_key(
            threads=threads_by_algo.get(id(algo), getattr(algo, "threads", None)),
            block_threads=block_threads_by_algo.get(
                id(algo), getattr(algo, "block_threads", None)
            ),
            compile_identity=compile_identity,
        )
        algo.__dict__["_link_input_suffixes"] = [".ltoir"] * len(extras)
        algo.__dict__["lto_irs"] = extras

    return ltoir_blob


def make_invocable_from_specialization(
    specialization: Algorithm, *, threads=None, block_threads=None
):
    if threads is not None:
        specialization.threads = threads
    if block_threads is not None:
        specialization.block_threads = block_threads

    compile_identity = specialization._qualify_private_symbols(
        threads=threads,
        block_threads=block_threads,
    )

    collector = _COOP_SPECIALIZATION_COLLECTOR.get()
    if collector is not None:
        collector.append((specialization, threads, block_threads))
        return specialization

    from ._compiler._artifacts import make_binary_tempfile

    lto_irs = specialization.get_lto_ir(
        threads=threads,
        block_threads=block_threads,
        compile_identity=compile_identity,
    )
    precompiled_temp_files = list(
        getattr(specialization, "_precompiled_ltoir_files", ())
    )
    suffixes = specialization.__dict__.get(
        "_link_input_suffixes", [".ltoir"] * len(lto_irs)
    )
    if len(suffixes) != len(lto_irs):
        raise RuntimeError("Cached coop link input metadata is inconsistent.")
    owned_temp_files = [
        make_binary_tempfile(link_input, suffix)
        for link_input, suffix in zip(lto_irs, suffixes)
    ]
    return Invocable(
        temp_files=precompiled_temp_files + owned_temp_files,
        owned_temp_files=owned_temp_files,
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )


class Invocable:
    def __init__(
        self,
        temp_files: Sequence[BinaryIO],
        temp_storage_bytes: int,
        temp_storage_alignment: int,
        algorithm: Algorithm,
        owned_temp_files: Sequence[BinaryIO] | None = None,
    ):
        self._temp_files = temp_files
        self._owned_temp_files = () if owned_temp_files is None else owned_temp_files
        self._temp_storage_bytes = temp_storage_bytes
        self._temp_storage_alignment = temp_storage_alignment
        self.specialization = algorithm
        self._temp_file_finalizer = weakref.finalize(
            self, _cleanup_temp_files, tuple(v.name for v in self._owned_temp_files)
        )
        self._numba_type = None

    @property
    def temp_storage_bytes(self):
        return self._temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        return self._temp_storage_alignment

    @property
    def storage_abi(self):
        return self.specialization.storage_abi

    @property
    def execution_scope(self):
        return self.specialization.execution_scope

    @property
    def synchronization_scope(self):
        return self.specialization.synchronization_scope

    @property
    def files(self):
        return [v.name for v in self._temp_files]

    @property
    def _numba_type_(self):
        """Return a compiler-local callable type without global registration."""

        if self._numba_type is None:
            templates = self.specialization.codegen(self)
            self._numba_type = types.Function(templates)
        return self._numba_type

    def __call__(self, *args):
        raise RuntimeError(
            "__call__ should not be called directly outside of a numba_cuda_mlir.cuda.jit(...) kernel."
        )


def _cleanup_temp_files(paths):
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
