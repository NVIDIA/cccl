# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import os
import re
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from io import StringIO
from textwrap import dedent
from types import FunctionType as PyFunctionType
from typing import BinaryIO, Callable, Literal, Mapping, Optional, Sequence

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.extending import overload, typing_registry
from numba_cuda_mlir.types import signature

from . import _nvrtc as nvrtc

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
_MethodName = Literal["construct", "assign"]
_MethodsMapping = Mapping[_MethodName, Callable]
_SUPPORTED_WARP_THREADS = frozenset({1, 2, 4, 8, 16, 32})

_COOP_SPECIALIZATION_COLLECTOR: ContextVar[
    list[tuple[object, int | None, int | tuple[int, ...] | None]] | None
] = ContextVar("cuda_coop_numba_mlir_specialization_collector", default=None)
_DEVICE_LTOIR_CACHE: dict[tuple[int, str, tuple[tuple[str, object], ...]], bytes] = {}


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


def _validate_warp_threads(threads):
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads not in _SUPPORTED_WARP_THREADS
    ):
        supported = ", ".join(str(value) for value in sorted(_SUPPORTED_WARP_THREADS))
        raise ValueError(
            "threads_in_warp must be one of the supported CUB logical warp "
            f"sizes {{{supported}}}; got {threads!r}."
        )
    return threads


def _normalize_block_threads(threads_per_block):
    if threads_per_block is None:
        return None
    if isinstance(threads_per_block, bool):
        raise ValueError(
            f"threads_per_block must be a positive integer or dim tuple; got {threads_per_block!r}."
        )
    if isinstance(threads_per_block, int):
        block_threads = threads_per_block
    elif isinstance(threads_per_block, (tuple, list)):
        if not (1 <= len(threads_per_block) <= 3):
            raise ValueError(
                "threads_per_block dim tuple must have one, two, or three entries; "
                f"got {threads_per_block!r}."
            )
        block_threads = 1
        for dim in threads_per_block:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    "threads_per_block dim tuple entries must be positive integers; "
                    f"got {threads_per_block!r}."
                )
            block_threads *= dim
    else:
        raise ValueError(
            f"threads_per_block must be a positive integer or dim tuple; got {threads_per_block!r}."
        )
    if block_threads <= 0 or block_threads > 1024:
        raise ValueError(
            "threads_per_block must describe between 1 and 1024 total threads; "
            f"got {threads_per_block!r}."
        )
    return block_threads


def _symbol_component(value):
    component = re.sub(r"\W+", "_", str(value)).strip("_")
    return component or "anon"


def _hash_symbol_value(hasher, value, depth=0):
    hasher.update(type(value).__module__.encode("utf-8", errors="backslashreplace"))
    hasher.update(b".")
    hasher.update(type(value).__qualname__.encode("utf-8", errors="backslashreplace"))
    hasher.update(b":")

    if depth > 2:
        hasher.update(f"id:{id(value):x}".encode("ascii"))
        return

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        hasher.update(repr(value).encode("utf-8", errors="backslashreplace"))
    elif isinstance(value, (tuple, list)):
        hasher.update(b"[")
        for item in value:
            _hash_symbol_value(hasher, item, depth + 1)
            hasher.update(b",")
        hasher.update(b"]")
    elif isinstance(value, dict):
        hasher.update(b"{")
        for key, item in sorted(value.items(), key=lambda entry: repr(entry[0])):
            _hash_symbol_value(hasher, key, depth + 1)
            hasher.update(b":")
            _hash_symbol_value(hasher, item, depth + 1)
            hasher.update(b",")
        hasher.update(b"}")
    elif hasattr(value, "__code__"):
        nested_code = value.__code__
        hasher.update(getattr(value, "__module__", "unknown").encode("utf-8"))
        hasher.update(b".")
        hasher.update(
            getattr(
                value, "__qualname__", getattr(value, "__name__", type(value).__name__)
            ).encode("utf-8", errors="backslashreplace")
        )
        hasher.update(nested_code.co_code)
        hasher.update(repr(nested_code.co_consts).encode("utf-8"))
        hasher.update(f":id:{id(value):x}".encode("ascii"))
    elif hasattr(value, "__name__") and type(value).__module__ == "builtins":
        hasher.update(value.__name__.encode("utf-8", errors="backslashreplace"))
    else:
        try:
            representation = repr(value)
        except Exception:
            representation = f"<unrepr {id(value):x}>"
        hasher.update(representation.encode("utf-8", errors="backslashreplace"))
        hasher.update(f":id:{id(value):x}".encode("ascii"))


def _callable_symbol_component(fn):
    module = getattr(fn, "__module__", "unknown")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__))
    code = getattr(fn, "__code__", None)
    if code is None:
        digest = f"id_{id(fn):x}"
    else:
        hasher = hashlib.sha1()
        hasher.update(code.co_code)
        hasher.update(repr(code.co_consts).encode("utf-8"))
        hasher.update(str(code.co_argcount).encode("ascii"))
        hasher.update(str(code.co_kwonlyargcount).encode("ascii"))
        hasher.update(b"defaults:")
        _hash_symbol_value(hasher, getattr(fn, "__defaults__", None))
        hasher.update(b"kwdefaults:")
        _hash_symbol_value(hasher, getattr(fn, "__kwdefaults__", None))
        for name, cell in zip(code.co_freevars, getattr(fn, "__closure__", None) or ()):
            hasher.update(b"freevar:")
            hasher.update(name.encode("utf-8", errors="backslashreplace"))
            hasher.update(b"=")
            try:
                _hash_symbol_value(hasher, cell.cell_contents)
            except ValueError:
                hasher.update(b"<empty>")
        globals_dict = getattr(fn, "__globals__", {})
        for name in code.co_names:
            if name in globals_dict:
                hasher.update(b"global:")
                hasher.update(name.encode("utf-8", errors="backslashreplace"))
                hasher.update(b"=")
                _hash_symbol_value(hasher, globals_dict[name])
        digest = hasher.hexdigest()[:12]
    return _symbol_component(f"{module}_{qualname}_{digest}")


def _python_operator_symbol_name(
    binary_op, ret_dtype, arg_dtypes, *, stateful_name=None
):
    if stateful_name is None:
        op_component = _callable_symbol_component(binary_op)
    else:
        op_component = (
            f"{_symbol_component(stateful_name)}_"
            f"{_symbol_component(binary_op.__name__)}"
        )
    signature = f"{_symbol_component(ret_dtype)}__" + "_".join(
        _symbol_component(dtype) for dtype in arg_dtypes
    )
    return f"cuda_coop_numba_mlir_F{op_component}_{signature}"


def _python_type_method_symbol_name(numba_type, method_name, method):
    return (
        f"cuda_coop_numba_mlir_type_{_symbol_component(numba_type)}_"
        f"{method_name}_{_callable_symbol_component(method)}"
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

    from numba_cuda_mlir._mlir import ir as mlir_ir
    from numba_cuda_mlir.models import StructModel as MlirStructModel
    from numba_cuda_mlir.models import mlir_data_manager
    from numba_cuda_mlir.numba_cuda.datamodel import default_manager
    from numba_cuda_mlir.numba_cuda.datamodel.models import (
        StructModel as CudaStructModel,
    )
    from numba_cuda_mlir.numba_cuda.models import cuda_data_manager

    def get_member_types(manager, model_type):
        try:
            model = manager.lookup(numba_type)
        except (KeyError, NotImplementedError, TypeError, ValueError):
            return None
        if not isinstance(model, model_type):
            return None
        return tuple(model.get_type(index) for index in range(model.field_count))

    cuda_manager = cuda_data_manager.chain(default_manager)
    cuda_members = get_member_types(cuda_manager, CudaStructModel)
    if cuda_members is None:
        return None

    try:
        current_context = mlir_ir.Context.current
    except ValueError:
        current_context = None

    if current_context is None:
        with mlir_ir.Context(), mlir_ir.Location.unknown():
            mlir_members = get_member_types(mlir_data_manager, MlirStructModel)
    else:
        try:
            current_location = mlir_ir.Location.current
        except ValueError:
            current_location = None
        if current_location is None:
            with mlir_ir.Location.unknown():
                mlir_members = get_member_types(mlir_data_manager, MlirStructModel)
        else:
            mlir_members = get_member_types(mlir_data_manager, MlirStructModel)

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


def method_to_signature(numba_type, method):
    ptr_type = types.CPointer(numba_type)
    if method == "construct":
        return signature(types.void, ptr_type)
    elif method == "assign":
        return signature(types.void, ptr_type, ptr_type)
    else:
        raise ValueError("Unexpected method {}".format(method))


def _compile_device_ltoir(fn, *, sig, abi_info):
    # numba-cuda-mlir's compile() mutates dispatcher targetoptions when passed
    # an existing dispatcher. Compile from the underlying Python function so
    # reused device dispatchers do not accumulate ABI/linkage state across
    # multiple coop specializations.
    py_func = getattr(fn, "py_func", fn)
    cache_key = (id(py_func), repr(sig), tuple(sorted(abi_info.items())))
    cached = _DEVICE_LTOIR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ltoir, _ = cuda.compile(
        py_func,
        sig=sig,
        output="ltoir",
        abi_info=abi_info,
        forceinline=True,
    )
    if isinstance(ltoir, str):
        ltoir_blob = ltoir.encode("utf-8")
    else:
        ltoir_blob = bytes(ltoir)
    _DEVICE_LTOIR_CACHE[cache_key] = ltoir_blob
    return ltoir_blob


@lru_cache(maxsize=None)
def _adapt_python_operator_abi(
    py_func: Callable,
    *,
    stateful: bool,
    return_by_pointer: bool,
    arguments_by_pointer: tuple[bool, ...],
) -> Callable:
    """Adapt value-level Python operators to the C++ aggregate pointer ABI."""

    if not return_by_pointer and not any(arguments_by_pointer):
        return py_func

    if len(arguments_by_pointer) not in {1, 2}:
        raise TypeError(
            "cuda.coop.numba_mlir aggregate Python operators must accept one "
            "or two value arguments"
        )

    device_op = cuda.jit(device=True, inline="always")(py_func)
    first_by_pointer = arguments_by_pointer[0]

    if len(arguments_by_pointer) == 1:
        if stateful and return_by_pointer:

            def adapted(state, first, result):
                first_value = first[0] if first_by_pointer else first
                result[0] = device_op(state, first_value)

        elif stateful:

            def adapted(state, first):
                first_value = first[0] if first_by_pointer else first
                return device_op(state, first_value)

        elif return_by_pointer:

            def adapted(first, result):
                first_value = first[0] if first_by_pointer else first
                result[0] = device_op(first_value)

        else:

            def adapted(first):
                first_value = first[0] if first_by_pointer else first
                return device_op(first_value)

        return adapted

    second_by_pointer = arguments_by_pointer[1]
    if stateful and return_by_pointer:

        def adapted(state, first, second, result):
            first_value = first[0] if first_by_pointer else first
            second_value = second[0] if second_by_pointer else second
            result[0] = device_op(state, first_value, second_value)

    elif stateful:

        def adapted(state, first, second):
            first_value = first[0] if first_by_pointer else first
            second_value = second[0] if second_by_pointer else second
            return device_op(state, first_value, second_value)

    elif return_by_pointer:

        def adapted(first, second, result):
            first_value = first[0] if first_by_pointer else first
            second_value = second[0] if second_by_pointer else second
            result[0] = device_op(first_value, second_value)

    else:

        def adapted(first, second):
            first_value = first[0] if first_by_pointer else first
            second_value = second[0] if second_by_pointer else second
            return device_op(first_value, second_value)

    return adapted


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
    def __init__(self, numba_type, methods: _MethodsMapping):
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
        method_symbols = {
            method_name: _python_type_method_symbol_name(
                numba_type, method_name, method
            )
            for method_name, method in methods.items()
        }

        buf = StringIO()
        w = buf.write
        if "construct" in methods:
            construct_name = method_symbols["construct"]
            w(f'extern "C" __device__ void {construct_name}(void *ptr);\n')
        if "assign" in methods:
            assign_name = method_symbols["assign"]
            w(
                f'extern "C" __device__ void {assign_name}'
                "(void *dst, const void *src);\n"
            )

        w(f"struct __align__({alignment}) storage_t\n")
        w("{\n")
        if "construct" in methods:
            w("    __device__ storage_t() {\n")
            w(f"        {construct_name}(data);\n")
            w("    }\n")
        if "assign" in methods:
            w("    __device__ storage_t& operator=(const storage_t &rhs) {\n")
            w(f"        {assign_name}(data, rhs.data);\n")
            w("        return *this;\n")
            w("    }\n")
        w(f"    char data[{size}];\n")
        w("};\n")

        self.code = buf.getvalue()

        for method in methods:
            method_fn = methods[method]
            ltoir = _compile_device_ltoir(
                method_fn,
                sig=method_to_signature(numba_type, method),
                abi_info={"abi_name": method_symbols[method]},
            )
            self.lto_irs.append(ltoir)


def numba_type_to_wrapper(
    numba_type: types.Type, methods: Optional[_MethodsMapping] = None
):
    if methods is None:
        methods = {}
    for method in methods:
        if method not in ["construct", "assign"]:
            raise ValueError("Unexpected method {}".format(method))
    return TypeWrapper(numba_type, methods)


class Parameter:
    def __init__(self, is_output=False):
        self.is_output = is_output

    def __repr__(self) -> str:
        return f"Parameter(out={self.is_output})"

    def specialize(self, _):
        return self

    def is_provided_by_user(self):
        return not self.is_output


class Value(Parameter):
    def __init__(self, value_type, is_output=False):
        self.value_type = value_type
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Value(dtype={self.value_type}, out={self.is_output})"

    def dtype(self):
        return self.value_type

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_type) + " " + name

    def mangled_name(self):
        return f"{self.value_type}"


class PointerOffset(Value):
    def __init__(self, value_type, pointer_arg_index=0):
        self.pointer_arg_index = pointer_arg_index
        super().__init__(value_type)

    def __repr__(self) -> str:
        return (
            f"PointerOffset(dtype={self.value_type}, "
            f"pointer_arg_index={self.pointer_arg_index})"
        )

    def mangled_name(self):
        return f"Offset{self.value_type}"


class Pointer(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Pointer(dtype={self.value_dtype}, out={self.is_output})"

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_dtype) + "* " + name

    def dtype(self):
        return types.Array(self.value_dtype, 1, "A")

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


class StatefulFunction:
    """Stateful Python device callable used as a cooperative primitive operator."""

    def __init__(self, op, dtype, name=None):
        self.op = op
        self.dtype = dtype
        self.name = name


class StatefulOperator:
    def __init__(self, mangled_name, op_type, ret_cpp_type, arg_cpp_types, ltoir):
        self.op_type = op_type
        self.name = mangled_name
        self.ltoir = ltoir
        self.ret_cpp_type = ret_cpp_type
        self.arg_cpp_types = arg_cpp_types
        self.is_output = False

    def mangled_name(self):
        return f"{self.name}"

    def forward_decl(self):
        arg_decls = ["char *state"]
        if self.ret_cpp_type == "storage_t":
            ret_decl = "void"
        else:
            ret_decl = self.ret_cpp_type
        for arg in self.arg_cpp_types:
            arg_decls.append("const void*" if arg == "storage_t" else arg)
        if self.ret_cpp_type == "storage_t":
            arg_decls.append("void*")
        return f'extern "C" __device__ {ret_decl} {self.mangled_name()}({", ".join(arg_decls)});'

    def cpp_decl(self, name):
        return f"char* {name}_state"

    def dtype(self):
        return types.Array(self.op_type, 1, "C")

    def wrap_decl(self, name):
        param_decls = []
        param_refs = []
        for aid, arg_type in enumerate(self.arg_cpp_types):
            arg_name = f"wp_{aid}"
            param_decls.append(f"const {arg_type}& {arg_name}")
            param_refs.append("&" + arg_name if arg_type == "storage_t" else arg_name)

        param_decls_csv = ", ".join(param_decls)
        param_refs_csv = ", ".join(param_refs)

        buf = StringIO()
        w = buf.write
        state_name = f"{name}_state"
        w(f"auto {name} = [{state_name}]({param_decls_csv}) {{\n")
        if self.ret_cpp_type == "storage_t":
            w(f"    {self.ret_cpp_type} result;\n")
            call_args = ", ".join([state_name, *param_refs, "&result"])
            w(f"    {self.mangled_name()}({call_args});\n")
            w("    return result;\n")
        else:
            w(f"    return {self.mangled_name()}(\n")
            w(f"        {state_name}, {param_refs_csv});\n")
        w("};\n")
        src = buf.getvalue()
        return src

    def is_provided_by_user(self):
        return True


class StatelessOperator:
    def __init__(self, mangled_name, ret_cpp_type, arg_cpp_types, ltoir):
        self.name = mangled_name
        self.ltoir = ltoir
        self.ret_cpp_type = ret_cpp_type
        self.arg_cpp_types = arg_cpp_types
        self.is_output = False

    def mangled_name(self):
        return f"{self.name}"

    def forward_decl(self):
        arg_decls = []
        if self.ret_cpp_type == "storage_t":
            ret_decl = "void"
        else:
            ret_decl = self.ret_cpp_type
        for arg in self.arg_cpp_types:
            arg_decls.append("const void*" if arg == "storage_t" else arg)
        if self.ret_cpp_type == "storage_t":
            arg_decls.append("void*")
        mangled_name = self.mangled_name()
        arg_decls_csv = ", ".join(arg_decls)
        return f'extern "C" __device__ {ret_decl} {mangled_name}({arg_decls_csv});'

    def wrap_decl(self, name):
        param_decls = []
        param_refs = []
        for aid, arg_type in enumerate(self.arg_cpp_types):
            arg_name = f"wp_{aid}"
            param_decls.append(f"const {arg_type}& {arg_name}")
            param_refs.append("&" + arg_name if arg_type == "storage_t" else arg_name)

        buf = StringIO()
        w = buf.write
        param_decls_csv = ", ".join(param_decls)
        param_refs_csv = ", ".join(param_refs)
        mangled_name = self.mangled_name()

        w(f"auto {name} = []({param_decls_csv}) {{\n")
        if self.ret_cpp_type == "storage_t":
            w("    storage_t result;\n")
            call_args = ", ".join([*param_refs, "&result"])
            w(f"    {mangled_name}({call_args});\n")
            w("    return result;\n")
        else:
            w(f"    return {mangled_name}({param_refs_csv});\n")
        w("};\n")
        src = buf.getvalue()
        return src

    def is_provided_by_user(self):
        return False


class DependentPythonOperator:
    def __init__(self, ret_dtype, arg_dtypes, op):
        self.ret_dtype = ret_dtype
        self.arg_dtypes = arg_dtypes
        self.op = op

    def specialize(self, template_arguments):
        op = self.op.resolve(template_arguments)
        ret_dtype = self.ret_dtype.resolve(template_arguments)
        ret_cpp_type = numba_type_to_cpp(ret_dtype)
        ret_numba_type = (
            types.CPointer(ret_dtype) if ret_cpp_type == "storage_t" else ret_dtype
        )
        arg_cpp_types = []
        arg_dtypes = []
        arg_numba_types = []
        for arg in self.arg_dtypes:
            arg_dtype = arg.resolve(template_arguments)
            arg_cpp_type = numba_type_to_cpp(arg_dtype)
            arg_cpp_types.append(arg_cpp_type)
            arg_dtypes.append(str(arg_dtype))
            arg_numba_types.append(
                types.CPointer(arg_dtype) if arg_cpp_type == "storage_t" else arg_dtype
            )

        if isinstance(op, StatefulFunction):
            binary_op = op.op.__call__
            compile_op = _adapt_python_operator_abi(
                getattr(binary_op, "py_func", binary_op),
                stateful=True,
                return_by_pointer=ret_cpp_type == "storage_t",
                arguments_by_pointer=tuple(
                    arg_cpp_type == "storage_t" for arg_cpp_type in arg_cpp_types
                ),
            )
            op_name = op.name
            if op_name is None:
                op_name = (
                    f"{type(op.op).__module__}.{type(op.op).__qualname__}.{id(op.op):x}"
                )
            mangled_name = _python_operator_symbol_name(
                binary_op,
                ret_dtype,
                arg_dtypes,
                stateful_name=op_name,
            )
            if ret_cpp_type == "storage_t":
                binary_op_signature = signature(
                    types.void,
                    types.CPointer(op.dtype),
                    *arg_numba_types,
                    ret_numba_type,
                )
            else:
                binary_op_signature = signature(
                    ret_numba_type, types.CPointer(op.dtype), *arg_numba_types
                )
            abi_info = {"abi_name": mangled_name}
            ltoir = _compile_device_ltoir(
                compile_op, sig=binary_op_signature, abi_info=abi_info
            )
            return StatefulOperator(
                mangled_name, op.dtype, ret_cpp_type, arg_cpp_types, ltoir
            )
        else:
            binary_op = op
            compile_op = _adapt_python_operator_abi(
                getattr(binary_op, "py_func", binary_op),
                stateful=False,
                return_by_pointer=ret_cpp_type == "storage_t",
                arguments_by_pointer=tuple(
                    arg_cpp_type == "storage_t" for arg_cpp_type in arg_cpp_types
                ),
            )
            mangled_name = _python_operator_symbol_name(
                binary_op,
                ret_dtype,
                arg_dtypes,
            )
            if ret_cpp_type == "storage_t":
                binary_op_signature = signature(
                    types.void, *arg_numba_types, ret_numba_type
                )
            else:
                binary_op_signature = signature(ret_numba_type, *arg_numba_types)
            abi_info = {"abi_name": mangled_name}
            ltoir = _compile_device_ltoir(
                compile_op, sig=binary_op_signature, abi_info=abi_info
            )
            return StatelessOperator(mangled_name, ret_cpp_type, arg_cpp_types, ltoir)


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
        type_definitions=None,
        fake_return=False,
        output_by_reference=False,
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
        self.type_definitions = type_definitions
        self.fake_return = fake_return
        self.output_by_reference = output_by_reference
        self._temp_storage_bytes = None
        self._temp_storage_alignment = None
        self.threads = None
        self.block_threads = None
        self._private_symbol_digest = None
        self._private_symbol_key = None

    def __repr__(self) -> str:
        return f"{self.struct_name}::{self.method_name}{self.template_parameters}: {self.parameters}"

    def _symbol_base_name(self):
        namespace = (
            ""
            if self._private_symbol_digest is None
            else f"_S{self._private_symbol_digest}"
        )
        return f"{self.c_name}{namespace}_{self.method_name}"

    def _qualify_private_symbols(self, *, threads=None, block_threads=None):
        """Give each emitted provider interface a deterministic private namespace."""

        key = algo_coalesce_key(
            self,
            threads=threads,
            block_threads=block_threads,
        )
        if self._private_symbol_key is not None:
            if self._private_symbol_key != key:
                raise RuntimeError(
                    "Provider symbols were already qualified for a different "
                    "threads/block_threads configuration."
                )
            return

        self._private_symbol_key = key
        self._private_symbol_digest = hashlib.sha1(
            repr(key).encode("utf-8")
        ).hexdigest()[:20]

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
            type_definitions=self.type_definitions,
            fake_return=self.fake_return,
            output_by_reference=self.output_by_reference,
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
        return isinstance(param, (StatelessOperator, CxxFunction))

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
            if isinstance(param, (Pointer, Array, PointerReference, StatefulOperator)):
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
                elif isinstance(param, StatefulOperator):
                    body_lines.append(
                        f"    char *{cast_name} = reinterpret_cast<char *>({abi_name});"
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

        for method in self.parameters:
            for param in method:
                if isinstance(param, (StatelessOperator, StatefulOperator)):
                    if param.name not in udf_declarations:
                        udf_declarations[param.name] = param.forward_decl()
                        lto_irs.append(param.ltoir)

        return _dedupe_ltoirs(lto_irs), udf_declarations

    def _source_code(self, threads=None, block_threads=None):
        self._qualify_private_symbols(
            threads=threads,
            block_threads=block_threads,
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
        temp_storage_type_name = f"temp_storage_t_{alias_suffix}"
        temp_storage_bytes_symbol, temp_storage_alignment_symbol = (
            self._temp_storage_symbol_names()
        )

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
        w(
            f"using {temp_storage_type_name} = typename {algorithm_type_name}::TempStorage;\n"
        )
        w(
            "__device__ constexpr unsigned "
            f"{temp_storage_bytes_symbol} = sizeof({temp_storage_type_name});\n"
        )
        w(
            "__device__ constexpr unsigned "
            f"{temp_storage_alignment_symbol} = alignof({temp_storage_type_name});\n"
        )

        src = buf.getvalue()

        for method in self.parameters:
            param_decls = []
            func_decls = []
            param_args = []
            out_param = None

            param_arg_positions_by_pid = {}
            for pid, param in enumerate(method[1:]):
                if isinstance(param, StatelessOperator):
                    func_decls.append(param.wrap_decl(f"param_{pid}"))
                    param_args.append(f"param_{pid}")
                elif isinstance(param, StatefulOperator):
                    name = f"param_{pid}"
                    func_decls.append(param.wrap_decl(name))
                    param_args.append(name)
                    param_decls.append(param.cpp_decl(name))
                elif isinstance(param, CxxFunction):
                    param_args.append(param.cpp)
                else:
                    name = f"param_{pid}"
                    if isinstance(param, PointerOffset):
                        param_decls.append(param.cpp_decl(name))
                        pointer_arg_pos = param_arg_positions_by_pid.get(
                            param.pointer_arg_index
                        )
                        if pointer_arg_pos is None:
                            raise ValueError(
                                "PointerOffset must reference an earlier pointer "
                                "parameter."
                            )
                        param_args[pointer_arg_pos] = (
                            f"({param_args[pointer_arg_pos]} + {name})"
                        )
                        continue
                    if isinstance(param, TransformedArray):
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

            threads_for_method = threads if threads is not None else self.threads
            if self.struct_name.startswith("Warp"):
                if threads_for_method is None:
                    raise ValueError("Warp algorithm must specify number of threads")
                threads_for_method = _validate_warp_threads(threads_for_method)
                block_threads_for_method = (
                    block_threads if block_threads is not None else self.block_threads
                )
                if block_threads_for_method is None:
                    block_threads_for_method = 1024
                block_threads_for_method = _normalize_block_threads(
                    block_threads_for_method
                )
                warp_storage_count = max(
                    1,
                    (block_threads_for_method + threads_for_method - 1)
                    // threads_for_method,
                )
                provide_alloc_version = True
                storage = (
                    "unsigned __coop_thread_rank = threadIdx.x + blockDim.x * "
                    "(threadIdx.y + blockDim.y * threadIdx.z);\n"
                    "    "
                    f"__shared__ {temp_storage_type_name} temp_storages"
                    f"[{warp_storage_count}];\n"
                    f"    {temp_storage_type_name} &temp_storage = temp_storages"
                    f"[__coop_thread_rank / {threads_for_method}];\n"
                )
                if threads_for_method == 32:
                    sync = "__syncwarp();"
                else:
                    sync = (
                        "__syncwarp((((1u << "
                        f"{threads_for_method}"
                        ") - 1u) << (((__coop_thread_rank & 31) / "
                        f"{threads_for_method}"
                        f") * {threads_for_method})));"
                    )
            elif self.struct_name.startswith("Block"):
                provide_alloc_version = True
                storage = f"__shared__ {temp_storage_type_name} temp_storage;"
                sync = "__syncthreads();"
            else:
                provide_alloc_version = False
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
            w(f"{temp_storage_type_name} *temp_storage, ")
            w(f"{param_decls_csv}) {{\n")
            for decl in func_decls:
                w(f"    {decl}\n")
            if out_param and not self.output_by_reference:
                w(f"    {out_param} = ")
            else:
                w("    ")
            w(f"{algorithm_type_name}(*temp_storage).")
            w(f"{method_name}({param_args_csv});\n")
            w("}\n\n")
            self._emit_abi_wrapper(
                w,
                method,
                exported_name=mangled_name,
                internal_name=mangled_name,
                temp_storage_type_name=temp_storage_type_name,
            )

            src += buf.getvalue()

        return (
            src,
            support_lto_irs,
            (temp_storage_bytes_symbol, temp_storage_alignment_symbol),
            udf_declarations,
        )

    def _make_lto_ir_cache_key(self, threads=None, block_threads=None):
        resolved_threads = threads if threads is not None else self.threads
        resolved_block_threads = (
            block_threads if block_threads is not None else self.block_threads
        )
        if self.struct_name.startswith("Warp"):
            if resolved_threads is None:
                raise ValueError("Warp algorithm must specify number of threads")
            resolved_threads = _validate_warp_threads(resolved_threads)
            if resolved_block_threads is None:
                resolved_block_threads = 1024
            resolved_block_threads = _normalize_block_threads(resolved_block_threads)
        return resolved_threads, resolved_block_threads

    def get_lto_ir(self, threads=None, block_threads=None):
        cache_key = self._make_lto_ir_cache_key(threads, block_threads)
        existing = self.__dict__.get("lto_irs")
        if existing is not None:
            existing_key = self.__dict__.get("_lto_ir_cache_key")
            if existing_key != cache_key:
                raise RuntimeError(
                    "LTO IR was already compiled for a different "
                    "threads/block_threads configuration."
                )
            return existing

        src, support_lto_irs, temp_storage_symbols, _ = self._source_code(
            threads=threads, block_threads=block_threads
        )

        device = cuda.get_current_device()
        cc_major, cc_minor = device.compute_capability
        cc = cc_major * 10 + cc_minor
        _, ltoir = nvrtc.compile(cpp=src, cc=cc, rdc=True, code="lto")
        ltoir_blob = bytes(ltoir)
        ptx = _ltoir_to_ptx(ltoir_blob, name=self.c_name, cc=cc)

        lto_irs = list(support_lto_irs)
        lto_irs.append(ltoir_blob)

        from ._common import find_unsigned

        abi_globals = {
            symbol: find_unsigned(symbol, ptx) for symbol in temp_storage_symbols
        }
        self._temp_storage_bytes = abi_globals[temp_storage_symbols[0]]
        self._temp_storage_alignment = abi_globals[temp_storage_symbols[1]]
        self.__dict__["_lto_ir_cache_key"] = cache_key
        self.__dict__["_link_input_suffixes"] = [".ltoir"] * len(lto_irs)
        self.__dict__["lto_irs"] = lto_irs
        return lto_irs

    def codegen(self, func_to_overload):
        if len(self.template_parameters):
            raise ValueError("Cannot generate codegen for a template")

        for method in self.parameters:
            self.codegen_method(func_to_overload, method, self.mangled_name(method))
            self.codegen_method(
                func_to_overload, method[1:], self.mangled_name(method) + "_alloc"
            )

    def codegen_method(self, func_to_overload, method, mangled_name):
        if len(self.template_parameters):
            raise ValueError("Cannot generate codegen for a template")

        def ignore_param(param):
            # Stateless operators and C++ functions do not require any
            # additional argument handling or code generation, so we can
            # safely ignore them during this code gen phase.
            ignore = isinstance(param, (StatelessOperator, CxxFunction))
            return ignore

        num_user_provided_params = sum(
            [param.is_provided_by_user() for param in method]
        )
        returns_value = any(param.is_output for param in method)
        link_files = getattr(func_to_overload, "files", [])
        expected_input_types = []
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

            expected_input_types.append(param.dtype())
            if isinstance(param, (Pointer, Array, PointerReference, StatefulOperator)):
                abi_input_types.append(types.CPointer(types.none))
                arg_transforms.append("ptr")
            else:
                abi_input_types.append(param.dtype())
                arg_transforms.append("value")

        abi_name = f"{mangled_name}__abi"
        extern_fn = cuda.declare_device(
            abi_name, signature(ret_type, *abi_input_types), link=link_files
        )

        def algorithm_impl(*args):
            if len(args) != len(expected_input_types):
                return None
            typingctx = mlir_target.typing_context
            for actual_type, expected_type in zip(args, expected_input_types):
                if typingctx.can_convert(actual_type, expected_type) is None:
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
        overload(func_to_overload, inline="always", typing_registry=typing_registry)(
            wrapped_algorithm_impl
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
    udf_decls = OrderedDict()
    for method in algo.parameters:
        for param in method:
            if isinstance(param, (StatelessOperator, StatefulOperator)):
                if param.name not in udf_decls:
                    udf_decls[param.name] = param.forward_decl()
    return udf_decls


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
            param.is_output,
        )
    if isinstance(param, Value):
        return ("Value", str(param.value_type), param.is_output)
    if isinstance(param, StatelessOperator):
        return (
            "StatelessOperator",
            param.name,
            param.ret_cpp_type,
            tuple(param.arg_cpp_types),
            _lto_ir_digest(param.ltoir),
        )
    if isinstance(param, StatefulOperator):
        return (
            "StatefulOperator",
            param.name,
            str(param.op_type),
            param.ret_cpp_type,
            tuple(param.arg_cpp_types),
            _lto_ir_digest(param.ltoir),
        )
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

    normalize_threads = getattr(algo, "_make_lto_ir_cache_key", None)
    if callable(normalize_threads):
        resolved_threads, resolved_block_threads = normalize_threads(
            threads,
            block_threads,
        )
    else:
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
            threads=threads, block_threads=block_threads
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

    device = cuda.get_current_device()
    cc_major, cc_minor = device.compute_capability
    cc = cc_major * 10 + cc_minor

    if bundle_name is None:
        bundle_name = f"cuda_coop_numba_mlir_bundle_{hashlib.sha1(src.encode('utf-8')).hexdigest()[:16]}"

    _, ltoir = nvrtc.compile(cpp=src, cc=cc, rdc=True, code="lto")
    ltoir_blob = bytes(ltoir)
    ptx = _ltoir_to_ptx(ltoir_blob, name=bundle_name, cc=cc)

    symbols = []
    for rep in reps:
        bytes_symbol, alignment_symbol = rep_symbols[id(rep)]
        symbols.append(bytes_symbol)
        symbols.append(alignment_symbol)

    from ._common import find_unsigned, make_binary_tempfile

    abi_globals = {symbol: find_unsigned(symbol, ptx) for symbol in symbols}

    bundle_temp_file = _SharedTempFile(make_binary_tempfile(ltoir_blob, ".ltoir"))

    for algo in all_algos:
        rep = rep_for_algo_id[id(algo)]
        bytes_symbol, alignment_symbol = rep_symbols[id(rep)]
        extras = _collect_extra_ltoirs(algo)
        algo._temp_storage_bytes = abi_globals[bytes_symbol]
        algo._temp_storage_alignment = abi_globals[alignment_symbol]
        algo._lto_irs = extras
        algo._precompiled_ltoir_files = (bundle_temp_file,)
        algo.__dict__["_lto_ir_cache_key"] = algo._make_lto_ir_cache_key(
            threads=threads_by_algo.get(id(algo), getattr(algo, "threads", None)),
            block_threads=block_threads_by_algo.get(
                id(algo), getattr(algo, "block_threads", None)
            ),
        )
        algo.__dict__["_link_input_suffixes"] = [".ltoir"] * len(extras)
        algo.__dict__["lto_irs"] = extras

    return ltoir_blob


def make_invocable_from_specialization(
    specialization: Algorithm, *, threads=None, block_threads=None
):
    if specialization.struct_name.startswith("Warp"):
        threads = _validate_warp_threads(
            threads if threads is not None else specialization.threads
        )
        if block_threads is not None:
            block_threads = _normalize_block_threads(block_threads)
    if threads is not None:
        specialization.threads = threads
    if block_threads is not None:
        specialization.block_threads = block_threads

    specialization._qualify_private_symbols(
        threads=threads,
        block_threads=block_threads,
    )

    collector = _COOP_SPECIALIZATION_COLLECTOR.get()
    if collector is not None:
        collector.append((specialization, threads, block_threads))
        return specialization

    from ._common import make_binary_tempfile

    lto_irs = specialization.get_lto_ir(threads=threads, block_threads=block_threads)
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
        algorithm.codegen(self)

    @property
    def temp_storage_bytes(self):
        return self._temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        return self._temp_storage_alignment

    @property
    def files(self):
        return [v.name for v in self._temp_files]

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
