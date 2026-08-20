# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""NVRTC LTO-IR provider for CUB BlockLoad and BlockStore."""

from __future__ import annotations

import hashlib
import math
import tempfile
import threading
import weakref
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from ._runtime import validate_cutlass_runtime
from ._thread_data import ThreadData

_LINK_LIBRARIES_ATTR = "link-libraries"
_DISPATCHER_ATTR = "_cuda_coop_cutlass_trace_finalize_dispatcher"
_TARGET_ATTR = "_cuda_coop_cutlass_trace_finalize_target"
_ARTIFACTS = tempfile.TemporaryDirectory(prefix="cuda-coop-cutlass-")
_STATE_LOCK = threading.RLock()


@dataclass(frozen=True)
class _TypeSpec:
    dsl_type: type
    cpp_type: str
    token: str


def _type_specs() -> dict[type, _TypeSpec]:
    from cutlass.base_dsl.typing import (
        Float32,
        Float64,
        Int32,
        Int64,
        Uint8,
        Uint32,
        Uint64,
    )

    return {
        Uint8: _TypeSpec(Uint8, "unsigned char", "u8"),
        Int32: _TypeSpec(Int32, "int", "i32"),
        Uint32: _TypeSpec(Uint32, "unsigned int", "u32"),
        Int64: _TypeSpec(Int64, "long long", "i64"),
        Uint64: _TypeSpec(Uint64, "unsigned long long", "u64"),
        Float32: _TypeSpec(Float32, "float", "f32"),
        Float64: _TypeSpec(Float64, "double", "f64"),
    }


_INTEGER_TOKENS = frozenset({"u8", "i32", "u32", "i64", "u64"})
_FLOAT_TOKENS = frozenset({"f32", "f64"})
_NOT_PLAIN_SCALAR = object()


def _coerce_plain_scalar(
    value: Any,
    value_type: type,
    *,
    name: str,
    allow_nonfinite: bool,
    convert: bool = True,
) -> Any:
    token = _type_specs()[value_type].token
    if type(value) is int:
        if token not in _INTEGER_TOKENS:
            raise TypeError(
                f"cuda.coop.cutlass {name} dtype does not match {value_type.__name__}"
            )
        bits = int(token.lstrip("iu"))
        lower = 0 if token.startswith("u") else -(1 << (bits - 1))
        upper = (1 << bits) - 1 if token.startswith("u") else (1 << (bits - 1)) - 1
        if not lower <= value <= upper:
            raise ValueError(
                f"cuda.coop.cutlass {name} is not representable in "
                f"{value_type.__name__}"
            )
        return value_type(value) if convert else value
    if type(value) is float:
        if token not in _FLOAT_TOKENS:
            raise TypeError(
                f"cuda.coop.cutlass {name} dtype does not match {value_type.__name__}"
            )
        if not allow_nonfinite and not math.isfinite(value):
            raise ValueError(f"cuda.coop.cutlass {name} must be finite")
        limit = float(np.finfo(np.float32 if token == "f32" else np.float64).max)
        if math.isfinite(value) and abs(value) > limit:
            raise ValueError(
                f"cuda.coop.cutlass {name} is not representable in "
                f"{value_type.__name__}"
            )
        return value_type(value) if convert else value
    return _NOT_PLAIN_SCALAR


def _canonical_type(value: Any, *, feature: str) -> type:
    specs = _type_specs()
    if isinstance(value, type) and value in specs:
        return value
    value_type = type(value)
    if value_type in specs:
        return value_type
    ordinary_types = {
        int: "Int32",
        float: "Float32",
        np.uint8: "Uint8",
        np.int32: "Int32",
        np.uint32: "Uint32",
        np.int64: "Int64",
        np.uint64: "Uint64",
        np.float32: "Float32",
        np.float64: "Float64",
    }
    target_name = ordinary_types.get(value if isinstance(value, type) else value_type)
    if target_name is not None:
        return next(
            candidate for candidate in specs if candidate.__name__ == target_name
        )
    name = str(getattr(value, "name", getattr(value, "__name__", value))).lower()
    name = name.replace("numpy.", "").replace("<class '", "").replace("'>", "")
    aliases = {
        "uint8": "Uint8",
        "int32": "Int32",
        "uint32": "Uint32",
        "int64": "Int64",
        "uint64": "Uint64",
        "float32": "Float32",
        "float64": "Float64",
        "int": "Int32",
        "float": "Float32",
    }
    target_name = aliases.get(name)
    for candidate in specs:
        if candidate.__name__ == target_name:
            return candidate
    dsl_dtype = getattr(value, "dtype", None)
    if isinstance(dsl_dtype, type) and dsl_dtype in specs:
        return dsl_dtype
    mlir_type = getattr(value, "type", None)
    type_name = None if mlir_type is None else str(mlir_type)
    if bool(getattr(value, "is_float", False)):
        target_name = {"f32": "Float32", "f64": "Float64"}.get(type_name)
    elif type_name in {"i32", "i64"}:
        signed = getattr(value, "signed", None)
        if signed is None:
            raise TypeError("cannot infer the signedness of a CUTLASS integer value")
        target_name = {
            ("i32", True): "Int32",
            ("i32", False): "Uint32",
            ("i64", True): "Int64",
            ("i64", False): "Uint64",
        }.get((type_name, signed))
    else:
        target_name = None
    for candidate in specs:
        if candidate.__name__ == target_name:
            return candidate
    supported = ", ".join(spec.dsl_type.__name__ for spec in specs.values())
    raise NotImplementedError(
        f"cuda.coop.cutlass {feature} supports {supported}; got {value!r}"
    )


def _binding_parts(binding: Any) -> tuple[str, Any]:
    kind = getattr(getattr(binding, "kind", None), "value", None)
    if kind not in {"omitted", "static", "runtime"}:
        raise TypeError("load/store plan contains an invalid argument binding")
    return kind, getattr(binding, "value", None) if kind == "static" else None


def _static_scalar_key(kind: str, value: Any) -> tuple[str, str, str] | None:
    if kind != "static":
        return None
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__, repr(value)


@dataclass(frozen=True, eq=False)
class _LoadStoreRequest:
    kind: str
    value_type: type
    block_dim: tuple[int, int, int]
    items_per_thread: int
    valid_items_kind: str
    valid_items_value: Any
    oob_default_kind: str
    oob_default_value: Any
    offset_kind: str
    offset_value: Any

    @classmethod
    def from_plan(cls, plan: Any, *, value_type: type) -> "_LoadStoreRequest":
        require_supported = getattr(plan, "require_supported", None)
        if callable(require_supported):
            plan = require_supported()
        target = getattr(getattr(plan, "target", None), "value", None)
        if target != "cub_block":
            raise ValueError("CUTLASS load/store requires a CUB block lowering plan")
        operation = getattr(getattr(plan, "call", None), "operation", None)
        if operation is None:
            operation = getattr(plan, "operation", None)
        if operation is None:
            raise TypeError("CUTLASS load/store plan has no operation semantics")

        kind = getattr(getattr(operation, "kind", None), "value", None)
        if kind not in {"load", "store"}:
            raise ValueError("CUTLASS load/store plan has an invalid operation kind")
        algorithm = getattr(getattr(operation, "algorithm", None), "value", None)
        if algorithm != "direct":
            raise NotImplementedError(
                "cuda.coop.cutlass currently supports the DIRECT algorithm only"
            )
        items_per_thread = getattr(operation, "items_per_thread", None)
        if (
            not isinstance(items_per_thread, int)
            or isinstance(items_per_thread, bool)
            or items_per_thread < 1
        ):
            raise ValueError("load/store plan requires a positive item count")

        participation = getattr(plan, "participation", None)
        block_dim = getattr(participation, "exact_block_dim", None)
        if block_dim is None:
            block_dim = getattr(plan, "block_dim", None)
        block_dim = tuple(block_dim or ())
        if len(block_dim) != 3 or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1
            for dim in block_dim
        ):
            raise ValueError("load/store plan requires exact positive block dimensions")

        plan_type = _canonical_type(getattr(operation, "dtype", None), feature=kind)
        if plan_type is not value_type:
            raise TypeError("load/store plan dtype does not match the memory dtype")
        valid_kind, valid_value = _binding_parts(operation.valid_items)
        oob_kind, oob_value = _binding_parts(operation.oob_default)
        offset_kind, offset_value = _binding_parts(operation.offset)
        if kind == "store" and oob_kind != "omitted":
            raise ValueError("BlockStore does not accept an out-of-bounds default")
        if oob_kind != "omitted" and valid_kind == "omitted":
            raise ValueError("oob_default requires valid_items")
        if oob_kind == "static":
            plain_value = _coerce_plain_scalar(
                oob_value,
                value_type,
                name="load oob_default",
                allow_nonfinite=False,
                convert=False,
            )
            if plain_value is _NOT_PLAIN_SCALAR:
                try:
                    oob_type = _canonical_type(oob_value, feature="load")
                except (TypeError, NotImplementedError) as error:
                    raise TypeError(
                        "cuda.coop.cutlass load oob_default dtype cannot be resolved"
                    ) from error
                if oob_type is not value_type:
                    raise TypeError(
                        "cuda.coop.cutlass load oob_default dtype does not match "
                        "the tensor dtype"
                    )
        if valid_kind == "static":
            tile_items = math.prod(block_dim) * items_per_thread
            if (
                isinstance(valid_value, bool)
                or not isinstance(valid_value, Integral)
                or not 0 <= int(valid_value) <= tile_items
            ):
                raise ValueError(
                    f"valid_items must be between zero and the tile size ({tile_items})"
                )

        provenance = getattr(plan, "provenance", None)
        expected_provenance = (
            "CUB",
            f"cub/block/block_{kind}.cuh",
            f"cub::Block{kind.title()}",
            kind.title(),
        )
        if getattr(provenance, "semantic_key", None) != expected_provenance:
            raise ValueError("load/store plan has incompatible CUB provenance")

        return cls(
            kind=kind,
            value_type=value_type,
            block_dim=block_dim,
            items_per_thread=items_per_thread,
            valid_items_kind=valid_kind,
            valid_items_value=valid_value,
            oob_default_kind=oob_kind,
            oob_default_value=oob_value,
            offset_kind=offset_kind,
            offset_value=offset_value,
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.value_type.__name__,
            self.block_dim,
            self.items_per_thread,
            self.valid_items_kind,
            _static_scalar_key(self.valid_items_kind, self.valid_items_value),
            self.oob_default_kind,
            _static_scalar_key(self.oob_default_kind, self.oob_default_value),
            self.offset_kind,
            _static_scalar_key(self.offset_kind, self.offset_value),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LoadStoreRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def symbol_name(self) -> str:
        spec = _type_specs()[self.value_type]
        digest = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()[:12]
        x, y, z = self.block_dim
        return (
            f"cuda_coop_cutlass_cub_{self.kind}_block_b{x}x{y}x{z}_"
            f"direct_{spec.token}_x{self.items_per_thread}_{digest}"
        )


def _cpp_scalar(value: Any, cpp_type: str) -> str:
    if isinstance(value, bool):
        literal = "true" if value else "false"
    elif isinstance(value, Integral):
        literal = str(int(value))
    elif isinstance(value, Real) and math.isfinite(float(value)):
        literal = repr(float(value))
    else:
        raise TypeError("static scalar options must be finite numeric values")
    return f"static_cast<{cpp_type}>({literal})"


def _binding_expression(
    kind: str,
    value: Any,
    *,
    runtime_name: str,
    cpp_type: str | None = None,
) -> str | None:
    if kind == "omitted":
        return None
    if kind == "runtime":
        return runtime_name
    if cpp_type is not None:
        return _cpp_scalar(value, cpp_type)
    return str(int(value))


def _render_request(request: _LoadStoreRequest) -> str:
    spec = _type_specs()[request.value_type]
    x, y, z = request.block_dim
    title = request.kind.title()
    algorithm = f"::cub::BLOCK_{request.kind.upper()}_DIRECT"
    is_load = request.kind == "load"
    params = [f"{'const ' if is_load else ''}{spec.cpp_type}* base"]
    if not is_load:
        params.extend(
            f"{spec.cpp_type} item{index}" for index in range(request.items_per_thread)
        )
    if request.valid_items_kind == "runtime":
        params.append("int valid_items")
    if request.oob_default_kind == "runtime":
        params.append(f"{spec.cpp_type} oob_default")
    if request.offset_kind == "runtime":
        params.append("long long offset")
    if is_load:
        params.append(f"{spec.cpp_type}* result_items")

    offset = _binding_expression(
        request.offset_kind,
        request.offset_value,
        runtime_name="offset",
    )
    pointer_type = f"{'const ' if is_load else ''}{spec.cpp_type}*"
    lines = [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using block_type = ::cub::Block{title}<{spec.cpp_type}, {x}, "
        f"{request.items_per_thread}, {algorithm}, {y}, {z}>;",
        "  __shared__ typename block_type::TempStorage temp_storage;",
        f"  {pointer_type} tile = base;",
    ]
    if offset is not None:
        lines.append(f"  tile += {offset};")
    if is_load:
        lines.append(f"  {spec.cpp_type} items[{request.items_per_thread}];")
    else:
        item_values = ", ".join(
            f"item{index}" for index in range(request.items_per_thread)
        )
        lines.append(
            f"  {spec.cpp_type} items[{request.items_per_thread}] = {{{item_values}}};"
        )
    call_arguments = ["tile", "items"]
    valid_items = _binding_expression(
        request.valid_items_kind,
        request.valid_items_value,
        runtime_name="valid_items",
    )
    if valid_items is not None:
        call_arguments.append(valid_items)
    oob_default = _binding_expression(
        request.oob_default_kind,
        request.oob_default_value,
        runtime_name="oob_default",
        cpp_type=spec.cpp_type,
    )
    if oob_default is not None:
        call_arguments.append(oob_default)
    lines.extend(
        (
            f"  block_type(temp_storage).{title}({', '.join(call_arguments)});",
            "  __syncthreads();",
        )
    )
    if is_load:
        lines.extend(
            f"  result_items[{index}] = items[{index}];"
            for index in range(request.items_per_thread)
        )
    lines.append("}")
    return "\n".join(lines)


def _render_bundle_source(requests: set[_LoadStoreRequest]) -> str:
    ordered = sorted(requests, key=lambda request: repr(request.semantic_key))
    bodies = "\n\n".join(_render_request(request) for request in ordered)
    return (
        "#include <cub/block/block_load.cuh>\n"
        "#include <cub/block/block_store.cuh>\n\n"
        'extern "C" {\n'
        f"{bodies}\n"
        "}\n"
    )


def _shape_and_stride(value: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    shape = getattr(value, "shape", None)
    stride = getattr(value, "strides", None)
    if stride is None:
        stride = getattr(value, "stride", None)
    if shape is None or stride is None:
        raise NotImplementedError(
            "cuda.coop.cutlass load/store requires explicit shape and stride metadata"
        )
    shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    stride = tuple(stride) if isinstance(stride, (tuple, list)) else (stride,)
    return shape, stride


def _layout_leaf_pairs(
    shape: Any,
    stride: Any,
) -> tuple[tuple[Any, Any], ...] | None:
    shape_is_tree = isinstance(shape, (tuple, list))
    stride_is_tree = isinstance(stride, (tuple, list))
    if shape_is_tree != stride_is_tree:
        return None
    if not shape_is_tree:
        return ((shape, stride),)
    if not shape or len(shape) != len(stride):
        return None

    leaves: list[tuple[Any, Any]] = []
    for shape_child, stride_child in zip(shape, stride):
        child_leaves = _layout_leaf_pairs(shape_child, stride_child)
        if child_leaves is None:
            return None
        leaves.extend(child_leaves)
    return tuple(leaves)


def _static_layout_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = value.__index__()
    except (AttributeError, TypeError, ValueError):
        return None
    if isinstance(normalized, bool):
        return None
    return int(normalized)


def _compact_layout_reason(value: Any) -> str | None:
    shape, stride = _shape_and_stride(value)
    leaves = _layout_leaf_pairs(shape, stride)
    if leaves is None:
        return "shape and stride layouts are not congruent"
    extents = tuple(_static_layout_int(extent) for extent, _ in leaves)
    strides = tuple(_static_layout_int(item_stride) for _, item_stride in leaves)
    if any(item is None for item in (*extents, *strides)):
        return "layout is not statically known"
    dimensions = tuple(zip(strides, extents))
    if any(stride_value <= 0 or extent <= 0 for stride_value, extent in dimensions):
        return "layout contains a non-positive extent or stride"

    expected_stride = 1
    for stride_value, extent in sorted(dimensions):
        if extent == 1:
            continue
        if stride_value != expected_stride:
            return "layout is not compact"
        expected_stride *= extent
    return None


def _require_contiguous_pointer(
    value: Any,
    *,
    feature: str,
    required_elements: int | None = None,
) -> Any:
    layout_reason = _compact_layout_reason(value)
    if layout_reason is not None:
        raise NotImplementedError(
            f"cuda.coop.cutlass {feature} requires a statically compact "
            f"contiguous tensor; {layout_reason}"
        )
    if required_elements is not None:
        shape, stride = _shape_and_stride(value)
        leaves = _layout_leaf_pairs(shape, stride)
        if leaves is None:
            raise RuntimeError("validated tensor layout was not congruent")
        extents = tuple(_static_layout_int(extent) for extent, _ in leaves)
        if any(extent is None for extent in extents):
            raise RuntimeError("validated tensor extent was not static")
        available_elements = math.prod(
            extent for extent in extents if extent is not None
        )
        if required_elements > available_elements:
            raise ValueError(
                f"cuda.coop.cutlass {feature} requires {required_elements} "
                "elements after applying its static offset and valid_items, "
                f"but the tensor provides {available_elements}"
            )
    candidates = [value]
    for name in ("iterator", "pointer", "ptr", "_pointer", "_ptr"):
        try:
            candidate = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if candidate is not None:
            candidates.append(candidate)
    pointer = None
    for candidate in candidates:
        to_llvm_ptr = getattr(candidate, "to_llvm_ptr", None)
        pointer = (
            to_llvm_ptr()
            if callable(to_llvm_ptr)
            else getattr(candidate, "llvm_ptr", None)
        )
        if pointer is not None:
            break
    if pointer is None:
        raise TypeError(
            f"cuda.coop.cutlass {feature} tensor must expose a raw LLVM pointer"
        )

    from cutlass._mlir.dialects import llvm

    pointer_type = llvm.PointerType(pointer.type)
    if pointer_type.address_space != 0:
        pointer = llvm.addrspacecast(llvm.PointerType.get(0), pointer)
    return pointer


def _memory_type(value: Any, *, feature: str) -> type:
    for name in ("element_type", "dtype", "_dtype"):
        dtype = getattr(value, name, None)
        if dtype is not None:
            return _canonical_type(dtype, feature=feature)
    iterator = getattr(value, "iterator", None)
    dtype = getattr(iterator, "dtype", None)
    if dtype is not None:
        return _canonical_type(dtype, feature=feature)
    raise TypeError(
        f"cuda.coop.cutlass {feature} tensor must expose element_type or dtype"
    )


def _runtime_arguments(
    request: _LoadStoreRequest,
    *,
    valid_items: Any,
    oob_default: Any,
    offset: Any,
) -> tuple[list[type], list[Any]]:
    from cutlass.base_dsl.typing import Int32, Int64

    parameter_types: list[type] = []
    arguments: list[Any] = []
    if request.valid_items_kind == "runtime":
        parameter_types.append(Int32)
        try:
            arguments.append(
                valid_items if isinstance(valid_items, Int32) else Int32(valid_items)
            )
        except Exception as error:
            raise TypeError(
                "valid_items must be convertible to CUTLASS Int32"
            ) from error
    if request.oob_default_kind == "runtime":
        try:
            runtime_type = _canonical_type(oob_default, feature="load")
        except Exception as error:
            raise TypeError("oob_default must match the tensor dtype") from error
        if runtime_type is not request.value_type:
            raise TypeError("oob_default must match the tensor dtype")
        parameter_types.append(request.value_type)
        arguments.append(oob_default)
    if request.offset_kind == "runtime":
        parameter_types.append(Int64)
        try:
            arguments.append(offset if isinstance(offset, Int64) else Int64(offset))
        except Exception as error:
            raise TypeError("offset must be convertible to CUTLASS Int64") from error
    return parameter_types, arguments


def _required_static_elements(plan: Any) -> int | None:
    operation = plan.call.operation
    participation = plan.participation
    if participation is None:
        raise TypeError("load/store plan has no participation contract")
    valid_kind, valid_value = _binding_parts(operation.valid_items)
    offset_kind, offset_value = _binding_parts(operation.offset)
    if valid_kind == "runtime" or offset_kind == "runtime":
        return None
    valid_items = (
        math.prod(participation.exact_block_dim) * operation.items_per_thread
        if valid_kind == "omitted"
        else int(valid_value)
    )
    offset = 0 if offset_kind == "omitted" else int(offset_value)
    return offset + valid_items


def materialize_load(
    *,
    plan: Any,
    source: Any,
    output: ThreadData,
    valid_items: Any,
    oob_default: Any,
    offset: Any,
) -> ThreadData:
    """Trace one external CUB BlockLoad call and populate ``output``."""

    from cutlass import cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.ffi import ffi

    value_type = _memory_type(source, feature="load")
    if (
        output.dtype is not None
        and _canonical_type(output.dtype, feature="load") is not value_type
    ):
        raise TypeError("cuda.coop.cutlass load source dtype does not match output")
    request = _LoadStoreRequest.from_plan(plan, value_type=value_type)
    pointer = _require_contiguous_pointer(
        source,
        feature="load",
        required_elements=_required_static_elements(plan),
    )
    parameter_types, arguments = _runtime_arguments(
        request,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
    )
    result = cute.make_rmem_tensor(output.items_per_thread, value_type)
    _register_request(request)
    ffi(
        name=request.symbol_name,
        params_types=[
            llvm.PointerType.get(0),
            *parameter_types,
            llvm.PointerType.get(0),
        ],
        return_type=None,
    )(pointer, *arguments, result.iterator.llvm_ptr)
    output.dtype = value_type
    for index in range(output.items_per_thread):
        output[index] = result[index]
    return output


def materialize_store(
    *,
    plan: Any,
    destination: Any,
    value: ThreadData,
    valid_items: Any,
    offset: Any,
) -> None:
    """Trace one external CUB BlockStore call."""

    from cutlass._mlir.dialects import llvm
    from cutlass.cute.ffi import ffi

    if not isinstance(value, ThreadData):
        raise TypeError("cuda.coop.cutlass store value must be ThreadData")
    if value.dtype is None:
        raise TypeError("cuda.coop.cutlass store value must have a dtype")
    value_type = _canonical_type(value.dtype, feature="store")
    destination_type = _memory_type(destination, feature="store")
    if destination_type is not value_type:
        raise TypeError(
            "cuda.coop.cutlass store destination dtype does not match value"
        )
    request = _LoadStoreRequest.from_plan(plan, value_type=value_type)
    pointer = _require_contiguous_pointer(
        destination,
        feature="store",
        required_elements=_required_static_elements(plan),
    )
    parameter_types, arguments = _runtime_arguments(
        request,
        valid_items=valid_items,
        oob_default=None,
        offset=offset,
    )
    values = _thread_data_values(value, value_type)
    _register_request(request)
    ffi(
        name=request.symbol_name,
        params_types=[
            llvm.PointerType.get(0),
            *([value_type] * value.items_per_thread),
            *parameter_types,
        ],
        return_type=None,
    )(pointer, *values, *arguments)


def _signless_integer_item_matches_dtype(item: Any, value_type: type) -> bool:
    spec = _type_specs()[value_type]
    if spec.token not in {"i32", "u32", "i64", "u64"}:
        return False
    if getattr(item, "signed", None) is not None:
        return False
    mlir_type = getattr(item, "type", None)
    return mlir_type is not None and str(mlir_type) == f"i{spec.token[1:]}"


def _thread_data_values(value: ThreadData, value_type: type) -> tuple[Any, ...]:
    values = tuple(value)
    converted: list[Any] = []
    for index, item in enumerate(values):
        plain_item = _coerce_plain_scalar(
            item,
            value_type,
            name=f"store ThreadData item {index}",
            allow_nonfinite=True,
        )
        if plain_item is not _NOT_PLAIN_SCALAR:
            converted.append(plain_item)
            continue
        try:
            item_type = _canonical_type(item, feature="store")
        except (TypeError, NotImplementedError) as error:
            if _signless_integer_item_matches_dtype(item, value_type):
                converted.append(item)
                continue
            raise TypeError(
                "cuda.coop.cutlass.store ThreadData item "
                f"{index} type cannot be reconciled with declared dtype"
            ) from error
        if item_type is not value_type:
            raise TypeError(
                "cuda.coop.cutlass.store ThreadData dtype does not match "
                "initialized item types"
            )
        converted.append(item)
    return tuple(converted)


@dataclass
class _TraceSession:
    module_op: Any
    requests: set[_LoadStoreRequest]


_SESSIONS: weakref.WeakKeyDictionary[Any, _TraceSession] = weakref.WeakKeyDictionary()


def _dsl() -> Any:
    return validate_cutlass_runtime().dsl_type._get_dsl()


def _same_operation(left: Any, right: Any) -> bool:
    left = getattr(left, "operation", left)
    right = getattr(right, "operation", right)
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return isinstance(result, bool) and result


def _active_module_op() -> Any | None:
    try:
        from cutlass._mlir import ir

        insertion_point = ir.InsertionPoint.current
        operation = (
            None
            if insertion_point is None or insertion_point.block is None
            else insertion_point.block.owner
        )
    except Exception:
        return None
    while operation is not None:
        raw = getattr(operation, "operation", operation)
        if str(getattr(raw, "name", "")) == "builtin.module":
            return raw
        operation = getattr(raw, "parent", None)
    return None


def _trace_finalize_dispatcher(dsl: Any, module: Any, function_name: str) -> None:
    target = getattr(dsl, _TARGET_ATTR, None)
    if callable(target):
        target(dsl, module, function_name)


def _ensure_trace_hook() -> Any:
    dsl = _dsl()
    if getattr(dsl, _DISPATCHER_ATTR, None) is None:
        dsl.register_trace_finalize_hook(_trace_finalize_dispatcher)
        setattr(dsl, _DISPATCHER_ATTR, _trace_finalize_dispatcher)
    setattr(dsl, _TARGET_ATTR, _trace_finalize)
    return dsl


def _register_request(request: _LoadStoreRequest) -> None:
    dsl = _ensure_trace_hook()
    module_op = _active_module_op()
    if module_op is None:
        raise RuntimeError(
            "cuda.coop.cutlass load/store must be called while tracing a kernel"
        )
    with _STATE_LOCK:
        session = _SESSIONS.get(dsl.compile_options)
        if session is None or not _same_operation(session.module_op, module_op):
            session = _TraceSession(module_op, set())
            _SESSIONS[dsl.compile_options] = session
        session.requests.add(request)


def _include_dirs() -> list[Path]:
    from cuda.coop._headers import resolve_include_paths

    paths = resolve_include_paths(
        start=Path(__file__),
        required_headers=(
            "cub/block/block_load.cuh",
            "cub/block/block_store.cuh",
        ),
    )
    return list(paths.as_tuple())


def _configured_arch() -> str:
    from cutlass.base_dsl.compiler import GPUArch

    dsl = _dsl()
    options = getattr(getattr(dsl, "compile_options", None), "options", None)
    option = options.get(GPUArch) if hasattr(options, "get") else None
    arch = str(getattr(option, "value", "")).strip()
    if not arch:
        arch = str(getattr(getattr(dsl, "envar", None), "arch", "")).strip()
    if not arch:
        from cutlass.base_dsl.runtime import cuda as cuda_runtime

        major, minor = cuda_runtime.get_compute_capability_major_minor()
        if major is None or minor is None:
            raise RuntimeError("unable to resolve a CUDA architecture for NVRTC")
        arch = f"{major}{minor}"
    for prefix in ("compute_", "compute", "sm_", "sm"):
        if arch.startswith(prefix):
            arch = arch[len(prefix) :]
            break
    if not arch or not arch.rstrip("af").isdigit():
        raise RuntimeError(f"unsupported CUDA architecture {arch!r}")
    return f"compute_{arch}"


def _program_log(nvrtc: Any, program: Any) -> str:
    error, size = nvrtc.nvrtcGetProgramLogSize(program)
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS or size <= 0:
        return ""
    log = bytearray(size)
    if nvrtc.nvrtcGetProgramLog(program, log)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return ""
    return bytes(log).decode("utf-8", errors="replace").strip("\x00\n")


def _compile_ltoir(source: str) -> Path:
    import cuda.bindings.nvrtc as nvrtc

    options = [
        b"--std=c++17",
        b"--relocatable-device-code=true",
        b"-default-device",
        b"-nvvm-version=nvvm-latest",
        f"--gpu-architecture={_configured_arch()}".encode(),
        b"-dlto",
        *(f"-I{path}".encode() for path in _include_dirs()),
    ]
    source_bytes = source.encode()
    error, program = nvrtc.nvrtcCreateProgram(
        source_bytes,
        b"cuda_coop_cutlass_load_store.cu",
        0,
        [],
        [],
    )
    if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError("failed to create the CUTLASS load/store NVRTC program")
    try:
        error = nvrtc.nvrtcCompileProgram(program, len(options), options)[0]
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(
                "failed to compile the CUTLASS load/store provider to LTO-IR: "
                + _program_log(nvrtc, program)
            )
        error, size = nvrtc.nvrtcGetLTOIRSize(program)
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("failed to query CUTLASS load/store LTO-IR size")
        blob = bytearray(size)
        if nvrtc.nvrtcGetLTOIR(program, blob)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("failed to retrieve CUTLASS load/store LTO-IR")
    finally:
        nvrtc.nvrtcDestroyProgram(program)

    directory = Path(_ARTIFACTS.name)
    directory.mkdir(parents=True, exist_ok=True)
    digest_input = b"\0".join((source_bytes, *options))
    digest = hashlib.sha256(digest_input).hexdigest()
    artifact = directory / f"cuda_coop_cutlass_{digest}.ltoir"
    artifact.write_bytes(blob)
    return artifact


def _append_link_library(module: Any, path: Path) -> None:
    from cutlass._mlir import ir

    found = False
    for operation in module.body.operations:
        if str(getattr(operation, "name", "")) != "gpu.module":
            continue
        found = True
        existing_attributes = (
            operation.attributes[_LINK_LIBRARIES_ATTR]
            if _LINK_LIBRARIES_ATTR in operation.attributes
            else ()
        )
        existing = {
            attribute.value
            for attribute in existing_attributes
            if getattr(attribute, "value", "")
        }
        existing.add(str(path))
        operation.attributes[_LINK_LIBRARIES_ATTR] = ir.ArrayAttr.get(
            [ir.StringAttr.get(item) for item in sorted(existing)]
        )
    if not found:
        raise RuntimeError("CUTLASS trace contains no gpu.module for provider linking")


def _trace_finalize(dsl: Any, module: Any, function_name: str) -> None:
    del function_name
    with _STATE_LOCK:
        session = _SESSIONS.pop(dsl.compile_options, None)
    if session is None or not session.requests:
        return
    if not _same_operation(session.module_op, module.operation):
        raise RuntimeError("CUTLASS provider requests escaped their originating trace")
    source = _render_bundle_source(session.requests)
    artifact = _compile_ltoir(source)
    _append_link_library(module, artifact)


__all__ = ["materialize_load", "materialize_store"]
