# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed CUB Load/Store requests lowered from the shared group planner."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from numbers import Integral, Real
from typing import Any

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)

from .._compiler import _rendering, _state, _types
from .._compiler._types import ALL_PROVIDER_TYPES, TYPE_SPECS
from .._thread_data import _UNSET, ThreadData, _make_rmem_tensor
from .._thread_group import ThreadGroup
from ._load_store_layout import contiguous_layout_reason, static_layout_elements

_ROOT_SCOPE = "cuda.coop.cutlass"
_MAX_STATIC_OFFSET = (1 << 63) - 1
_LLVM_LOCAL_ADDRESS_SPACE = 5
_provider_types = _types


@dataclasses.dataclass(frozen=True, eq=False)
class _CubLoadStoreRequest:
    plan: GroupLoweringPlan
    value_type: type
    kind: str = "cub_group_load_store"

    def __post_init__(self):
        self.plan.require_supported()
        if self.plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise NotImplementedError("CUTLASS Load/Store currently requires a block")
        if not isinstance(self.plan.implementation, AlgorithmSpec):
            raise TypeError("Load/Store requires a shared AlgorithmSpec")
        if self.operation.dtype is not self.value_type:
            raise TypeError("Load/Store provider dtype does not match its plan")
        if self.plan.result is not None:
            raise ValueError("Load/Store operates in place and has no result contract")
        if not self.uses_scratch and (
            self.plan.synchronization.storage_reuse_barrier
            is not SynchronizationScope.NONE
        ):
            raise ValueError(
                "storage-free Load/Store must not introduce a reuse barrier"
            )
        if (
            self.uses_scratch
            and self.plan.synchronization.storage_reuse_barrier
            not in {
                SynchronizationScope.BLOCK,
                SynchronizationScope.NONE,
            }
        ):
            raise ValueError(
                "block Load/Store requires block-scoped reuse synchronization"
            )
        if self.operation.oob_default.kind is BindingKind.STATIC:
            _validate_static_oob_default(
                self.operation.oob_default.value, self.value_type
            )

    @property
    def operation(self):
        operation = self.plan.call.operation
        if not isinstance(operation, GroupLoadStoreSemantics):
            raise TypeError("Load/Store requires shared group semantics")
        return operation

    @property
    def operation_kind(self):
        return self.operation.kind

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def block_dim(self):
        return self.plan.participation.exact_block_dim

    @property
    def uses_scratch(self):
        return self.plan.temp_storage.ownership is not StorageOwnership.NONE

    @property
    def cpp_type(self):
        arguments = ", ".join(
            _render_template_argument(self, name, value)
            for name, value in self.implementation.ordered_template_arguments
        )
        return f"::cub::{self.implementation.struct_name}<{arguments}>"

    @property
    def scratch_requirement_key(self):
        return ("cub_load_store_layout", self.implementation.semantic_key)

    @property
    def semantic_key(self):
        return self.plan.artifact_key

    def __eq__(self, other):
        if not isinstance(other, _CubLoadStoreRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self):
        return hash(self.semantic_key)

    @property
    def symbol_name(self):
        signature = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()[:16]
        return (
            f"cuda_coop_cutlass_{self.operation_kind.value}_"
            f"{TYPE_SPECS[self.value_type].token}_{signature}"
        )


def _render_cub_load_store(request):
    request.__post_init__()
    operation = request.operation
    spec = TYPE_SPECS[request.value_type]
    is_load = operation.kind is GroupLoadStoreKind.LOAD
    params = [f"{'const ' if is_load else ''}{spec.cpp_type}* base"]
    if not is_load:
        params.extend(
            f"{spec.cpp_type} item{i}" for i in range(operation.items_per_thread)
        )
    if operation.valid_items.kind is BindingKind.RUNTIME:
        params.append("int valid_items")
    if operation.oob_default.kind is BindingKind.RUNTIME:
        params.append(f"{spec.cpp_type} oob_default")
    if operation.offset.kind is BindingKind.RUNTIME:
        params.append("long long offset")
    if request.uses_scratch:
        params.extend(
            (
                "unsigned int temp_storage_smem_addr",
                "int temp_storage_bytes",
                "int temp_storage_auto_sync",
            )
        )
    if is_load:
        params.append(f"{spec.cpp_type}* result_items")
    lines = [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = {request.cpp_type};",
    ]
    storage = ""
    if request.uses_scratch:
        lines.extend(
            [
                "  using storage_type = typename implementation_type::TempStorage;",
                "  if (temp_storage_bytes < sizeof(storage_type) ||",
                "      (temp_storage_smem_addr & (alignof(storage_type) - 1)) != 0) {",
                '    asm volatile("trap;");',
                "  }",
                "  unsigned long long generic_addr;",
                '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_addr) : "l"(static_cast<unsigned long long>(temp_storage_smem_addr)));',
                "  auto& storage = *reinterpret_cast<storage_type*>(generic_addr);",
            ]
        )
        storage = "storage"
    if operation.valid_items.kind is BindingKind.RUNTIME:
        count = request.plan.resolved_group.static_size * operation.items_per_thread
        lines.append(
            f'  if (valid_items < 0 || valid_items > {count}) {{ asm volatile("trap;"); }}'
        )
    if operation.offset.kind is BindingKind.RUNTIME:
        lines.append('  if (offset < 0) { asm volatile("trap;"); }')
    offset = _binding_expr(request, operation.offset, runtime_name="offset")
    lines.append(f"  auto* tile_ptr = base{'' if offset is None else ' + ' + offset};")
    if is_load:
        preserve = (
            operation.valid_items.kind is not BindingKind.OMITTED
            and operation.oob_default.kind is BindingKind.OMITTED
        )
        initial = (
            ", ".join(f"result_items[{i}]" for i in range(operation.items_per_thread))
            if preserve
            else ""
        )
    else:
        initial = ", ".join(f"item{i}" for i in range(operation.items_per_thread))
    lines.append(
        f"  {spec.cpp_type} items[{operation.items_per_thread}] = {{{initial}}};"
    )
    args = ["tile_ptr", "items"]
    for binding, name in (
        (operation.valid_items, "valid_items"),
        (operation.oob_default, "oob_default"),
    ):
        expression = _binding_expr(
            request, binding, runtime_name=name, oob_default=name == "oob_default"
        )
        if expression is not None:
            args.append(expression)
    lines.append(
        f"  implementation_type({storage}).{request.implementation.method_name}({', '.join(args)});"
    )
    if is_load:
        lines.extend(
            f"  result_items[{i}] = items[{i}];"
            for i in range(operation.items_per_thread)
        )
    if request.uses_scratch:
        lines.append("  if (temp_storage_auto_sync != 0) { __syncthreads(); }")
    return [*lines, "}"]


def _make_request(
    *,
    group,
    launch,
    kind,
    value_type,
    items_per_thread,
    algorithm,
    valid_items_binding,
    oob_default_binding,
    offset_binding,
    temp_storage=None,
):
    plan = _make_group_load_store_plan(
        group=group,
        launch=launch,
        kind=kind,
        dtype=value_type,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        offset=offset_binding,
        temp_storage=temp_storage,
    ).require_supported()
    return _CubLoadStoreRequest(plan, value_type)


def provider_load(
    *,
    group,
    launch,
    source,
    output,
    algorithm,
    valid_items,
    valid_items_binding,
    oob_default,
    oob_default_binding,
    offset,
    offset_binding,
    temp_storage=None,
):
    value_type = _resolve_memory_type(source, primitive_name="load")
    if output.dtype is not None and _resolve_type(output.dtype) is not value_type:
        raise TypeError("cuda.coop.cutlass.load source dtype does not match output")
    request = _make_request(
        group=group,
        launch=launch,
        kind=GroupLoadStoreKind.LOAD,
        value_type=value_type,
        items_per_thread=output.items_per_thread,
        algorithm=algorithm,
        valid_items_binding=valid_items_binding,
        oob_default_binding=oob_default_binding,
        offset_binding=offset_binding,
        temp_storage=temp_storage,
    )
    pointer = _memory_pointer(
        source,
        primitive_name="load",
        required_elements=_required_static_elements(request),
    )
    runtime_types, runtime_args = _runtime_binding_args(
        request.operation,
        value_type=value_type,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
    )
    result = _make_rmem_tensor(output.items_per_thread, value_type, output.alignment)
    if valid_items is not None and oob_default is None:
        for i, value in enumerate(output._values):
            result[i] = value_type(0) if value is _UNSET else value_type(value)
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        scratch_types, scratch_args = _scratch_arguments(request, temp_storage)
        ffi(
            name=request.symbol_name,
            params_types=[
                llvm.PointerType.get(0),
                *runtime_types,
                *scratch_types,
                llvm.PointerType.get(0),
            ],
            return_type=None,
        )(pointer, *runtime_args, *scratch_args, result.iterator.llvm_ptr)
        output.dtype = value_type
        for i in range(output.items_per_thread):
            output[i] = result[i]
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


def provider_store(
    *,
    group,
    launch,
    destination,
    value,
    algorithm,
    valid_items,
    valid_items_binding,
    offset,
    offset_binding,
    temp_storage=None,
):
    if isinstance(value, ThreadData):
        value_type, values = _types.resolve_thread_data_value_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
            feature="store",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        values = tuple(values)
    else:
        value_type = _resolve_type(value, feature="store")
        values = (value,)
    if _resolve_memory_type(destination, primitive_name="store") is not value_type:
        raise TypeError(
            "cuda.coop.cutlass.store destination dtype does not match value dtype"
        )
    request = _make_request(
        group=group,
        launch=launch,
        kind=GroupLoadStoreKind.STORE,
        value_type=value_type,
        items_per_thread=len(values),
        algorithm=algorithm,
        valid_items_binding=valid_items_binding,
        oob_default_binding=ArgumentBinding.omitted(),
        offset_binding=offset_binding,
        temp_storage=temp_storage,
    )
    pointer = _memory_pointer(
        destination,
        primitive_name="store",
        required_elements=_required_static_elements(request),
    )
    runtime_types, runtime_args = _runtime_binding_args(
        request.operation,
        value_type=value_type,
        valid_items=valid_items,
        oob_default=None,
        offset=offset,
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        scratch_types, scratch_args = _scratch_arguments(request, temp_storage)
        ffi(
            name=request.symbol_name,
            params_types=[
                llvm.PointerType.get(0),
                *([value_type] * len(values)),
                *runtime_types,
                *scratch_types,
            ],
            return_type=None,
        )(pointer, *values, *runtime_args, *scratch_args)
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


def _resolve_type(value, *, allowed=ALL_PROVIDER_TYPES, feature="load/store"):
    return _types.resolve_provider_type(
        value,
        allowed=allowed,
        feature=feature,
        root_scope=_ROOT_SCOPE,
        namespace="load/store",
        canonical_type=_types.canonical_dsl_type,
    )


def _normalize_algorithm(algorithm: Any) -> GroupLoadStoreAlgorithm:
    token = getattr(algorithm, "value", algorithm)
    if isinstance(token, str):
        token = token.lower().replace("-", "_")
    try:
        return GroupLoadStoreAlgorithm(token)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in GroupLoadStoreAlgorithm)
        raise ValueError(
            f"{_ROOT_SCOPE}.load/store algorithm must be one of {choices}"
        ) from exc


def _make_group_load_store_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    kind: GroupLoadStoreKind,
    dtype: Any,
    items_per_thread: int,
    algorithm: Any,
    valid_items: ArgumentBinding,
    oob_default: ArgumentBinding,
    offset: ArgumentBinding,
    source: str = "cutlass_root",
    temp_storage=None,
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for group Load or Store."""

    operation = GroupLoadStoreSemantics(
        kind=kind,
        dtype=dtype,
        items_per_thread=items_per_thread,
        algorithm=_normalize_algorithm(algorithm),
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        storage_ownership=(
            StorageOwnership.IMPLEMENTATION
            if temp_storage is None
            else StorageOwnership.CALLER
        ),
        storage_sharing=None if temp_storage is None else temp_storage.sharing,
        storage_size_in_bytes=None
        if temp_storage is None
        else temp_storage.size_in_bytes,
        storage_alignment=None if temp_storage is None else temp_storage.alignment,
        storage_auto_sync=True if temp_storage is None else temp_storage.auto_sync,
    )
    call = make_group_primitive_call(group, operation, source=source)
    return plan_group_primitive(call, launch)


def _render_template_argument(
    request: _CubLoadStoreRequest,
    name: str,
    value: Any,
) -> str:
    if name == "T":
        if value is not request.value_type:
            raise ValueError("group load/store template dtype does not match request")
        return TYPE_SPECS[value].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render load/store template argument {name}={value!r}")


def _cpp_oob_literal(request: _CubLoadStoreRequest) -> str:
    value = getattr(request.operation.oob_default.value, "value", None)
    if value is None:
        value = request.operation.oob_default.value
    cpp_type = TYPE_SPECS[request.value_type].cpp_type
    if isinstance(value, bool):
        literal = "true" if value else "false"
    elif isinstance(value, Integral):
        literal = str(value)
    elif isinstance(value, Real) and math.isfinite(float(value)):
        literal = repr(float(value))
    else:
        raise TypeError("static oob_default must be a finite scalar literal")
    return f"static_cast<{cpp_type}>({literal})"


def _binding_expr(
    request: _CubLoadStoreRequest,
    binding: ArgumentBinding,
    *,
    runtime_name: str,
    oob_default: bool = False,
) -> str | None:
    if binding.kind is BindingKind.OMITTED:
        return None
    if binding.kind is BindingKind.RUNTIME:
        return runtime_name
    if oob_default:
        return _cpp_oob_literal(request)
    return str(binding.value)


def _memory_dtype(value: Any) -> Any:
    for name in ("element_type", "dtype", "_dtype"):
        dtype = getattr(value, name, None)
        if dtype is not None:
            return dtype
    iterator = getattr(value, "iterator", None)
    return getattr(iterator, "dtype", None)


@dataclasses.dataclass(frozen=True)
class _ContiguousMemoryProof:
    pointer: Any
    available_elements: int | None


def _is_local_memory_space(value: Any) -> bool:
    try:
        if int(value) == _LLVM_LOCAL_ADDRESS_SPACE:
            return True
    except Exception:
        pass
    try:
        name = str(getattr(value, "name", value)).strip().lower()
    except Exception:
        return False
    return name in {"local", "local_memory", "rmem"}


def _uses_local_memory(value: Any) -> bool:
    candidates = [value]
    for name in ("iterator", "pointer", "ptr", "_pointer", "_ptr"):
        try:
            candidate = getattr(value, name)
        except Exception:
            continue
        if candidate is not None:
            candidates.append(candidate)
    for candidate in candidates:
        for name in ("memspace", "space", "address_space"):
            try:
                memory_space = getattr(candidate, name)
            except Exception:
                continue
            if _is_local_memory_space(memory_space):
                return True
    return False


def _try_raw_memory_pointer(value: Any) -> Any | None:
    candidates = [value]
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        try:
            candidates.append(data_ptr())
        except Exception:
            pass
    for name in ("iterator", "pointer", "ptr", "_pointer", "_ptr"):
        try:
            candidate = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if candidate is not None:
            candidates.append(candidate)

    for candidate in candidates:
        to_llvm_ptr = getattr(candidate, "to_llvm_ptr", None)
        if callable(to_llvm_ptr):
            pointer = to_llvm_ptr()
        else:
            pointer = getattr(candidate, "llvm_ptr", None)
        if pointer is None:
            continue
        try:
            pointer_type = llvm.PointerType(pointer.type)
        except Exception:
            continue
        if pointer_type.address_space == _LLVM_LOCAL_ADDRESS_SPACE:
            return None
        if pointer_type.address_space != 0:
            pointer = llvm.addrspacecast(llvm.PointerType.get(0), pointer)
        return pointer

    return None


def _contiguous_memory_proof(
    value: Any,
    *,
    primitive_name: str,
) -> tuple[_ContiguousMemoryProof | None, str]:
    """Classify raw-pointer eligibility without registering a provider request."""

    layout_reason = contiguous_layout_reason(value)
    if layout_reason is not None:
        return None, layout_reason
    if _uses_local_memory(value):
        return None, "uses register/local memory instead of a primitive base pointer"
    pointer = _try_raw_memory_pointer(value)
    if pointer is None:
        return None, "does not expose a raw iterator/pointer"
    return (
        _ContiguousMemoryProof(pointer, static_layout_elements(value)),
        "raw pointer and compact layout proven",
    )


def _memory_pointer(
    value: Any,
    *,
    primitive_name: str,
    required_elements: int | None = None,
) -> Any:
    proof, reason = _contiguous_memory_proof(
        value,
        primitive_name=primitive_name,
    )
    if proof is not None:
        if (
            required_elements is not None
            and proof.available_elements is not None
            and required_elements > proof.available_elements
        ):
            raise ValueError(
                f"{_ROOT_SCOPE}.{primitive_name} requires {required_elements} "
                "elements after applying its static offset, group instances, "
                f"and valid_items, but the operand provides "
                f"{proof.available_elements}"
            )
        return proof.pointer

    operand = "tensor"
    raise NotImplementedError(
        f"{_ROOT_SCOPE}.{primitive_name} {operand} must prove a raw contiguous "
        f"iterator/pointer for CUB primitive lowering; {reason}"
    )


def _resolve_memory_type(value: Any, *, primitive_name: str) -> type:
    dtype = _memory_dtype(value)
    if dtype is None:
        raise TypeError(
            f"{_ROOT_SCOPE}.{primitive_name} memory operand must expose element_type "
            "or dtype"
        )
    return _resolve_type(
        dtype,
        allowed=ALL_PROVIDER_TYPES,
        feature=primitive_name,
    )


def _validate_static_oob_default(value: Any, value_type: type) -> None:
    plain_value = _provider_types.coerce_plain_scalar(
        value,
        value_type,
        name="load oob_default",
        scope=_ROOT_SCOPE,
        allow_nonfinite=False,
        convert=False,
    )
    if plain_value is not _provider_types._NOT_PLAIN_SCALAR:
        return
    try:
        actual_type = _resolve_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
            feature="load",
        )
    except (TypeError, NotImplementedError) as exc:
        raise TypeError(
            f"{_ROOT_SCOPE}.load oob_default must match the memory dtype"
        ) from exc
    if actual_type is not value_type:
        raise TypeError(f"{_ROOT_SCOPE}.load oob_default must match the memory dtype")
    scalar_value = getattr(value, "value", value)
    if isinstance(scalar_value, bool) or not isinstance(
        scalar_value,
        (Integral, Real),
    ):
        raise TypeError(
            f"{_ROOT_SCOPE}.load oob_default must be a finite scalar literal"
        )
    if isinstance(scalar_value, Real) and not math.isfinite(float(scalar_value)):
        raise ValueError(f"{_ROOT_SCOPE}.load oob_default must be finite")


def _coerce_runtime_oob_default(value: Any, value_type: type) -> Any:
    if isinstance(value, value_type):
        return value
    raise TypeError(
        f"{_ROOT_SCOPE}.load runtime oob_default must match the memory dtype "
        f"{value_type.__name__}"
    )


def _runtime_binding_args(
    operation: GroupLoadStoreSemantics,
    *,
    value_type: type,
    valid_items: Any,
    oob_default: Any,
    offset: Any,
) -> tuple[list[type], list[Any]]:
    param_types: list[type] = []
    args: list[Any] = []
    if operation.valid_items.kind is BindingKind.RUNTIME:
        param_types.append(Int32)
        args.append(_provider_types.as_valid_items_arg(valid_items, scope=_ROOT_SCOPE))
    if operation.oob_default.kind is BindingKind.RUNTIME:
        param_types.append(value_type)
        args.append(_coerce_runtime_oob_default(oob_default, value_type))
    if operation.offset.kind is BindingKind.RUNTIME:
        param_types.append(Int64)
        try:
            args.append(offset if isinstance(offset, Int64) else Int64(offset))
        except Exception as exc:
            raise TypeError(
                f"{_ROOT_SCOPE}.load/store offset must be convertible to Int64"
            ) from exc
    return param_types, args


def _required_static_elements(request: _CubLoadStoreRequest) -> int | None:
    """Return the largest statically reachable operand prefix."""

    operation = request.operation
    if (
        operation.valid_items.kind is BindingKind.RUNTIME
        or operation.offset.kind is BindingKind.RUNTIME
    ):
        return None

    group_size = request.plan.resolved_group.static_size
    if group_size is None:
        raise ValueError("group load/store request requires a static group size")
    tile_items = group_size * operation.items_per_thread
    valid_items = (
        tile_items
        if operation.valid_items.kind is BindingKind.OMITTED
        else int(operation.valid_items.value)
    )
    offset = (
        0
        if operation.offset.kind is BindingKind.OMITTED
        else int(operation.offset.value)
    )

    group_instances = 1
    if request.plan.target is GroupLoweringTarget.CUB_WARP:
        block_threads = math.prod(request.block_dim)
        group_instances, remainder = divmod(block_threads, group_size)
        if remainder or group_instances < 1:
            raise ValueError("group WarpLoad/Store requires complete group instances")
    return offset + (group_instances - 1) * tile_items + valid_items


def _scratch_arguments(request, temp_storage):
    if not request.uses_scratch:
        return (), ()
    from .._compiler._storage import register_deferred_temp_storage_event
    from .._temp_storage import TempStorage

    if temp_storage is None:
        temp_storage = TempStorage()
    elif not isinstance(temp_storage, TempStorage):
        raise TypeError(
            "cuda.coop.cutlass Load/Store scratch must be CUTLASS TempStorage"
        )
    arguments = register_deferred_temp_storage_event(
        temp_storage,
        primitive_name=request.operation_kind.value,
        requirement_key=request.scratch_requirement_key,
    )
    return (Uint32, Int32, Int32), arguments


def _scratch_layout_probe(request):
    if not request.uses_scratch:
        return None
    return _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key,
        f"typename {request.cpp_type}::TempStorage",
    )


_rendering.register_bundle_renderer(
    "cub_group_load_store",
    render=_render_cub_load_store,
    scratch_layout_probe=_scratch_layout_probe,
    include_lines=(
        "#include <cub/block/block_load.cuh>",
        "#include <cub/block/block_store.cuh>",
        "#include <cuda/std/cstdint>",
    ),
    cccl_headers=(
        ("cub/block/block_load.cuh", "cub/block/block_load.cuh"),
        ("cub/block/block_store.cuh", "cub/block/block_store.cuh"),
    ),
)
