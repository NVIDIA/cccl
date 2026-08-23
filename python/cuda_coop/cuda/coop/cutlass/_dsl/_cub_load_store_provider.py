# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider renderer for group Load and Store."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from numbers import Integral, Real
from typing import Any

from cutlass import cute as _cute
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
)

from .. import _group_load_store as _group_frontend
from .._internal import ThreadData
from .._prims import is_cutlass_array_operand
from .._thread_group import ThreadGroup
from . import _provider as _provider_support
from ._load_store import contiguous_layout_reason, static_layout_elements
from ._provider import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from ._provider import TYPE_SPECS as _TYPE_SPECS
from ._single_phase import get_active_single_phase_context
from ._symbols import block_dim_token as _block_dim_token

_ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
_MAX_STATIC_OFFSET = (1 << 63) - 1


def _validate_binding_payload(
    binding: ArgumentBinding,
    value: Any,
    *,
    name: str,
) -> None:
    if binding.kind is BindingKind.OMITTED:
        if value is not None:
            raise ValueError(f"omitted {name} binding cannot carry a value")
        return
    if binding.kind is BindingKind.RUNTIME:
        if value is None:
            raise ValueError(f"runtime {name} binding requires a value")
        return
    if value != binding.value:
        raise ValueError(f"static {name} value does not match its binding")


def _validate_plan(
    plan: GroupLoweringPlan,
    *,
    value_type: type,
    kind: GroupLoadStoreKind,
    external_scratch: bool,
) -> GroupLoadStoreSemantics:
    plan.require_supported()
    if plan.target not in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }:
        raise ValueError("group load/store request requires a CUB lowering plan")
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("group load/store request requires an AlgorithmSpec")
    operation = plan.call.operation
    if not isinstance(operation, GroupLoadStoreSemantics):
        raise TypeError("group load/store request requires load/store semantics")
    if operation.kind is not kind:
        raise ValueError("group load/store request kind does not match its plan")
    if operation.dtype is not value_type:
        raise ValueError("group load/store request dtype does not match its plan")
    if operation.valid_items.kind is BindingKind.STATIC:
        valid_items = operation.valid_items.value
        if isinstance(valid_items, bool) or not isinstance(valid_items, Integral):
            raise TypeError("static group load/store valid_items must be an integer")
        group_size = plan.resolved_group.static_size
        assert group_size is not None
        tile_items = group_size * operation.items_per_thread
        if not 0 <= valid_items <= tile_items:
            raise ValueError(
                "static group load/store valid_items must be between zero and "
                f"the group tile size ({tile_items})"
            )
    if operation.offset.kind is BindingKind.STATIC:
        offset = operation.offset.value
        if isinstance(offset, bool) or not isinstance(offset, Integral):
            raise TypeError("static group load/store offset must be an integer")
        if offset < 0:
            raise ValueError("static group load/store offset must be non-negative")
        if offset > _MAX_STATIC_OFFSET:
            raise ValueError(
                "static group load/store offset must fit a signed 64-bit integer"
            )
    if operation.oob_default.kind is BindingKind.STATIC:
        _validate_static_oob_default(operation.oob_default.value, value_type)

    expected_target = (
        GroupLoweringTarget.CUB_BLOCK
        if plan.resolved_group.kind == "block"
        else GroupLoweringTarget.CUB_WARP
    )
    if plan.target is not expected_target:
        raise ValueError("group load/store target does not match its group")
    implementation = plan.implementation
    expected_struct = (
        f"Block{kind.value.title()}"
        if expected_target is GroupLoweringTarget.CUB_BLOCK
        else f"Warp{kind.value.title()}"
    )
    if implementation.struct_name != expected_struct:
        raise ValueError("group load/store implementation does not match its plan")
    if implementation.method_name != kind.value.title():
        raise ValueError("group load/store method does not match its plan")
    template_arguments = implementation.template_arguments
    if template_arguments.get("T") is not value_type:
        raise ValueError("group load/store template dtype does not match its plan")
    if template_arguments.get("ITEMS_PER_THREAD") != operation.items_per_thread:
        raise ValueError("group load/store item count does not match its plan")
    if expected_target is GroupLoweringTarget.CUB_WARP:
        if template_arguments.get("LOGICAL_WARP_THREADS") != (
            plan.resolved_group.static_size
        ):
            raise ValueError("group WarpLoad/Store width does not match its plan")
    else:
        participation = plan.participation
        if participation is None:
            raise ValueError("group BlockLoad/Store requires participation metadata")
        expected_dims = (
            template_arguments.get("BLOCK_DIM_X"),
            template_arguments.get("BLOCK_DIM_Y"),
            template_arguments.get("BLOCK_DIM_Z"),
        )
        if expected_dims != participation.exact_block_dim:
            raise ValueError("group BlockLoad/Store dimensions do not match its plan")

    temp_storage = plan.temp_storage
    if temp_storage is None:
        raise ValueError("group load/store requires a temporary-storage contract")
    expected_ownership = (
        StorageOwnership.CALLER if external_scratch else StorageOwnership.IMPLEMENTATION
    )
    if temp_storage.ownership is not expected_ownership:
        raise ValueError("group load/store storage ownership does not match request")
    if external_scratch:
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise ValueError("deferred CUB Load/Store scratch is block-scoped only")
        if (
            temp_storage.address_space != "shared"
            or temp_storage.instances != 1
            or temp_storage.instance_index != "cta"
            or not temp_storage.exact_layout_required
        ):
            raise ValueError(
                "deferred CUB BlockLoad/Store requires exact caller-owned "
                "shared storage"
            )
    if kind is GroupLoadStoreKind.LOAD:
        if plan.result is None or plan.result.result_items_per_thread != (
            operation.items_per_thread
        ):
            raise ValueError("group Load result does not match its item count")
    elif plan.result is not None:
        raise ValueError("group Store must not expose a result contract")
    return operation


@dataclasses.dataclass(frozen=True, eq=False)
class _CubLoadStoreRequest:
    plan: GroupLoweringPlan
    value_type: type
    external_scratch: bool = False
    kind: str = "cub_group_load_store"

    def __post_init__(self) -> None:
        _validate_plan(
            self.plan,
            value_type=self.value_type,
            kind=self.operation_kind,
            external_scratch=self.external_scratch,
        )

    @property
    def operation(self) -> GroupLoadStoreSemantics:
        operation = self.plan.call.operation
        if not isinstance(operation, GroupLoadStoreSemantics):
            raise TypeError("group load/store request requires load/store semantics")
        return operation

    @property
    def operation_kind(self) -> GroupLoadStoreKind:
        operation = self.plan.call.operation
        if not isinstance(operation, GroupLoadStoreSemantics):
            raise TypeError("group load/store request requires load/store semantics")
        return operation.kind

    @property
    def implementation(self) -> AlgorithmSpec:
        implementation = self.plan.implementation
        if not isinstance(implementation, AlgorithmSpec):
            raise TypeError("group load/store request requires an AlgorithmSpec")
        return implementation

    @property
    def block_dim(self) -> tuple[int, int, int]:
        participation = self.plan.participation
        if participation is None or participation.exact_block_dim is None:
            raise ValueError("group load/store request requires exact block dimensions")
        return participation.exact_block_dim

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        if not self.external_scratch:
            return self.plan.artifact_key
        return "external_scratch", self.plan.artifact_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CubLoadStoreRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def symbol_name(self) -> str:
        signature_plan = self.plan
        if self.external_scratch:
            assert self.plan.temp_storage is not None
            signature_plan = dataclasses.replace(
                self.plan,
                temp_storage=dataclasses.replace(
                    self.plan.temp_storage,
                    ownership=StorageOwnership.IMPLEMENTATION,
                    address_space=None,
                    cpp_type=None,
                    instances=None,
                    instance_index=None,
                    exact_layout_required=False,
                ),
            )
        assert signature_plan.artifact_key is not None
        signature = hashlib.sha256(
            repr(signature_plan.artifact_key).encode()
        ).hexdigest()[:12]
        symbol = (
            "cuda_coop_cutlass_cub_"
            f"{self.operation_kind.value}_{self.plan.resolved_group.kind}_"
            f"{_block_dim_token(self.block_dim)}_"
            f"{self.operation.algorithm.value}_"
            f"{_TYPE_SPECS[self.value_type].token}_"
            f"x{self.operation.items_per_thread}_{signature}"
        )
        if self.external_scratch:
            return f"{symbol}_external_scratch"
        return symbol

    @property
    def scratch_requirement_key(self) -> tuple[Any, ...]:
        return (
            "cub_temp_storage_layout",
            self.implementation.struct_name,
            tuple(
                (name, _render_template_argument(self, name, value))
                for name, value in self.implementation.ordered_template_arguments
            ),
        )

    @property
    def scratch_cpp_type(self) -> str:
        template_arguments = ", ".join(
            _render_template_argument(self, name, value)
            for name, value in self.implementation.ordered_template_arguments
        )
        return (
            f"typename ::cub::{self.implementation.struct_name}<"
            f"{template_arguments}>::TempStorage"
        )


def _render_template_argument(
    request: _CubLoadStoreRequest,
    name: str,
    value: Any,
) -> str:
    if name == "T":
        if value is not request.value_type:
            raise ValueError("group load/store template dtype does not match request")
        return _TYPE_SPECS[value].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render load/store template argument {name}={value!r}")


def _cpp_oob_literal(request: _CubLoadStoreRequest) -> str:
    value = getattr(request.operation.oob_default.value, "value", None)
    if value is None:
        value = request.operation.oob_default.value
    cpp_type = _TYPE_SPECS[request.value_type].cpp_type
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


def _storage_reuse_barrier_line(request: _CubLoadStoreRequest) -> str:
    synchronization = request.plan.synchronization
    if synchronization is None:
        raise ValueError("group load/store requires a synchronization contract")
    if synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK:
        return "  cuda_coop_cutlass_block_sync();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.WARP:
        return "  cuda_coop_cutlass_warp_sync();"
    raise ValueError("group load/store requires a block or warp reuse barrier")


def _render_cub_load_store(request: _CubLoadStoreRequest) -> list[str]:
    request.__post_init__()
    operation = request.operation
    implementation = request.implementation
    spec = _TYPE_SPECS[request.value_type]
    template_arguments = ", ".join(
        _render_template_argument(request, name, value)
        for name, value in implementation.ordered_template_arguments
    )
    is_load = operation.kind is GroupLoadStoreKind.LOAD
    params = [f"{'const ' if is_load else ''}{spec.cpp_type}* base"]
    if not is_load:
        params.extend(
            f"{spec.cpp_type} item{index}"
            for index in range(operation.items_per_thread)
        )
    if operation.valid_items.kind is BindingKind.RUNTIME:
        params.append("int valid_items")
    if operation.oob_default.kind is BindingKind.RUNTIME:
        params.append(f"{spec.cpp_type} oob_default")
    if operation.offset.kind is BindingKind.RUNTIME:
        params.append("long long offset")
    if request.external_scratch:
        params.extend(
            (
                "unsigned int temp_storage_smem_addr",
                "int temp_storage_bytes",
                "int temp_storage_auto_sync",
            )
        )
    if is_load:
        params.append(f"{spec.cpp_type}* result_items")

    storage = "storage"
    storage_lines = ["  __shared__ typename implementation_type::TempStorage storage;"]
    tile_pointer_lines: list[str] = []
    if request.external_scratch:
        storage_lines = [
            "  constexpr unsigned long long required_temp_bytes =",
            "      (unsigned long long)sizeof(typename implementation_type::TempStorage);",
            "  constexpr unsigned long long required_temp_alignment =",
            "      (unsigned long long)alignof(typename implementation_type::TempStorage);",
            "  if (temp_storage_bytes <= 0 ||",
            "      (unsigned long long)temp_storage_bytes < required_temp_bytes ||",
            "      ((unsigned long long)temp_storage_smem_addr &",
            "       (required_temp_alignment - 1ull)) != 0ull) {",
            '    asm volatile("trap;");',
            "  }",
            "  void* temp_storage_ptr =",
            "      cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr);",
            "  auto* storage_ptr = reinterpret_cast<",
            "      typename implementation_type::TempStorage*>(temp_storage_ptr);",
        ]
        storage = "*storage_ptr"
    elif request.plan.target is GroupLoweringTarget.CUB_WARP:
        block_threads = math.prod(request.block_dim)
        logical_width = implementation.template_arguments.get("LOGICAL_WARP_THREADS")
        if not isinstance(logical_width, int) or logical_width < 1:
            raise ValueError(
                "group WarpLoad/Store requires a static logical warp width"
            )
        if block_threads < logical_width or block_threads % logical_width != 0:
            raise ValueError("group WarpLoad/Store requires complete logical warps")
        storage_lines = [
            "  __shared__ typename implementation_type::TempStorage "
            f"storage[{block_threads // logical_width}];",
            "  unsigned int storage_instance =",
            f"      cuda_coop_cutlass_linear_tid() / {logical_width}u;",
        ]
        storage = "storage[storage_instance]"
        tile_pointer_lines = [
            "  tile_ptr += static_cast<long long>(storage_instance) * "
            f"{logical_width * operation.items_per_thread}ll;"
        ]

    offset_expr = _binding_expr(
        request,
        operation.offset,
        runtime_name="offset",
    )
    pointer_type = f"{'const ' if is_load else ''}{spec.cpp_type}*"
    pointer_lines = [f"  {pointer_type} tile_ptr = base;"]
    if offset_expr is not None:
        pointer_lines.append(f"  tile_ptr += {offset_expr};")
    pointer_lines.extend(tile_pointer_lines)

    if is_load:
        item_lines = [f"  {spec.cpp_type} items[{operation.items_per_thread}];"]
    else:
        values = ", ".join(
            f"item{index}" for index in range(operation.items_per_thread)
        )
        item_lines = [
            f"  {spec.cpp_type} items[{operation.items_per_thread}] = {{{values}}};"
        ]
    call_arguments = ["tile_ptr", "items"]
    valid_items_expr = _binding_expr(
        request,
        operation.valid_items,
        runtime_name="valid_items",
    )
    if valid_items_expr is not None:
        call_arguments.append(valid_items_expr)
    oob_default_expr = _binding_expr(
        request,
        operation.oob_default,
        runtime_name="oob_default",
        oob_default=True,
    )
    if oob_default_expr is not None:
        call_arguments.append(oob_default_expr)

    output_lines = []
    if is_load:
        output_lines = [
            f"  result_items[{index}] = items[{index}];"
            for index in range(operation.items_per_thread)
        ]
    barrier = _storage_reuse_barrier_line(request)
    barrier_lines = [barrier]
    if request.external_scratch:
        barrier_lines = [
            "  if (temp_storage_auto_sync != 0) {",
            f"  {barrier}",
            "  }",
        ]
    return [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = ::cub::{implementation.struct_name}<"
        f"{template_arguments}>;",
        *storage_lines,
        *pointer_lines,
        *item_lines,
        f"  implementation_type({storage}).{implementation.method_name}("
        f"{', '.join(call_arguments)});",
        *barrier_lines,
        *output_lines,
        "}",
    ]


def _cub_load_store_scratch_layout_probe(
    request: _CubLoadStoreRequest,
) -> _provider_support.ScratchLayoutProbe | None:
    if not request.external_scratch:
        return None
    return _provider_support.make_scratch_layout_probe(
        request.scratch_requirement_key,
        request.scratch_cpp_type,
    )


def _register_renderer() -> None:
    _provider_support.register_bundle_renderer(
        "cub_group_load_store",
        render=_render_cub_load_store,
        include_lines=(
            "#include <cub/block/block_load.cuh>",
            "#include <cub/block/block_store.cuh>",
            "#include <cub/warp/warp_load.cuh>",
            "#include <cub/warp/warp_store.cuh>",
        ),
        cccl_headers=(
            ("#include <cub/block/block_load.cuh>", "cub/block/block_load.cuh"),
            ("#include <cub/block/block_store.cuh>", "cub/block/block_store.cuh"),
            ("#include <cub/warp/warp_load.cuh>", "cub/warp/warp_load.cuh"),
            ("#include <cub/warp/warp_store.cuh>", "cub/warp/warp_store.cuh"),
        ),
        scratch_layout_probe=_cub_load_store_scratch_layout_probe,
    )


_register_renderer()

_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


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

    operand = "cutlass.Array" if is_cutlass_array_operand(value) else "tensor"
    raise NotImplementedError(
        f"{_ROOT_SCOPE}.{primitive_name} {operand} must prove a raw contiguous "
        f"iterator/pointer for CUB collective lowering; {reason}"
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
        allowed=_ALL_PROVIDER_TYPES,
        feature=primitive_name,
    )


def _validate_static_oob_default(value: Any, value_type: type) -> None:
    plain_value = _provider_support.coerce_plain_scalar(
        value,
        value_type,
        name="load oob_default",
        scope=_ROOT_SCOPE,
        allow_nonfinite=False,
        convert=False,
    )
    if plain_value is not _provider_support._NOT_PLAIN_SCALAR:
        return
    try:
        actual_type = _resolve_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
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
        args.append(
            _provider_support.as_valid_items_arg(valid_items, scope=_ROOT_SCOPE)
        )
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


def _deferred_temp_storage_for_load_store(
    *,
    group: ThreadGroup,
    explicit_temp_storage: Any,
    primitive_name: str,
) -> Any | None:
    context = get_active_single_phase_context()
    context_storage = context.temp_storage if context is not None else None
    if (
        explicit_temp_storage is not None
        and context_storage is not None
        and explicit_temp_storage is not context_storage
    ):
        raise ValueError(
            f"{_ROOT_SCOPE}.{primitive_name} received two TempStorage objects"
        )
    temp_storage = (
        explicit_temp_storage if explicit_temp_storage is not None else context_storage
    )
    if temp_storage is None:
        return None

    from .block._single_phase import TempStorage

    if not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"{_ROOT_SCOPE}.{primitive_name} temp_storage must be "
            f"{_ROOT_SCOPE}.TempStorage"
        )
    if not temp_storage.is_deferred:
        return None
    if group.kind != "block":
        raise ValueError(
            f"{_ROOT_SCOPE}.{primitive_name} deferred TempStorage is supported "
            "only for block groups"
        )
    return temp_storage


def _with_caller_owned_load_store_storage(
    plan: GroupLoweringPlan,
) -> GroupLoweringPlan:
    temp_storage = plan.temp_storage
    if temp_storage is None:
        raise ValueError("group load/store requires a temporary-storage contract")
    return dataclasses.replace(
        plan,
        temp_storage=dataclasses.replace(
            temp_storage,
            ownership=StorageOwnership.CALLER,
            address_space="shared",
            cpp_type="typename implementation_type::TempStorage",
            instances=1,
            instance_index="cta",
            exact_layout_required=True,
        ),
    )


def _make_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    kind: GroupLoadStoreKind,
    value_type: type,
    items_per_thread: int,
    algorithm: Any,
    valid_items_binding: ArgumentBinding,
    oob_default_binding: ArgumentBinding,
    offset_binding: ArgumentBinding,
    external_scratch: bool,
) -> _CubLoadStoreRequest:
    plan = _group_frontend._make_group_load_store_plan(
        group=group,
        launch=launch,
        kind=kind,
        dtype=value_type,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        offset=offset_binding,
        source="cutlass_group_load_store_provider",
    ).require_supported()
    if external_scratch:
        plan = _with_caller_owned_load_store_storage(plan)
    return _CubLoadStoreRequest(
        plan=plan,
        value_type=value_type,
        external_scratch=external_scratch,
    )


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


def provider_load(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    source: Any,
    output: ThreadData,
    algorithm: GroupLoadStoreAlgorithm,
    valid_items: Any,
    valid_items_binding: ArgumentBinding,
    oob_default: Any,
    oob_default_binding: ArgumentBinding,
    offset: Any,
    offset_binding: ArgumentBinding,
    temp_storage: Any = None,
) -> ThreadData:
    """Materialize one CUB BlockLoad or physical WarpLoad call."""

    if not isinstance(output, ThreadData):
        raise TypeError(f"{_ROOT_SCOPE}.load output must be ThreadData")
    _validate_binding_payload(
        valid_items_binding,
        valid_items,
        name="valid_items",
    )
    _validate_binding_payload(
        oob_default_binding,
        oob_default,
        name="oob_default",
    )
    _validate_binding_payload(offset_binding, offset, name="offset")
    value_type = _resolve_memory_type(source, primitive_name="load")
    if output.dtype is not None:
        output_type = _resolve_type(
            output.dtype,
            allowed=_ALL_PROVIDER_TYPES,
            feature="load",
        )
        if output_type is not value_type:
            raise TypeError(f"{_ROOT_SCOPE}.load source dtype does not match output")
    deferred_temp_storage = _deferred_temp_storage_for_load_store(
        group=group,
        explicit_temp_storage=temp_storage,
        primitive_name="load",
    )
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
        external_scratch=deferred_temp_storage is not None,
    )
    source_pointer = _memory_pointer(
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
    result_tensor = _cute.make_rmem_tensor(output.items_per_thread, value_type)
    session_snapshot = (
        _provider_support.snapshot_active_session_state()
        if deferred_temp_storage is not None
        else None
    )
    try:
        _provider_support.register_request(request)
        scratch_args: tuple[Any, ...] = ()
        scratch_param_types: list[type] = []
        if deferred_temp_storage is not None:
            scratch_args = _provider_support.register_deferred_temp_storage_event(
                deferred_temp_storage,
                primitive_name="load",
                requirement_key=request.scratch_requirement_key,
            )
            scratch_param_types = [Uint32, Int32, Int32]
        ffi(
            name=request.symbol_name,
            params_types=[
                llvm.PointerType.get(0),
                *runtime_types,
                *scratch_param_types,
                llvm.PointerType.get(0),
            ],
            return_type=None,
        )(
            source_pointer,
            *runtime_args,
            *scratch_args,
            result_tensor.iterator.llvm_ptr,
        )
        if output.dtype is None:
            output.dtype = value_type
        for index in range(output.items_per_thread):
            output[index] = result_tensor[index]
        return output
    except Exception:
        if session_snapshot is not None:
            _provider_support.restore_active_session_state(session_snapshot)
        raise


def provider_store(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    destination: Any,
    value: Any,
    algorithm: GroupLoadStoreAlgorithm,
    valid_items: Any,
    valid_items_binding: ArgumentBinding,
    offset: Any,
    offset_binding: ArgumentBinding,
    temp_storage: Any = None,
) -> None:
    """Materialize one CUB BlockStore or physical WarpStore call."""

    _validate_binding_payload(
        valid_items_binding,
        valid_items,
        name="valid_items",
    )
    _validate_binding_payload(offset_binding, offset, name="offset")
    if isinstance(value, ThreadData):
        value_type, values = _provider_support.resolve_thread_data_value_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="store",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        values = tuple(values)
    else:
        value_type = _resolve_type(
            value,
            allowed=_ALL_PROVIDER_TYPES,
            feature="store",
        )
        values = (value,)
    destination_type = _resolve_memory_type(destination, primitive_name="store")
    if destination_type is not value_type:
        raise TypeError(
            f"{_ROOT_SCOPE}.store destination dtype does not match value dtype"
        )
    deferred_temp_storage = _deferred_temp_storage_for_load_store(
        group=group,
        explicit_temp_storage=temp_storage,
        primitive_name="store",
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
        external_scratch=deferred_temp_storage is not None,
    )
    destination_pointer = _memory_pointer(
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
    session_snapshot = (
        _provider_support.snapshot_active_session_state()
        if deferred_temp_storage is not None
        else None
    )
    try:
        _provider_support.register_request(request)
        scratch_args: tuple[Any, ...] = ()
        scratch_param_types: list[type] = []
        if deferred_temp_storage is not None:
            scratch_args = _provider_support.register_deferred_temp_storage_event(
                deferred_temp_storage,
                primitive_name="store",
                requirement_key=request.scratch_requirement_key,
            )
            scratch_param_types = [Uint32, Int32, Int32]
        ffi(
            name=request.symbol_name,
            params_types=[
                llvm.PointerType.get(0),
                *([value_type] * len(values)),
                *runtime_types,
                *scratch_param_types,
            ],
            return_type=None,
        )(
            destination_pointer,
            *values,
            *runtime_args,
            *scratch_args,
        )
    except Exception:
        if session_snapshot is not None:
            _provider_support.restore_active_session_state(session_snapshot)
        raise


__all__ = [
    "_CubLoadStoreRequest",
    "provider_load",
    "provider_store",
]
