# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven CUB provider renderer for CUTLASS group scans."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any

import numpy as np
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupScanSemantics,
    LaunchFacts,
    Reference,
    ScanMode,
    ScanValueKind,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    make_scan_semantics,
    plan_group_primitive,
)
from cuda.coop._core.block import BlockScanAlgorithm
from cuda.coop._core.thread_group import ThreadGroup

from .._compiler import _rendering, _state, _storage, _types
from .._compiler._types import ALL_PROVIDER_TYPES, TYPE_SPECS
from .._group_reduce import _classify_valid_items
from .._operators import OPERATOR_CPP, operator_expression, validate_operator_dtype
from .._thread_data import ThreadData, _make_rmem_tensor

_provider_rendering = _rendering
_provider_state = _state
_provider_storage = _storage
_provider_types = _types

_ROOT_SCOPE = "cuda.coop.cutlass"

_BLOCK_ALGORITHM_TOKENS = {
    BlockScanAlgorithm.RAKING.value: "raking",
    BlockScanAlgorithm.RAKING_MEMOIZE.value: "raking_memoize",
    BlockScanAlgorithm.WARP_SCANS.value: "warp_scans",
}
_SCAN_OPERATOR_CPP = OPERATOR_CPP


def _make_group_scan_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    value_kind: ScanValueKind,
    items_per_thread: int,
    mode: str,
    op: str,
    initial_value: Any = None,
    aggregate: bool = False,
    valid_items: Any = None,
    algorithm: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for one CUTLASS scan."""

    mode = ScanMode(mode).value
    if group.kind == "block" and algorithm is None:
        algorithm = BlockScanAlgorithm.RAKING
    if mode == ScanMode.INCLUSIVE.value and initial_value is not None:
        raise ValueError(
            f"{_ROOT_SCOPE}.scan initial_value is not supported for inclusive scans"
        )
    if mode == ScanMode.EXCLUSIVE.value and op != "sum" and initial_value is None:
        raise ValueError(
            f"{_ROOT_SCOPE}.scan requires initial_value for non-default exclusive scans"
        )

    initial_descriptor = (
        Reference(Dependency("T"), name="initial_value")
        if initial_value is not None
        else None
    )
    scan_operator = None
    if op != "sum" or initial_descriptor is not None:
        try:
            cpp = _SCAN_OPERATOR_CPP[op]
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported group scan operator {op!r}"
            ) from exc
        scan_operator = CxxOperator(
            cpp=cpp,
            dtype=Dependency("T"),
            name="scan_op",
        )

    primitive = make_scan_semantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        scan_operator=scan_operator,
        initial_value=initial_descriptor,
        aggregate=aggregate,
    )
    call = make_group_primitive_call(
        group,
        GroupScanSemantics(
            primitive=primitive,
            cub_algorithm=algorithm,
            valid_items=_classify_valid_items(valid_items, primitive="scan"),
        ),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _normalize_cpp_operator(cpp: str) -> str:
    return cpp.strip().replace("<T>", "<>").removesuffix("{}")


def _validate_scan_request_plan(
    plan: GroupLoweringPlan,
    *,
    op: str,
    value_type: type,
    external_scratch: bool,
) -> GroupScanSemantics:
    plan.require_supported()
    if plan.target not in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }:
        raise ValueError("CUB scan request requires a CUB lowering plan")
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("CUB scan request requires an AlgorithmSpec")

    operation = plan.call.operation
    if not isinstance(operation, GroupScanSemantics):
        raise TypeError("group scan request requires scan semantics")
    if operation.dtype is not value_type:
        raise ValueError("group scan request dtype does not match its plan")

    scan_operator = operation.scan_operator
    if scan_operator is None:
        if op != "sum":
            raise ValueError("group scan request operator does not match its plan")
    else:
        if not isinstance(scan_operator, CxxOperator):
            raise NotImplementedError(
                "CUTLASS group scan currently supports built-in C++ operators only"
            )
        expected_cpp = _normalize_cpp_operator(operator_expression(op))
        if _normalize_cpp_operator(scan_operator.cpp) != expected_cpp:
            raise ValueError("group scan request operator does not match its plan")

    initial_value = operation.initial_value
    if initial_value is not None and not isinstance(initial_value, Reference):
        if not (
            isinstance(initial_value, CxxFunction)
            and initial_value.cpp == "{T}{0}"
            and op == "sum"
            and operation.mode is ScanMode.EXCLUSIVE
            and operation.valid_items.kind is not BindingKind.OMITTED
        ):
            raise NotImplementedError(
                "Scan initial descriptor must be a runtime value or canonical typed zero"
            )

    implementation = plan.implementation
    expected_target = (
        GroupLoweringTarget.CUB_BLOCK
        if plan.resolved_group.kind == "block"
        else GroupLoweringTarget.CUB_WARP
    )
    if plan.target is not expected_target:
        raise ValueError("CUB scan target does not match its resolved group")
    expected_struct = (
        "BlockScan" if plan.target is GroupLoweringTarget.CUB_BLOCK else "WarpScan"
    )
    if implementation.struct_name != expected_struct:
        raise ValueError("CUB scan implementation does not match its group target")
    expected_prefix = (
        "Exclusive" if operation.mode is ScanMode.EXCLUSIVE else "Inclusive"
    )
    expected_method = (
        f"{expected_prefix}ScanPartial"
        if operation.valid_items.kind is not BindingKind.OMITTED
        else f"{expected_prefix}{'Sum' if scan_operator is None else 'Scan'}"
    )
    if implementation.method_name != expected_method:
        raise ValueError("CUB scan method does not match its plan semantics")

    template_arguments = implementation.template_arguments
    if template_arguments.get("T") is not value_type:
        raise ValueError("CUB scan template dtype does not match its request")
    if operation.operand_kind.value == ScanValueKind.ARRAY.value:
        if template_arguments.get("ITEMS_PER_THREAD") != operation.items_per_thread:
            raise ValueError("CUB scan item count does not match its request")
    elif "ITEMS_PER_THREAD" in template_arguments:
        raise ValueError("scalar CUB scan plan cannot carry an array item count")
    participation = plan.participation
    if participation is None:
        raise ValueError("CUB scan plan requires a participation contract")
    block_dim = participation.exact_block_dim
    if plan.target is GroupLoweringTarget.CUB_BLOCK:
        expected_dims = (
            template_arguments.get("BLOCK_DIM_X"),
            template_arguments.get("BLOCK_DIM_Y"),
            template_arguments.get("BLOCK_DIM_Z"),
        )
        if expected_dims != block_dim:
            raise ValueError("CUB BlockScan dimensions do not match its plan")
    else:
        if operation.operand_kind.value != ScanValueKind.SCALAR.value:
            raise ValueError("CUB WarpScan request requires a scalar operand")
        if template_arguments.get("VIRTUAL_WARP_THREADS") != (
            plan.resolved_group.static_size
        ):
            raise ValueError("group WarpScan width does not match its plan")

    temp_storage = plan.temp_storage
    if temp_storage is None:
        raise ValueError("CUB scan plan requires a temporary-storage contract")
    if external_scratch:
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise ValueError("external Scan scratch is block-scoped only")
        if (
            temp_storage.ownership
            not in {StorageOwnership.CALLER, StorageOwnership.IMPLEMENTATION}
            or temp_storage.address_space != "shared"
            or temp_storage.instances != 1
            or not temp_storage.exact_layout_required
        ):
            raise ValueError("BlockScan requires exact shared storage")
    elif temp_storage.ownership is not StorageOwnership.IMPLEMENTATION:
        raise ValueError("internal Scan scratch requires implementation ownership")

    result = plan.result
    if result is None or result.has_aggregate != operation.aggregate:
        raise ValueError("CUB scan result contract does not match aggregate semantics")
    return operation


@dataclasses.dataclass(frozen=True, eq=False)
class _CubScanRequest:
    plan: GroupLoweringPlan
    op: str
    value_type: type
    external_scratch: bool = False
    kind: str = "cub_group_scan"

    def __post_init__(self) -> None:
        _validate_scan_request_plan(
            self.plan,
            op=self.op,
            value_type=self.value_type,
            external_scratch=self.external_scratch,
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        if not self.external_scratch:
            return self.plan.artifact_key
        return "external_scratch", self.plan.artifact_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CubScanRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def operation(self) -> GroupScanSemantics:
        operation = self.plan.call.operation
        assert isinstance(operation, GroupScanSemantics)
        return operation

    @property
    def group(self) -> ThreadGroup:
        group = self.plan.resolved_group
        assert isinstance(group, ThreadGroup)
        return group

    @property
    def items_per_thread(self) -> int:
        return self.operation.items_per_thread

    @property
    def is_array(self) -> bool:
        return self.operation.primitive.value_kind is ScanValueKind.ARRAY

    @property
    def has_initial_value(self) -> bool:
        return self.operation.initial_value is not None

    @property
    def has_runtime_initial(self) -> bool:
        return isinstance(self.operation.initial_value, Reference)

    @property
    def has_aggregate(self) -> bool:
        return self.operation.aggregate

    @property
    def valid_items(self) -> ArgumentBinding:
        return self.operation.valid_items

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def _algorithm_suffix(self) -> str:
        if self.plan.target is GroupLoweringTarget.CUB_WARP:
            return "warp"
        implementation = self.plan.implementation
        assert isinstance(implementation, AlgorithmSpec)
        algorithm = implementation.template_arguments.get("ALGORITHM")
        try:
            return _BLOCK_ALGORITHM_TOKENS[algorithm]
        except KeyError as exc:
            raise ValueError(
                f"unsupported CUB BlockScan algorithm {algorithm!r}"
            ) from exc

    @property
    def symbol_name(self) -> str:
        implementation = self.plan.implementation
        assert isinstance(implementation, AlgorithmSpec)
        value_kind = f"x{self.items_per_thread}" if self.is_array else "scalar"
        initial = "initial" if self.has_initial_value else "noinit"
        aggregate = "aggregate" if self.has_aggregate else "value"
        valid_items = {
            BindingKind.OMITTED: "all",
            BindingKind.RUNTIME: "valid_runtime",
            BindingKind.STATIC: f"valid_{self.valid_items.value}",
        }[self.valid_items.kind]
        signature = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()[:12]
        symbol = (
            "cuda_coop_cutlass_cub_scan_"
            f"{self.group.symbol_suffix}_"
            f"{implementation.method_name.lower()}_{self.op}_"
            f"{TYPE_SPECS[self.value_type].token}_{value_kind}_"
            f"{self._algorithm_suffix}_{initial}_{aggregate}_{valid_items}_"
            f"{signature}"
        )
        if self.external_scratch:
            return f"{symbol}_external_scratch"
        return symbol

    @property
    def scratch_requirement_key(self) -> tuple[Any, ...]:
        """Identity of the instantiated CUB class whose layout is required."""

        implementation = self.plan.implementation
        assert isinstance(implementation, AlgorithmSpec)
        return (
            "cub_temp_storage_layout",
            implementation.struct_name,
            tuple(
                (name, _render_cub_template_argument(self, name, value))
                for name, value in implementation.ordered_template_arguments
            ),
        )

    @property
    def scratch_cpp_type(self) -> str:
        implementation = self.plan.implementation
        assert isinstance(implementation, AlgorithmSpec)
        template_arguments = ", ".join(
            _render_cub_template_argument(self, name, value)
            for name, value in implementation.ordered_template_arguments
        )
        return (
            f"typename ::cub::{implementation.struct_name}<"
            f"{template_arguments}>::TempStorage"
        )


def _render_cub_template_argument(
    request: _CubScanRequest,
    name: str,
    value: Any,
) -> str:
    if name == "T":
        if value is not request.value_type:
            raise ValueError("CUB scan template dtype does not match its request")
        return TYPE_SPECS[request.value_type].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render CUB scan template argument {name}={value!r}")


def _storage_reuse_barrier_line(plan: GroupLoweringPlan) -> str:
    synchronization = plan.synchronization
    if synchronization is None:
        raise ValueError("Scan plan requires a synchronization contract")
    if synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK:
        return "  __syncthreads();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.WARP:
        _, logical_width = _warp_instances(plan)
        mask = "0xffffffffu"
        if logical_width < 32:
            mask = f"{(1 << logical_width) - 1}u << ((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % 32u / {logical_width}u * {logical_width}u)"
        return f"  __syncwarp({mask});"
    if synchronization.storage_reuse_barrier is SynchronizationScope.NONE:
        return ""
    raise ValueError("Scan plan requires a storage reuse barrier")


def _warp_instances(plan: GroupLoweringPlan) -> tuple[int, int]:
    participation = plan.participation
    if participation is None:
        raise ValueError("WarpScan plan requires a participation contract")
    block_dim = participation.exact_block_dim
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    implementation = plan.implementation
    assert isinstance(implementation, AlgorithmSpec)
    logical_width = implementation.template_arguments.get("VIRTUAL_WARP_THREADS")
    if not isinstance(logical_width, int) or logical_width < 1:
        raise ValueError("WarpScan plan requires a static logical warp width")
    if block_threads < logical_width or block_threads % logical_width != 0:
        raise ValueError("WarpScan plan requires complete logical warps")
    return block_threads // logical_width, logical_width


def _render_cub_scan(request: _CubScanRequest) -> list[str]:
    operation = _validate_scan_request_plan(
        request.plan,
        op=request.op,
        value_type=request.value_type,
        external_scratch=request.external_scratch,
    )
    implementation = request.plan.implementation
    assert isinstance(implementation, AlgorithmSpec)
    spec = TYPE_SPECS[request.value_type]
    template_arguments = ", ".join(
        _render_cub_template_argument(request, name, value)
        for name, value in implementation.ordered_template_arguments
    )

    storage = "storage"
    storage_lines = ["  __shared__ typename implementation_type::TempStorage storage;"]
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
            "  unsigned long long generic_addr;",
            '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_addr) : "l"(static_cast<unsigned long long>(temp_storage_smem_addr)));',
            "  void* temp_storage_ptr = reinterpret_cast<void*>(generic_addr);",
            "  auto* storage_ptr = reinterpret_cast<",
            "      typename implementation_type::TempStorage*>(temp_storage_ptr);",
        ]
        storage = "*storage_ptr"
    elif request.plan.target is GroupLoweringTarget.CUB_WARP:
        instances, logical_width = _warp_instances(request.plan)
        storage_lines = [
            "  __shared__ typename implementation_type::TempStorage "
            f"storage[{instances}];",
            "  unsigned int storage_instance =",
            f"      (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / {logical_width}u;",
        ]
        storage = "storage[storage_instance]"

    params: list[str] = []
    input_lines: list[str] = []
    output_lines: list[str] = []
    if request.is_array:
        params.extend(
            f"{spec.cpp_type} item{index}" for index in range(request.items_per_thread)
        )
        values = ", ".join(f"item{index}" for index in range(request.items_per_thread))
        input_lines.extend(
            [
                f"  {spec.cpp_type} input_items[{request.items_per_thread}] = "
                f"{{{values}}};",
                f"  {spec.cpp_type} output_items[{request.items_per_thread}];",
            ]
        )
        call_arguments = ["input_items", "output_items"]
    else:
        params.append(f"{spec.cpp_type} value")
        input_lines.append(f"  {spec.cpp_type} result = value;")
        call_arguments = ["value", "result"]

    if request.has_runtime_initial:
        params.append(f"{spec.cpp_type} initial_value")
        call_arguments.append("initial_value")
    elif request.has_initial_value:
        call_arguments.append(f"static_cast<{spec.cpp_type}>(0)")
    if operation.scan_operator is not None or request.has_valid_items:
        call_arguments.append(operator_expression(request.op))
    if request.valid_items.kind is BindingKind.RUNTIME:
        params.append("int valid_items")
        call_arguments.append("valid_items")
    elif request.valid_items.kind is BindingKind.STATIC:
        call_arguments.append(str(request.valid_items.value))
    if request.external_scratch:
        params.extend(
            (
                "unsigned int temp_storage_smem_addr",
                "int temp_storage_bytes",
                "int temp_storage_auto_sync",
            )
        )
    if request.has_aggregate:
        params.append(f"{spec.cpp_type}* aggregate_output")
        input_lines.append(f"  {spec.cpp_type} aggregate;")
        call_arguments.append("aggregate")
    if request.is_array:
        params.append(f"{spec.cpp_type}* result_items")
        output_lines.extend(
            f"  result_items[{index}] = output_items[{index}];"
            for index in range(request.items_per_thread)
        )
    else:
        output_lines.append("  return result;")

    barrier = _storage_reuse_barrier_line(request.plan)
    barrier_lines = [barrier] if barrier else []
    if request.external_scratch and barrier:
        barrier_lines = [
            "  if (temp_storage_auto_sync != 0) {",
            f"  {barrier}",
            "  }",
        ]
    aggregate_lines = (
        ["  *aggregate_output = aggregate;"] if request.has_aggregate else []
    )
    return [
        f"{'void' if request.is_array else spec.cpp_type} "
        f"{request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = ::cub::{implementation.struct_name}<"
        f"{template_arguments}>;",
        *(
            [
                f"  if (valid_items < 1 || valid_items > {request.group.static_size}) {{",
                '    asm volatile("trap;" : : :);',
                "  }",
            ]
            if request.valid_items.kind is BindingKind.RUNTIME
            else []
        ),
        *storage_lines,
        *input_lines,
        f"  implementation_type({storage}).{implementation.method_name}("
        f"{', '.join(call_arguments)});",
        *barrier_lines,
        *aggregate_lines,
        *output_lines,
        "}",
    ]


def _cub_scan_scratch_layout_probe(
    request: _CubScanRequest,
) -> _provider_types.ScratchLayoutProbe | None:
    if not request.external_scratch:
        return None
    return _provider_rendering.make_scratch_layout_probe(
        request.scratch_requirement_key,
        request.scratch_cpp_type,
    )


def _register_renderer() -> None:
    _provider_rendering.register_bundle_renderer(
        "cub_group_scan",
        render=_render_cub_scan,
        include_lines=(
            "#include <cuda/functional>",
            "#include <cuda/std/functional>",
            "#include <cub/block/block_scan.cuh>",
            "#include <cub/warp/warp_scan.cuh>",
        ),
        cccl_headers=(
            ("#include <cub/block/block_scan.cuh>", "cub/block/block_scan.cuh"),
            ("#include <cub/warp/warp_scan.cuh>", "cub/warp/warp_scan.cuh"),
        ),
        scratch_layout_probe=_cub_scan_scratch_layout_probe,
    )


_register_renderer()

_resolve_type = _provider_types.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _typed_value(value, value_type, *, name="value", initial=False):
    if initial:
        if isinstance(value, np.generic):
            if _provider_types.canonical_dsl_type(value) is not value_type:
                raise TypeError(
                    "cuda.coop.cutlass.scan initial_value dtype must match value dtype"
                )
            value = value.item()
        if type(value) is float and not math.isfinite(value):
            raise ValueError("cuda.coop.cutlass.scan initial_value must be finite")
    converted = _provider_types.coerce_plain_scalar(
        value,
        value_type,
        name=f"scan {name}",
        scope=_ROOT_SCOPE,
        allow_nonfinite=not initial,
    )
    if converted is not _provider_types._NOT_PLAIN_SCALAR:
        return converted
    if initial and _provider_types.canonical_dsl_type(value) is not value_type:
        raise TypeError(
            "cuda.coop.cutlass.scan initial_value dtype must match value dtype"
        )
    return value_type(value)


def _validate_aggregate_output(output, *, value_type):
    if output is None:
        return
    if not isinstance(output, ThreadData):
        raise TypeError("cuda.coop.cutlass.scan aggregate_output must be ThreadData")
    if output.items_per_thread != 1:
        raise ValueError(
            "cuda.coop.cutlass.scan aggregate_output must contain one item"
        )
    if (
        output.dtype is not None
        and _provider_types.canonical_dsl_type(output.dtype) is not value_type
    ):
        raise TypeError(
            "cuda.coop.cutlass.scan aggregate_output dtype must match value dtype"
        )


def _with_block_storage(plan, descriptor):
    storage, synchronization = plan.temp_storage, plan.synchronization
    if (
        plan.target is not GroupLoweringTarget.CUB_BLOCK
        or storage is None
        or synchronization is None
    ):
        raise ValueError("Scan TempStorage applies only to block groups")
    return dataclasses.replace(
        plan,
        temp_storage=dataclasses.replace(
            storage,
            ownership=StorageOwnership.IMPLEMENTATION
            if descriptor is None
            else StorageOwnership.CALLER,
            exact_layout_required=True,
            sharing=None if descriptor is None else descriptor.sharing,
            requested_size_in_bytes=None
            if descriptor is None
            else descriptor.size_in_bytes,
            requested_alignment=None if descriptor is None else descriptor.alignment,
            auto_sync=True if descriptor is None else descriptor.auto_sync,
        ),
        synchronization=dataclasses.replace(
            synchronization,
            storage_reuse_barrier=SynchronizationScope.BLOCK
            if descriptor is None or descriptor.auto_sync
            else SynchronizationScope.NONE,
        ),
    )


def _materialize_scan(
    *,
    request,
    value,
    values,
    initial_value,
    aggregate_output,
    valid_items,
    temp_storage,
):
    value_type = request.value_type
    _validate_aggregate_output(aggregate_output, value_type=value_type)
    initial_args = []
    if request.has_runtime_initial:
        initial_args = [
            _typed_value(initial_value, value_type, name="initial_value", initial=True)
        ]
    elif initial_value is not None:
        raise ValueError("Scan initial value does not match its plan")
    typed_values = [_typed_value(item, value_type) for item in values]
    valid_args = []
    if request.valid_items.kind is BindingKind.RUNTIME:
        valid_args = [
            _provider_types.as_valid_items_arg(valid_items, scope=_ROOT_SCOPE)
        ]
    aggregate_tensor = (
        _make_rmem_tensor(1, value_type, aggregate_output.alignment)
        if aggregate_output is not None
        else None
    )
    result_tensor = (
        _make_rmem_tensor(request.items_per_thread, value_type, value.alignment)
        if request.is_array
        else None
    )
    snapshot = _provider_state.snapshot_active_session_state()
    try:
        _provider_state.register_request(request)
        scratch_args = ()
        if request.external_scratch:
            if temp_storage is None:
                from .._temp_storage import TempStorage

                temp_storage = TempStorage()
            scratch_args = _provider_storage.register_deferred_temp_storage_event(
                temp_storage,
                primitive_name="scan",
                requirement_key=request.scratch_requirement_key,
            )
        result = ffi(
            name=request.symbol_name,
            params_types=[
                *([value_type] * len(values)),
                *([value_type] if request.has_runtime_initial else []),
                *([Int32] if valid_args else []),
                *([Uint32, Int32, Int32] if request.external_scratch else []),
                *([llvm.PointerType.get(0)] if aggregate_tensor is not None else []),
                *([llvm.PointerType.get(0)] if result_tensor is not None else []),
            ],
            return_type=None if request.is_array else value_type,
        )(
            *typed_values,
            *initial_args,
            *valid_args,
            *scratch_args,
            *(
                [aggregate_tensor.iterator.llvm_ptr]
                if aggregate_tensor is not None
                else []
            ),
            *([result_tensor.iterator.llvm_ptr] if result_tensor is not None else []),
        )
        if aggregate_output is not None:
            aggregate_output.dtype = value_type
            aggregate_output[0] = value_type(aggregate_tensor[0])
        if result_tensor is not None:
            return ThreadData(
                request.items_per_thread,
                dtype=_provider_types.thread_data_output_dtype(value, value_type),
                values=[
                    value_type(result_tensor[i])
                    for i in range(request.items_per_thread)
                ],
                alignment=value.alignment,
            )
        return value_type(result)
    except BaseException:
        _provider_state.restore_active_session_state(snapshot)
        raise


def provider_scan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value: Any,
    mode: str = "exclusive",
    op: str = "sum",
    initial_value: Any = None,
    algorithm: Any = None,
    aggregate_output: Any = None,
    valid_items: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Materialize one current-plan CUB Scan with exact scratch and launch facts."""
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass.scan group must be a ThreadGroup")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("Scan provider requires exact launch facts")
    if temp_storage is not None:
        from .._temp_storage import TempStorage

        if not isinstance(temp_storage, TempStorage):
            raise TypeError(
                "cuda.coop.cutlass.scan temp_storage must be CUTLASS TempStorage"
            )
        if group.kind != "block":
            raise ValueError(
                "cuda.coop.cutlass.scan TempStorage applies only to block groups"
            )
    if isinstance(value, ThreadData):
        value_type, values = _provider_types.resolve_thread_data_value_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
            feature="scan",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        value_kind = ScanValueKind.ARRAY
    else:
        value_type = _resolve_type(value, allowed=ALL_PROVIDER_TYPES, feature="scan")
        values, value_kind = (value,), ScanValueKind.SCALAR
    validate_operator_dtype(op, value_type, primitive="scan")
    plan = _make_group_scan_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        value_kind=value_kind,
        items_per_thread=len(values),
        mode=mode,
        op=op,
        initial_value=initial_value,
        aggregate=aggregate_output is not None,
        valid_items=valid_items,
        algorithm=algorithm,
    ).require_supported()
    external_scratch = plan.target is GroupLoweringTarget.CUB_BLOCK
    if external_scratch:
        plan = _with_block_storage(plan, temp_storage)
    request = _CubScanRequest(plan, op, value_type, external_scratch=external_scratch)
    return _materialize_scan(
        request=request,
        value=value,
        values=values,
        initial_value=initial_value,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


__all__ = ["_CubScanRequest", "provider_scan"]
