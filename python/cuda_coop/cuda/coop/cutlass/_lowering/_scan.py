# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven CUB provider renderer for CUTLASS group scans."""

from __future__ import annotations

import dataclasses
import hashlib
from numbers import Integral
from typing import Any

from cutlass import cute as _cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
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

from .._compiler import _rendering as _provider_rendering
from .._compiler import _state as _provider_state
from .._compiler import _storage as _provider_storage
from .._compiler import _types as _provider_types
from .._compiler._call_context import get_active_single_phase_context
from .._compiler._types import SCAN_REDUCE_TYPES as _SCAN_REDUCE_TYPES
from .._compiler._types import TYPE_SPECS as _TYPE_SPECS
from .._thread_data import ThreadData
from .._thread_group import ThreadGroup
from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)

_ROOT_SCOPE = "cuda.coop.cutlass"

_BLOCK_ALGORITHM_TOKENS = {
    BlockScanAlgorithm.RAKING.value: "raking",
    BlockScanAlgorithm.RAKING_MEMOIZE.value: "raking_memoize",
    BlockScanAlgorithm.WARP_SCANS.value: "warp_scans",
}
_SCAN_OPERATOR_CPP = {
    "sum": "::cuda::std::plus<T>",
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}


def _is_boolean_control(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(value, np.bool_):
            return True
    try:
        from cutlass.base_dsl.typing import Boolean
    except ImportError:
        return False
    return isinstance(value, Boolean)


def _normalize_scan_mode(mode: Any) -> str:
    if _is_boolean_control(mode):
        raise TypeError(f"{_ROOT_SCOPE}.scan mode must not be boolean")
    try:
        return ScanMode(mode).value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_ROOT_SCOPE}.scan mode must be 'exclusive' or 'inclusive'"
        ) from exc


def _classify_valid_items(valid_items: Any) -> ArgumentBinding:
    if valid_items is None:
        return ArgumentBinding.omitted()
    if _is_boolean_control(valid_items):
        raise TypeError(f"{_ROOT_SCOPE}.scan valid_items must be an integer")
    if isinstance(valid_items, Integral):
        return ArgumentBinding.static(int(valid_items))
    from cutlass.base_dsl.typing import Integer

    if isinstance(valid_items, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(f"{_ROOT_SCOPE}.scan valid_items must be an integer")


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

    mode = _normalize_scan_mode(mode)
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
            valid_items=_classify_valid_items(valid_items),
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
        expected_cpp = _normalize_cpp_operator(_provider_types.cub_op_expr(op))
        if _normalize_cpp_operator(scan_operator.cpp) != expected_cpp:
            raise ValueError("group scan request operator does not match its plan")

    initial_value = operation.initial_value
    if initial_value is not None and not isinstance(initial_value, Reference):
        raise NotImplementedError(
            "CUTLASS group scan currently supports runtime initial values only"
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
    expected_storage_ownership = (
        StorageOwnership.CALLER if external_scratch else StorageOwnership.IMPLEMENTATION
    )
    if temp_storage.ownership is not expected_storage_ownership:
        raise ValueError("CUB group Scan storage ownership does not match its request")
    if external_scratch:
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise ValueError("caller-owned CUB Scan scratch is block-scoped only")
        if (
            temp_storage.address_space != "shared"
            or temp_storage.instances != 1
            or temp_storage.instance_index != "cta"
            or not temp_storage.exact_layout_required
        ):
            raise ValueError("CUB BlockScan requires exact caller-owned shared storage")

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
            f"{_TYPE_SPECS[self.value_type].token}_{value_kind}_"
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
        return _TYPE_SPECS[request.value_type].cpp_type
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
        return "  cuda_coop_cutlass_block_sync();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.WARP:
        return "  cuda_coop_cutlass_warp_sync();"
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
    spec = _TYPE_SPECS[request.value_type]
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
            "  void* temp_storage_ptr =",
            "      cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr);",
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
            f"      cuda_coop_cutlass_linear_tid() / {logical_width}u;",
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
        input_lines.append(f"  {spec.cpp_type} result;")
        call_arguments = ["value", "result"]

    if request.has_initial_value:
        params.append(f"{spec.cpp_type} initial_value")
        call_arguments.append("initial_value")
    elif (
        request.has_valid_items
        and operation.mode is ScanMode.EXCLUSIVE
        and request.op == "sum"
    ):
        call_arguments.append(spec.zero_literal)
    if operation.scan_operator is not None or request.has_valid_items:
        call_arguments.append(_provider_types.cub_op_expr(request.op))
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
    barrier_lines = [barrier]
    if request.external_scratch:
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

_resolve_type = _provider_state.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _validate_op_for_type(op: str, value_type: type) -> None:
    _provider_types.validate_scan_reduce_op_for_type(
        op,
        value_type,
        root_scope=_ROOT_SCOPE,
        feature="scan",
        namespace="thread_group",
    )


def _validate_aggregate_output(
    output: Any,
    *,
    value_type: type,
) -> ThreadData | None:
    return _provider_types.validate_thread_data_output(
        output=output,
        expected_items_per_thread=1,
        resolved_dtype=value_type,
        scope=_ROOT_SCOPE,
        primitive_name="scan",
        output_name="aggregate_output",
        resolve_type=_resolve_type,
        type_label=f"{_ROOT_SCOPE}.ThreadData",
    )


def _make_aggregate_tensor(
    aggregate_output: ThreadData | None,
    value_type: type,
) -> Any | None:
    if aggregate_output is None:
        return None
    return _cute.make_rmem_tensor(1, value_type)


def _populate_aggregate_output(
    aggregate_output: ThreadData | None,
    aggregate_tensor: Any | None,
) -> None:
    if aggregate_output is None:
        return
    assert aggregate_tensor is not None
    aggregate_output[0] = aggregate_tensor[0]


def _coerce_initial_value(
    request: _CubScanRequest,
    initial_value: Any,
) -> list[Any]:
    if not request.has_initial_value:
        if initial_value is not None:
            raise ValueError("scan initial value does not match its lowering plan")
        return []
    if initial_value is None:
        raise ValueError("scan lowering plan requires an initial value")
    return [
        _provider_types.coerce_scan_initial_value(
            initial_value=initial_value,
            value_type=request.value_type,
            root_scope=_ROOT_SCOPE,
            feature="scan",
            namespace="thread_group",
        )
    ]


def _temp_storage_for_scan(
    *,
    group: ThreadGroup,
    explicit_temp_storage: Any,
) -> Any | None:
    context = get_active_single_phase_context()
    context_storage = context.temp_storage if context is not None else None
    if (
        explicit_temp_storage is not None
        and context_storage is not None
        and explicit_temp_storage is not context_storage
    ):
        raise ValueError(f"{_ROOT_SCOPE}.scan received two TempStorage objects")
    temp_storage = (
        explicit_temp_storage if explicit_temp_storage is not None else context_storage
    )
    if temp_storage is None:
        return None

    from .._temp_storage import TempStorage

    if not isinstance(temp_storage, TempStorage):
        raise TypeError(
            f"{_ROOT_SCOPE}.scan temp_storage must be {_ROOT_SCOPE}.TempStorage"
        )
    if group.kind != "block":
        raise ValueError(
            f"{_ROOT_SCOPE}.scan TempStorage is supported only for block groups"
        )
    if not temp_storage.is_deferred and temp_storage.sharing == "exclusive":
        raise ValueError(
            f"{_ROOT_SCOPE}.scan fixed-capacity TempStorage does not support "
            "sharing='exclusive'; use sharing='shared' or deferred storage"
        )
    return temp_storage


def _external_scratch_args(
    temp_storage: Any,
    *,
    requirement_key: Any,
) -> tuple[Any, Any, Any]:
    if temp_storage.is_deferred:
        return _provider_storage.register_deferred_temp_storage_event(
            temp_storage,
            primitive_name="scan",
            requirement_key=requirement_key,
        )

    binding = _provider_storage.materialize_temp_storage_binding(
        temp_storage,
        scope=_ROOT_SCOPE,
        implicit_alignment=16,
    )
    return (
        binding.smem_addr_u32,
        Int32(binding.size_in_bytes),
        Int32(1 if binding.auto_sync else 0),
    )


def _with_caller_owned_scan_storage(plan: GroupLoweringPlan) -> GroupLoweringPlan:
    temp_storage = plan.temp_storage
    if temp_storage is None:
        raise ValueError("CUB scan plan requires a temporary-storage contract")
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


def _materialize_scan(
    *,
    plan: GroupLoweringPlan,
    value: Any,
    values: tuple[Any, ...],
    value_type: type,
    op: str,
    initial_value: Any,
    aggregate_output: Any,
    valid_items: Any,
    external_temp_storage: Any | None,
) -> Any:
    request = _CubScanRequest(
        plan=plan,
        op=op,
        value_type=value_type,
        external_scratch=external_temp_storage is not None,
    )
    aggregate_output_td = _validate_aggregate_output(
        aggregate_output,
        value_type=value_type,
    )
    if request.has_aggregate != (aggregate_output_td is not None):
        raise ValueError("scan aggregate output does not match its lowering plan")

    initial_args = _coerce_initial_value(request, initial_value)
    aggregate_tensor = _make_aggregate_tensor(aggregate_output_td, value_type)
    aggregate_args = (
        [aggregate_tensor.iterator.llvm_ptr] if aggregate_tensor is not None else []
    )
    aggregate_param_types = (
        [llvm.PointerType.get(0)] if aggregate_tensor is not None else []
    )
    if request.valid_items.kind is BindingKind.OMITTED:
        if valid_items is not None:
            raise ValueError("scan valid_items does not match its lowering plan")
        valid_args: list[Any] = []
        valid_param_types: list[type] = []
    elif request.valid_items.kind is BindingKind.STATIC:
        if valid_items != request.valid_items.value:
            raise ValueError("static scan valid_items does not match its plan")
        valid_args = []
        valid_param_types = []
    else:
        if valid_items is None:
            raise ValueError("scan lowering plan requires valid_items")
        valid_args = [
            _provider_types.as_valid_items_arg(valid_items, scope=_ROOT_SCOPE)
        ]
        valid_param_types = [Int32]

    assert plan.result is not None
    result_metadata = metadata_for_group(
        plan.resolved_group,
        visibility=plan.result.visibility,
    )
    session_snapshot = (
        _provider_state.snapshot_active_session_state()
        if external_temp_storage is not None
        else None
    )
    try:
        _provider_state.register_request(request)
        scratch_args: tuple[Any, ...] = ()
        scratch_param_types: list[type] = []
        if external_temp_storage is not None:
            scratch_addr, scratch_size, scratch_auto_sync = _external_scratch_args(
                external_temp_storage,
                requirement_key=request.scratch_requirement_key,
            )
            scratch_args = (scratch_addr, scratch_size, scratch_auto_sync)
            scratch_param_types = [Uint32, Int32, Int32]

        if request.is_array:
            result_tensor = _cute.make_rmem_tensor(request.items_per_thread, value_type)
            ffi(
                name=request.symbol_name,
                params_types=[
                    *([value_type] * request.items_per_thread),
                    *([value_type] if request.has_initial_value else []),
                    *valid_param_types,
                    *scratch_param_types,
                    *aggregate_param_types,
                    llvm.PointerType.get(0),
                ],
                return_type=None,
            )(
                *values,
                *initial_args,
                *valid_args,
                *scratch_args,
                *aggregate_args,
                result_tensor.iterator.llvm_ptr,
            )
            _populate_aggregate_output(aggregate_output_td, aggregate_tensor)
            assert isinstance(value, ThreadData)
            return attach_thread_data_metadata(
                ThreadData.from_values(
                    *(
                        result_tensor[index]
                        for index in range(request.items_per_thread)
                    ),
                    dtype=_provider_types.thread_data_output_dtype(
                        value,
                        value_type,
                    ),
                ),
                result_metadata,
            )

        result = ffi(
            name=request.symbol_name,
            params_types=[
                value_type,
                *([value_type] if request.has_initial_value else []),
                *valid_param_types,
                *scratch_param_types,
                *aggregate_param_types,
            ],
            return_type=value_type,
        )(
            values[0],
            *initial_args,
            *valid_args,
            *scratch_args,
            *aggregate_args,
        )
        _populate_aggregate_output(aggregate_output_td, aggregate_tensor)
        return _provider_state.remember_scalar_result_type(
            result,
            value_type,
            scope=_ROOT_SCOPE,
            compile_options_getter=lambda: (
                _provider_state._get_cute_dsl().compile_options
            ),
            group_metadata=result_metadata,
        )
    except Exception:
        if session_snapshot is not None:
            _provider_state.restore_active_session_state(session_snapshot)
        raise


def provider_scan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts | None = None,
    value: Any,
    mode: str = "exclusive",
    op: str = "sum",
    initial_value: Any = None,
    algorithm: Any = None,
    aggregate_output: Any = None,
    valid_items: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Materialize one planner-selected CUB BlockScan or physical WarpScan."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.scan group must be a ThreadGroup")
    if launch is None:
        if group.hierarchy is None or group.hierarchy.block_dim is None:
            raise TypeError("group scan provider requires exact launch facts")
        launch = LaunchFacts(exact_block_dim=group.hierarchy.block_dim)
    external_temp_storage = _temp_storage_for_scan(
        group=group,
        explicit_temp_storage=temp_storage,
    )

    def materialize(
        *,
        value_type: type,
        value_kind: ScanValueKind,
        values: tuple[Any, ...],
    ) -> Any:
        _validate_op_for_type(op, value_type)
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
            source="cutlass_group_scan_provider",
        ).require_supported()
        if external_temp_storage is not None:
            plan = _with_caller_owned_scan_storage(plan)
        validate_operand_domains(
            plan.resolved_group,
            {
                "value": value,
                **(
                    {"initial_value": initial_value}
                    if initial_value is not None
                    else {}
                ),
                **({"valid_items": valid_items} if valid_items is not None else {}),
            },
            scope=_ROOT_SCOPE,
            primitive_name="scan",
        )
        return _materialize_scan(
            plan=plan,
            value=value,
            values=values,
            value_type=value_type,
            op=op,
            initial_value=initial_value,
            aggregate_output=aggregate_output,
            valid_items=valid_items,
            external_temp_storage=external_temp_storage,
        )

    if isinstance(value, ThreadData):
        value_type, values = _provider_types.resolve_thread_data_value_type(
            value,
            allowed=_SCAN_REDUCE_TYPES,
            feature="scan",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        return materialize(
            value_type=value_type,
            value_kind=ScanValueKind.ARRAY,
            values=tuple(values),
        )

    value_type = _resolve_type(
        value,
        allowed=_SCAN_REDUCE_TYPES,
        feature="scan",
    )
    return materialize(
        value_type=value_type,
        value_kind=ScanValueKind.SCALAR,
        values=(value,),
    )


__all__ = ["_CubScanRequest", "provider_scan"]
