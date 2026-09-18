# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven CUDAX/CUB provider renderer for CUTLASS group reductions."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from cutlass.base_dsl.typing import Int32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
    CudaxCallDescription,
    CudaxReturnKind,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupReduceSemantics,
    LaunchFacts,
    ReduceOperation,
    ReduceValueKind,
    SynchronizationScope,
    make_group_primitive_call,
    make_reduce_semantics,
    plan_group_primitive,
    render_group_decl_lines,
    render_hierarchy_decl,
)
from cuda.coop._core.block import BlockReduceAlgorithm
from cuda.coop._core.thread_group import ThreadGroup

from .._compiler import _rendering, _state, _types
from .._compiler._types import ALL_PROVIDER_TYPES, TYPE_SPECS
from .._group_reduce import _classify_valid_items
from .._operators import OPERATOR_CPP, operator_expression, validate_operator_dtype
from .._thread_data import ThreadData

_provider_rendering = _rendering
_provider_state = _state
_provider_types = _types

_ROOT_SCOPE = "cuda.coop.cutlass"
_GRID_REDUCE_UNAVAILABLE = f"{_ROOT_SCOPE}.reduce does not support grid groups"
_REDUCE_OPERATOR_CPP = OPERATOR_CPP


def _make_group_reduce_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    value_kind: ReduceValueKind,
    items_per_thread: int,
    op: str,
    broadcast: bool,
    valid_items: ArgumentBinding | None = None,
    algorithm: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for one CUTLASS reduction."""

    reduce_operator = None
    operation = ReduceOperation.SUM
    if op != "sum":
        try:
            cpp = _REDUCE_OPERATOR_CPP[op]
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported group reduce operator {op!r}"
            ) from exc
        operation = ReduceOperation.REDUCE
        reduce_operator = CxxOperator(
            cpp=cpp,
            dtype=Dependency("T"),
            name="binary_op",
        )
    if valid_items is None:
        valid_items = ArgumentBinding.omitted()
    primitive = make_reduce_semantics(
        dtype=dtype,
        operation=operation,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        reduce_operator=reduce_operator,
        valid_items=valid_items,
    )
    call = make_group_primitive_call(
        group,
        GroupReduceSemantics(
            primitive=primitive,
            broadcast=broadcast,
            cub_algorithm=algorithm,
        ),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _validate_valid_items_payload(
    binding: ArgumentBinding,
    value: Any,
) -> None:
    if binding.kind is BindingKind.OMITTED:
        if value is not None:
            raise ValueError("omitted valid_items binding cannot carry a value")
        return
    if binding.kind is BindingKind.RUNTIME:
        if value is None:
            raise ValueError("runtime valid_items binding requires a value")
        return
    if value != binding.value:
        raise ValueError("static valid_items value does not match its binding")


def _validate_reduce_request_plan(
    plan: GroupLoweringPlan,
    *,
    op: str,
    value_type: type,
) -> GroupReduceSemantics:
    operation = plan.call.operation
    if not isinstance(operation, GroupReduceSemantics):
        raise TypeError("group reduce request requires reduce semantics")
    if operation.dtype is not value_type:
        raise ValueError("group reduce request dtype does not match its plan")
    expected_operation = ReduceOperation.SUM if op == "sum" else ReduceOperation.REDUCE
    if operation.operation is not expected_operation:
        raise ValueError("group reduce request operator does not match its plan")
    if expected_operation is ReduceOperation.REDUCE:
        expected_cpp = _REDUCE_OPERATOR_CPP.get(op)
        if expected_cpp is None or operation.reduce_operator is None:
            raise ValueError("group reduce request operator does not match its plan")
        if operation.reduce_operator.cpp != expected_cpp:
            raise ValueError("group reduce request operator does not match its plan")
    return operation


@dataclasses.dataclass(frozen=True, eq=False)
class _CudaxReduceRequest:
    plan: GroupLoweringPlan
    op: str
    value_type: type
    kind: str = "cudax_reduce"

    def __post_init__(self) -> None:
        self.plan.require_supported()
        if self.plan.target is not GroupLoweringTarget.CUDAX_GROUP:
            raise ValueError("cudax reduce request requires a CUDAX_GROUP plan")
        if self.plan.resolved_group.kind == "grid":
            raise NotImplementedError(_GRID_REDUCE_UNAVAILABLE)
        if not isinstance(self.plan.implementation, CudaxCallDescription):
            raise TypeError("cudax reduce request requires a CUDAX call description")
        operation = _validate_reduce_request_plan(
            self.plan,
            op=self.op,
            value_type=self.value_type,
        )
        expected_overload = "broadcasted" if operation.broadcast else "root_only"
        expected_return = (
            CudaxReturnKind.VALUE
            if operation.broadcast
            else CudaxReturnKind.OPTIONAL_VALUE
        )
        if (
            self.plan.implementation.overload != expected_overload
            or self.plan.implementation.return_kind is not expected_return
        ):
            raise ValueError("cudax reduce request result mode does not match its plan")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        return self.plan.artifact_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CudaxReduceRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def group(self) -> ThreadGroup:
        return self.plan.resolved_group

    @property
    def items_per_thread(self) -> int:
        return self.plan.call.operation.items_per_thread

    @property
    def broadcast(self) -> bool:
        return self.plan.call.operation.broadcast

    @property
    def _arity_suffix(self) -> str:
        if self.plan.call.operation.primitive.value_kind is ReduceValueKind.ARRAY:
            return f"_x{self.items_per_thread}"
        return ""

    @property
    def _block_suffix(self) -> str:
        return self.group.symbol_suffix

    @property
    def _result_mode_suffix(self) -> str:
        return "" if self.broadcast else "_root"

    @property
    def symbol_name(self) -> str:
        signature = hashlib.sha256(
            repr(self.semantic_key).encode("utf-8", errors="backslashreplace")
        ).hexdigest()[:12]
        return (
            "cuda_coop_cutlass_cudax_reduce_"
            f"{self._block_suffix}_{self.op}_"
            f"{TYPE_SPECS[self.value_type].token}{self._arity_suffix}"
            f"{self._result_mode_suffix}_{signature}"
        )


def _storage_reuse_barrier_line(plan: GroupLoweringPlan) -> str:
    synchronization = plan.synchronization
    if synchronization is None:
        raise ValueError("Reduce plan requires a synchronization contract")
    if synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK:
        return "  __syncthreads();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.WARP:
        if plan.target is GroupLoweringTarget.CUB_WARP:
            _, logical_width = _warp_instances(plan)
        else:
            group = plan.resolved_group
            logical_width = 32 if group.kind == "warp" else group.static_size
            if group.kind not in {"warp", "threads_within_warp"} or not isinstance(
                logical_width,
                int,
            ):
                raise ValueError("Reduce plan requires a static warp group")
        mask = "0xffffffffu"
        if logical_width < 32:
            mask = f"{(1 << logical_width) - 1}u << ((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % 32u / {logical_width}u * {logical_width}u)"
        return f"  __syncwarp({mask});"
    if synchronization.storage_reuse_barrier is SynchronizationScope.GROUP:
        return "  group.sync_aligned();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.NONE:
        return ""
    raise ValueError("Reduce plan requires a storage reuse barrier")


def _render_group_prelude(group: ThreadGroup) -> list[str]:
    if group.hierarchy.implicit:
        assert group.mapping is None
        hierarchy = "::cuda::experimental::implicit_hierarchy()"
        return [
            f"  ::cuda::experimental::coop::this_{group.kind} group{{{hierarchy}}};"
        ]
    return [
        *render_hierarchy_decl(group.hierarchy),
        *render_group_decl_lines(group),
    ]


def _render_cudax_reduce(request: _CudaxReduceRequest) -> list[str]:
    if request.items_per_thread <= 0:
        raise ValueError("cudax reduce items_per_thread must be positive")
    implementation = request.plan.implementation
    assert isinstance(implementation, CudaxCallDescription)
    runtime_parameters = tuple(
        parameter.name
        for parameter in implementation.parameters
        if parameter.kind.value == "runtime"
    )
    expected_parameters = tuple(
        f"item{index}" for index in range(request.items_per_thread)
    )
    if runtime_parameters != expected_parameters:
        raise ValueError(
            "cudax reduce runtime ABI does not match the shared lowering plan"
        )

    spec = TYPE_SPECS[request.value_type]
    params = [f"{spec.cpp_type} item{idx}" for idx in range(request.items_per_thread)]
    values = ", ".join(f"item{idx}" for idx in range(request.items_per_thread))
    lines = [
        f"{spec.cpp_type} {request.symbol_name}({', '.join(params)}) {{",
        *_render_group_prelude(request.group),
        *(
            [
                "  if (!::cuda::gpu_thread.is_part_of(group)) {",
                f"    return {spec.cpp_type}{{}};",
                "  }",
            ]
            if request.group.mapping is not None
            and request.group.complete_membership is False
            else []
        ),
        f"  {spec.cpp_type} thread_data[{request.items_per_thread}] = {{{values}}};",
    ]
    barrier_line = _storage_reuse_barrier_line(request.plan)
    if request.broadcast:
        lines.extend(
            [
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                "      ::cuda::experimental::broadcasted, group, thread_data,",
                f"      {operator_expression(request.op)});",
                f"  {spec.cpp_type} result = reduced;",
                *([barrier_line] if barrier_line else []),
                "  return result;",
            ]
        )
    else:
        lines.extend(
            [
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                f"      group, thread_data, {operator_expression(request.op)});",
                f"  {spec.cpp_type} result = reduced.value_or({spec.cpp_type}{{}});",
                *([barrier_line] if barrier_line else []),
                "  return result;",
            ]
        )
    lines.append("}")
    return lines


_BLOCK_ALGORITHM_TOKENS = {
    BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY: "raking_commutative",
    BlockReduceAlgorithm.RAKING: "raking",
    BlockReduceAlgorithm.WARP_REDUCTIONS: "warp_reductions",
    BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC: "nondeterministic",
}


@dataclasses.dataclass(frozen=True, eq=False)
class _CubReduceRequest:
    plan: GroupLoweringPlan
    op: str
    value_type: type
    kind: str = "cub_group_reduce"

    def __post_init__(self) -> None:
        self.plan.require_supported()
        if self.plan.target not in {
            GroupLoweringTarget.CUB_BLOCK,
            GroupLoweringTarget.CUB_WARP,
        }:
            raise ValueError("CUB reduce request requires a CUB lowering plan")
        if not isinstance(self.plan.implementation, AlgorithmSpec):
            raise TypeError("CUB reduce request requires an AlgorithmSpec")
        _validate_reduce_request_plan(
            self.plan,
            op=self.op,
            value_type=self.value_type,
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        return self.plan.artifact_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CubReduceRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def group(self) -> ThreadGroup:
        return self.plan.resolved_group

    @property
    def operation(self) -> GroupReduceSemantics:
        operation = self.plan.call.operation
        assert isinstance(operation, GroupReduceSemantics)
        return operation

    @property
    def items_per_thread(self) -> int:
        return self.operation.items_per_thread

    @property
    def valid_items_suffix(self) -> str:
        binding = self.operation.valid_items
        if binding.kind is BindingKind.OMITTED:
            return "full"
        if binding.kind is BindingKind.RUNTIME:
            return "valid_r"
        return f"valid_s{binding.value}"

    @property
    def algorithm_suffix(self) -> str:
        if self.plan.target is GroupLoweringTarget.CUB_WARP:
            return "warp"
        algorithm = self.operation.cub_algorithm or BlockReduceAlgorithm.WARP_REDUCTIONS
        return _BLOCK_ALGORITHM_TOKENS[algorithm]

    @property
    def symbol_name(self) -> str:
        arity = (
            f"_x{self.items_per_thread}"
            if self.operation.primitive.value_kind is ReduceValueKind.ARRAY
            else ""
        )
        signature = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()[:12]
        return (
            "cuda_coop_cutlass_cub_reduce_"
            f"{self.group.symbol_suffix}_{self.op}_"
            f"{TYPE_SPECS[self.value_type].token}{arity}_"
            f"{self.algorithm_suffix}_{self.valid_items_suffix}_{signature}"
        )


def _render_cub_template_argument(
    request: _CubReduceRequest,
    name: str,
    value: Any,
) -> str:
    if name == "T":
        if value is not request.value_type:
            raise ValueError("CUB reduce template dtype does not match its request")
        return TYPE_SPECS[request.value_type].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render CUB reduce template argument {name}={value!r}")


def _warp_instances(plan: GroupLoweringPlan) -> tuple[int, int]:
    participation = plan.participation
    if participation is None:
        raise ValueError("WarpReduce plan requires a participation contract")
    block_dim = participation.exact_block_dim
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    implementation = plan.implementation
    assert isinstance(implementation, AlgorithmSpec)
    logical_width = implementation.template_arguments.get("VIRTUAL_WARP_THREADS")
    if not isinstance(logical_width, int) or logical_width < 1:
        raise ValueError("WarpReduce plan requires a static logical warp width")
    if block_threads < logical_width or block_threads % logical_width != 0:
        raise ValueError("WarpReduce plan requires complete logical warps")
    return block_threads // logical_width, logical_width


def _render_cub_reduce(request: _CubReduceRequest) -> list[str]:
    implementation = request.plan.implementation
    assert isinstance(implementation, AlgorithmSpec)
    spec = TYPE_SPECS[request.value_type]
    template_arguments = ", ".join(
        _render_cub_template_argument(request, name, value)
        for name, value in implementation.ordered_template_arguments
    )
    runtime_valid_items = request.operation.valid_items.kind is BindingKind.RUNTIME
    params = [
        *(f"{spec.cpp_type} item{index}" for index in range(request.items_per_thread)),
        *(["int valid_items"] if runtime_valid_items else []),
    ]
    input_name = "item0"
    input_lines: list[str] = []
    if request.operation.primitive.value_kind is ReduceValueKind.ARRAY:
        values = ", ".join(f"item{index}" for index in range(request.items_per_thread))
        input_name = "thread_data"
        input_lines.append(
            f"  {spec.cpp_type} thread_data[{request.items_per_thread}] = {{{values}}};"
        )

    call_arguments = [input_name]
    if request.operation.operation is ReduceOperation.REDUCE:
        call_arguments.append(operator_expression(request.op))
    if runtime_valid_items:
        call_arguments.append("valid_items")
    elif request.operation.valid_items.kind is BindingKind.STATIC:
        call_arguments.append(str(request.operation.valid_items.value))

    storage = "storage"
    storage_lines = ["  __shared__ typename implementation_type::TempStorage storage;"]
    if request.plan.target is GroupLoweringTarget.CUB_WARP:
        instances, logical_width = _warp_instances(request.plan)
        storage_lines = [
            "  __shared__ typename implementation_type::TempStorage "
            f"storage[{instances}];",
            "  unsigned int storage_instance =",
            f"      (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / {logical_width}u;",
        ]
        storage = "storage[storage_instance]"

    barrier_line = _storage_reuse_barrier_line(request.plan)

    return [
        f"{spec.cpp_type} {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = ::cub::{implementation.struct_name}<"
        f"{template_arguments}>;",
        *(
            [
                f"  if (valid_items < 1 || valid_items > {request.group.static_size}) {{",
                '    asm volatile("trap;" : : :);',
                "  }",
            ]
            if runtime_valid_items
            else []
        ),
        *storage_lines,
        *input_lines,
        f"  {spec.cpp_type} result = implementation_type({storage})."
        f"{implementation.method_name}({', '.join(call_arguments)});",
        barrier_line,
        "  return result;",
        "}",
    ]


def _register_renderer() -> None:
    _provider_rendering.register_bundle_renderer(
        "cudax_reduce",
        render=_render_cudax_reduce,
        include_lines=(
            "#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
            "#define _CUDAX_DISABLE_CG_INTEROP",
            "#include <cuda/barrier>",
            "#include <cuda/devices>",
            "#include <cuda/functional>",
            "#include <cuda/hierarchy>",
            "#include <cuda/std/functional>",
            "#include <cuda/std/type_traits>",
            "#include <cuda/experimental/coop/algorithm>",
            "#include <cuda/experimental/coop/group>",
        ),
        cccl_headers=(
            (
                "#include <cuda/experimental/coop/algorithm>",
                "cuda/experimental/coop/algorithm",
            ),
            ("#include <cuda/experimental/coop/group>", "cuda/experimental/coop/group"),
        ),
    )
    _provider_rendering.register_bundle_renderer(
        "cub_group_reduce",
        render=_render_cub_reduce,
        include_lines=(
            "#include <cuda/functional>",
            "#include <cuda/std/functional>",
            "#include <cub/block/block_reduce.cuh>",
            "#include <cub/warp/warp_reduce.cuh>",
        ),
        cccl_headers=(
            ("#include <cub/block/block_reduce.cuh>", "cub/block/block_reduce.cuh"),
            ("#include <cub/warp/warp_reduce.cuh>", "cub/warp/warp_reduce.cuh"),
        ),
    )


_register_renderer()

_resolve_type = _provider_types.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def provider_reduce(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value: Any,
    op: str = "sum",
    broadcast: bool = True,
    valid_items: Any = None,
    valid_items_binding: ArgumentBinding | None = None,
    algorithm: Any = None,
) -> Any:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.reduce group must be a ThreadGroup")
    if not isinstance(broadcast, bool):
        raise TypeError(f"{_ROOT_SCOPE}.reduce broadcast must be a bool")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("group reduce provider requires exact launch facts")
    if valid_items_binding is None:
        valid_items_binding = _classify_valid_items(valid_items)
    _validate_valid_items_payload(valid_items_binding, valid_items)

    def materialize(
        *,
        value_type: type,
        value_kind: ReduceValueKind,
        values: tuple[Any, ...],
    ) -> Any:
        plan = _make_group_reduce_plan(
            group=group,
            launch=launch,
            dtype=value_type,
            value_kind=value_kind,
            items_per_thread=len(values),
            op=op,
            broadcast=broadcast,
            valid_items=valid_items_binding,
            algorithm=algorithm,
            source="cutlass_group_reduce_provider",
        ).require_supported()
        if plan.target is GroupLoweringTarget.CUDAX_GROUP:
            request: _CudaxReduceRequest | _CubReduceRequest = _CudaxReduceRequest(
                plan=plan,
                op=op,
                value_type=value_type,
            )
        elif plan.target in {
            GroupLoweringTarget.CUB_BLOCK,
            GroupLoweringTarget.CUB_WARP,
        }:
            request = _CubReduceRequest(
                plan=plan,
                op=op,
                value_type=value_type,
            )
        else:
            raise AssertionError("supported group Reduce plan has no provider target")

        runtime_valid_items = valid_items_binding.kind is BindingKind.RUNTIME
        runtime_valid_args = (
            [_provider_types.as_valid_items_arg(valid_items, scope=_ROOT_SCOPE)]
            if runtime_valid_items
            else []
        )
        typed_values = []
        for item in values:
            converted = _provider_types.coerce_plain_scalar(
                item,
                value_type,
                name="reduce value",
                scope=_ROOT_SCOPE,
                allow_nonfinite=True,
            )
            typed_values.append(
                value_type(item)
                if converted is _provider_types._NOT_PLAIN_SCALAR
                else converted
            )
        snapshot = _provider_state.snapshot_active_session_state()
        try:
            _provider_state.register_request(request)
            result = ffi(
                name=request.symbol_name,
                params_types=[
                    *([value_type] * len(values)),
                    *([Int32] if runtime_valid_items else []),
                ],
                return_type=value_type,
            )(*typed_values, *runtime_valid_args)
            return value_type(result)
        except BaseException:
            _provider_state.restore_active_session_state(snapshot)
            raise

    if isinstance(value, ThreadData):
        value_type, values = _provider_types.resolve_thread_data_value_type(
            value,
            allowed=ALL_PROVIDER_TYPES,
            feature="reduce",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        validate_operator_dtype(op, value_type)
        return materialize(
            value_type=value_type,
            value_kind=ReduceValueKind.ARRAY,
            values=tuple(values),
        )

    value_type = _resolve_type(
        value,
        allowed=ALL_PROVIDER_TYPES,
        feature="reduce",
    )
    validate_operator_dtype(op, value_type)
    return materialize(
        value_type=value_type,
        value_kind=ReduceValueKind.SCALAR,
        values=(value,),
    )


__all__ = [
    "_CudaxReduceRequest",
    "_CubReduceRequest",
    "provider_reduce",
]
