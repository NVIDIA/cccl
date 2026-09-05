# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven CUDAX/CUB provider renderer for CUTLASS group reductions."""

from __future__ import annotations

import dataclasses
from typing import Any

from cutlass.base_dsl.typing import Int32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    BindingKind,
    CudaxCallDescription,
    CudaxReturnKind,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupReduceSemantics,
    LaunchFacts,
    ReduceOperation,
    ReduceValueKind,
    SynchronizationScope,
)
from cuda.coop._core.block import BlockReduceAlgorithm

from .. import _group_reduce as _group_frontend
from .._value_metadata import metadata_for_group, validate_operand_domains
from . import _provider as _provider_support
from ._provider import SCAN_REDUCE_TYPES as _SCAN_REDUCE_TYPES
from ._provider import TYPE_SPECS as _TYPE_SPECS
from ._thread_data import ThreadData
from ._thread_group import (
    ThreadGroup,
    render_group_decl_lines,
    render_hierarchy_decl,
)

_ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
_GRID_REDUCE_UNAVAILABLE = (
    f"{_ROOT_SCOPE}.reduce grid lowering is blocked until the CUTLASS DSL "
    "provides a reviewed compiler-managed device workspace contract"
)


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
        expected_cpp = _group_frontend._REDUCE_OPERATOR_CPP.get(op)
        if expected_cpp is None or operation.reduce_operator is None:
            raise ValueError("group reduce request operator does not match its plan")
        if operation.reduce_operator.cpp != expected_cpp:
            raise ValueError("group reduce request operator does not match its plan")
    return operation


@dataclasses.dataclass(frozen=True, eq=False, init=False)
class _CudaxReduceRequest:
    plan: GroupLoweringPlan
    op: str
    value_type: type
    kind: str = "cudax_reduce"

    def __init__(
        self,
        *,
        value_type: type,
        op: str,
        plan: GroupLoweringPlan | None = None,
        group: ThreadGroup | None = None,
        items_per_thread: int = 1,
        broadcast: bool = True,
    ) -> None:
        """Create a request from a plan, with legacy construction for tests."""

        if plan is None:
            if group is None or group.hierarchy is None:
                raise TypeError("cudax reduce request requires a plan or static group")
            if group.hierarchy.block_dim is None:
                raise ValueError("cudax reduce request group requires block dimensions")
            plan = _group_frontend._make_group_reduce_plan(
                group=group,
                launch=LaunchFacts(exact_block_dim=group.hierarchy.block_dim),
                dtype=value_type,
                value_kind=(
                    ReduceValueKind.ARRAY
                    if items_per_thread > 1
                    else ReduceValueKind.SCALAR
                ),
                items_per_thread=items_per_thread,
                op=op,
                broadcast=broadcast,
                source="legacy_request_constructor",
            )
        plan.require_supported()
        if plan.target is not GroupLoweringTarget.CUDAX_GROUP:
            raise ValueError("cudax reduce request requires a CUDAX_GROUP plan")
        if plan.resolved_group.kind == "grid":
            raise NotImplementedError(_GRID_REDUCE_UNAVAILABLE)
        if not isinstance(plan.implementation, CudaxCallDescription):
            raise TypeError("cudax reduce request requires a CUDAX call description")
        operation = _validate_reduce_request_plan(
            plan,
            op=op,
            value_type=value_type,
        )
        expected_overload = "broadcasted" if operation.broadcast else "root_only"
        expected_return = (
            CudaxReturnKind.VALUE
            if operation.broadcast
            else CudaxReturnKind.OPTIONAL_VALUE
        )
        if (
            plan.implementation.overload != expected_overload
            or plan.implementation.return_kind is not expected_return
        ):
            raise ValueError("cudax reduce request result mode does not match its plan")
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "kind", "cudax_reduce")

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
        return (
            "cuda_coop_cutlass_cudax_reduce_"
            f"{self._block_suffix}_{self.op}_"
            f"{_TYPE_SPECS[self.value_type].token}{self._arity_suffix}"
            f"{self._result_mode_suffix}"
        )


def _storage_reuse_barrier_line(plan: GroupLoweringPlan) -> str:
    synchronization = plan.synchronization
    if synchronization is None:
        raise ValueError("Reduce plan requires a synchronization contract")
    if synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK:
        return "  cuda_coop_cutlass_block_sync();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.WARP:
        return "  cuda_coop_cutlass_warp_sync();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.GROUP:
        return "  group.sync_aligned();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.NONE:
        return ""
    raise ValueError("Reduce plan requires a storage reuse barrier")


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

    spec = _TYPE_SPECS[request.value_type]
    params = [f"{spec.cpp_type} item{idx}" for idx in range(request.items_per_thread)]
    values = ", ".join(f"item{idx}" for idx in range(request.items_per_thread))
    lines = [
        f"{spec.cpp_type} {request.symbol_name}({', '.join(params)}) {{",
        *render_hierarchy_decl(request.group.hierarchy),
        *render_group_decl_lines(request.group),
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
                f"      {_provider_support.cub_op_expr(request.op)});",
                f"  {spec.cpp_type} result = reduced;",
                *([barrier_line] if barrier_line else []),
                "  return result;",
            ]
        )
    else:
        lines.extend(
            [
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                "      group, thread_data, "
                f"{_provider_support.cub_op_expr(request.op)});",
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
        return (
            "cuda_coop_cutlass_cub_reduce_"
            f"{self.group.kind}_{self.group.block_dim_token}_{self.op}_"
            f"{_TYPE_SPECS[self.value_type].token}{arity}_"
            f"{self.algorithm_suffix}_{self.valid_items_suffix}"
        )


def _render_cub_template_argument(
    request: _CubReduceRequest,
    name: str,
    value: Any,
) -> str:
    if name == "T":
        if value is not request.value_type:
            raise ValueError("CUB reduce template dtype does not match its request")
        return _TYPE_SPECS[request.value_type].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render CUB reduce template argument {name}={value!r}")


def _render_cub_reduce(request: _CubReduceRequest) -> list[str]:
    implementation = request.plan.implementation
    assert isinstance(implementation, AlgorithmSpec)
    spec = _TYPE_SPECS[request.value_type]
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
        call_arguments.append(_provider_support.cub_op_expr(request.op))
    if runtime_valid_items:
        call_arguments.append("valid_items")
    elif request.operation.valid_items.kind is BindingKind.STATIC:
        call_arguments.append(str(request.operation.valid_items.value))

    storage = "storage"
    storage_lines = ["  __shared__ typename implementation_type::TempStorage storage;"]
    if request.plan.target is GroupLoweringTarget.CUB_WARP:
        assert request.plan.temp_storage is not None
        instances = request.plan.temp_storage.instances
        if not isinstance(instances, int) or instances < 1:
            raise ValueError("CUB WarpReduce plan requires static storage instances")
        logical_width = implementation.template_arguments.get("VIRTUAL_WARP_THREADS")
        if not isinstance(logical_width, int) or logical_width < 1:
            raise ValueError("CUB WarpReduce plan requires a static logical width")
        storage_lines = [
            "  __shared__ typename implementation_type::TempStorage "
            f"storage[{instances}];",
            "  unsigned int storage_instance =",
            f"      cuda_coop_cutlass_linear_tid() / {logical_width}u;",
        ]
        storage = "storage[storage_instance]"

    barrier_line = _storage_reuse_barrier_line(request.plan)

    return [
        f"{spec.cpp_type} {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = ::cub::{implementation.struct_name}<"
        f"{template_arguments}>;",
        *storage_lines,
        *input_lines,
        f"  {spec.cpp_type} result = implementation_type({storage})."
        f"{implementation.method_name}({', '.join(call_arguments)});",
        barrier_line,
        "  return result;",
        "}",
    ]


def _register_renderer() -> None:
    _provider_support.register_bundle_renderer(
        "cudax_reduce",
        render=_render_cudax_reduce,
        include_lines=(
            "#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
            "#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP",
            "#include <cuda/barrier>",
            "#include <cuda/devices>",
            "#include <cuda/functional>",
            "#include <cuda/hierarchy>",
            "#include <cuda/std/functional>",
            "#include <cuda/std/type_traits>",
            "#include <cuda/experimental/coop.cuh>",
            "#include <cuda/experimental/group.cuh>",
        ),
        cccl_headers=(
            ("#include <cuda/experimental/coop.cuh>", "cuda/experimental/coop.cuh"),
            ("#include <cuda/experimental/group.cuh>", "cuda/experimental/group.cuh"),
        ),
    )
    _provider_support.register_bundle_renderer(
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

_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _validate_op_for_type(op: str, value_type: type) -> None:
    _provider_support.validate_scan_reduce_op_for_type(
        op,
        value_type,
        root_scope=_ROOT_SCOPE,
        feature="reduce",
        namespace="thread_group",
    )


def _register_request(request: _CudaxReduceRequest | _CubReduceRequest) -> None:
    _provider_support.register_request(request)


def _remember_scalar_result_type(
    value: Any,
    value_type: type,
    *,
    plan: GroupLoweringPlan,
) -> Any:
    assert plan.result is not None
    return _provider_support.remember_scalar_result_type(
        value,
        value_type,
        scope=_ROOT_SCOPE,
        compile_options_getter=lambda: (
            _provider_support._get_cute_dsl().compile_options
        ),
        group_metadata=metadata_for_group(
            plan.resolved_group,
            visibility=plan.result.visibility,
        ),
    )


def provider_reduce(
    *,
    group: ThreadGroup,
    launch: LaunchFacts | None = None,
    value: Any,
    op: str = "sum",
    broadcast: bool = True,
    valid_items: Any = None,
    valid_items_binding: ArgumentBinding | None = None,
    algorithm: Any = None,
    source: str = "cutlass_group_reduce_provider",
) -> Any:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.reduce group must be a ThreadGroup")
    if not isinstance(broadcast, bool):
        raise TypeError(f"{_ROOT_SCOPE}.reduce broadcast must be a bool")
    if launch is None:
        if group.hierarchy is None or group.hierarchy.block_dim is None:
            raise TypeError("group reduce provider requires exact launch facts")
        launch = LaunchFacts(exact_block_dim=group.hierarchy.block_dim)
    if valid_items_binding is None:
        valid_items_binding = _group_frontend._classify_valid_items(valid_items)
    _validate_valid_items_payload(valid_items_binding, valid_items)

    def materialize(
        *,
        value_type: type,
        value_kind: ReduceValueKind,
        values: tuple[Any, ...],
    ) -> Any:
        plan = _group_frontend._make_group_reduce_plan(
            group=group,
            launch=launch,
            dtype=value_type,
            value_kind=value_kind,
            items_per_thread=len(values),
            op=op,
            broadcast=broadcast,
            valid_items=valid_items_binding,
            algorithm=algorithm,
            source=source,
        ).require_supported()
        validate_operand_domains(
            plan.resolved_group,
            {"value": value},
            scope=_ROOT_SCOPE,
            primitive_name="reduce",
        )
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
            [_provider_support.as_valid_items_arg(valid_items, scope=_ROOT_SCOPE)]
            if runtime_valid_items
            else []
        )
        _register_request(request)
        result = ffi(
            name=request.symbol_name,
            params_types=[
                *([value_type] * len(values)),
                *([Int32] if runtime_valid_items else []),
            ],
            return_type=value_type,
        )(
            *values,
            *runtime_valid_args,
        )
        return _remember_scalar_result_type(result, value_type, plan=plan)

    if isinstance(value, ThreadData):
        value_type, values = _provider_support.resolve_thread_data_value_type(
            value,
            allowed=_SCAN_REDUCE_TYPES,
            feature="reduce",
            scope=_ROOT_SCOPE,
            resolve_type=_resolve_type,
        )
        _validate_op_for_type(op, value_type)
        return materialize(
            value_type=value_type,
            value_kind=ReduceValueKind.ARRAY,
            values=tuple(values),
        )

    value_type = _resolve_type(
        value,
        allowed=_SCAN_REDUCE_TYPES,
        feature="reduce",
    )
    _validate_op_for_type(op, value_type)
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
