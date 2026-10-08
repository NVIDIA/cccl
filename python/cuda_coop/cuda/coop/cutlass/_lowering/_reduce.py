# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Emit CuTe extern calls for CUB reduction plans.

Requests bind generated definitions to plan identities and extern signatures.
Results are defined only at group rank zero. Finalization compiles the
queued C++ definitions and resolves their shared-memory layouts before
materializing the deferred scratch operands.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from cutlass.base_dsl.typing import Int32, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    Algorithm,
    ArgumentBinding,
    BindingKind,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupReduceSemantics,
    LaunchFacts,
    ReduceOperation,
    ReduceValueKind,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    make_reduce_semantics,
    plan_group_primitive,
)
from cuda.coop._core.block import BlockReduceAlgorithm
from cuda.coop._core.thread_group import ThreadGroup

from .._compiler import _rendering, _state, _types
from .._compiler._types import ALL_PROVIDER_TYPES, TYPE_SPECIFICATIONS
from .._group_reduce import _classify_valid_items
from .._operators import (
    OPERATOR_CPP,
    operator_expression,
    validate_operator_dtype,
)
from .._thread_data import ThreadData

_provider_rendering = _rendering
_provider_state = _state
_provider_types = _types

_ROOT_SCOPE = "cuda.coop.cutlass"
_REDUCE_OPERATOR_CPP = OPERATOR_CPP


def _make_group_reduce_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    value_kind: ReduceValueKind,
    items_per_thread: int,
    op: str,
    valid_items: ArgumentBinding | None = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> GroupLoweringPlan:
    """Build shared reduction semantics and select a lowering route.

    Represent built-in non-sum operators as C++ functors with a payload dtype
    dependency. Shared planning decides result visibility, participation,
    scratch, and specialization; this helper does not emit a device call.
    """

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
            cub_algorithm=algorithm,
            storage_ownership=(
                StorageOwnership.IMPLEMENTATION
                if temp_storage is None
                else StorageOwnership.CALLER
            ),
            storage_sharing=None
            if temp_storage is None
            else temp_storage.sharing,
            storage_size_in_bytes=None
            if temp_storage is None
            else temp_storage.size_in_bytes,
            storage_alignment=None
            if temp_storage is None
            else temp_storage.alignment,
            storage_auto_sync=True
            if temp_storage is None
            else temp_storage.auto_sync,
        ),
    )
    return plan_group_primitive(call, launch)


def _validate_valid_items_payload(
    binding: ArgumentBinding,
    value: Any,
) -> None:
    """Check that a count value agrees with its binding description.

    An omitted binding carries no value, a runtime binding needs an operand,
    and a static binding must match its recorded constant.
    """

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
    """Match a wrapper request to its planned dtype and operator.

    Check the C++ functor spelling as well as sum versus general Reduce so the
    rendered operation cannot differ from the artifact identity.
    """

    operation = plan.call.operation
    if not isinstance(operation, GroupReduceSemantics):
        raise TypeError("group reduce request requires reduce semantics")
    if operation.dtype is not value_type:
        raise ValueError("group reduce request dtype does not match its plan")
    expected_operation = (
        ReduceOperation.SUM if op == "sum" else ReduceOperation.REDUCE
    )
    if operation.operation is not expected_operation:
        raise ValueError(
            "group reduce request operator does not match its plan"
        )
    if expected_operation is ReduceOperation.REDUCE:
        expected_cpp = _REDUCE_OPERATOR_CPP.get(op)
        if expected_cpp is None or operation.reduce_operator is None:
            raise ValueError(
                "group reduce request operator does not match its plan"
            )
        if operation.reduce_operator.cpp != expected_cpp:
            raise ValueError(
                "group reduce request operator does not match its plan"
            )
    return operation


def _storage_reuse_barrier_line(plan: GroupLoweringPlan) -> str:
    """Render a barrier before wrapper-owned scratch is reused, or no line.

    CUB plans get a block barrier or a warp barrier. The warp mask covers only
    the logical group's lanes. NONE means the caller selected manual
    synchronization for explicit block scratch.
    """

    synchronization = plan.synchronization
    if synchronization is None:
        raise ValueError("Reduce plan requires a synchronization contract")
    if synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK:
        return "  __syncthreads();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.WARP:
        _, logical_width = _warp_instances(plan)
        mask = "0xffffffffu"
        if logical_width < 32:
            mask = (
                f"{(1 << logical_width) - 1}u << ((threadIdx.x + blockDim.x * "
                "(threadIdx.y + blockDim.y * threadIdx.z)) % 32u / "
                f"{logical_width}u * {logical_width}u)"
            )
        return f"  __syncwarp({mask});"
    if synchronization.storage_reuse_barrier is SynchronizationScope.NONE:
        return ""
    raise ValueError("Reduce plan requires a storage reuse barrier")


_BLOCK_ALGORITHM_TOKENS = {
    BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY: "raking_commutative",
    BlockReduceAlgorithm.RAKING: "raking",
    BlockReduceAlgorithm.WARP_REDUCTIONS: "warp_reductions",
    BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC: "nondeterministic",
}


@dataclasses.dataclass(frozen=True, eq=False)
class _CubReduceRequest:
    """Describe a direct CUB reduction with a shared specialization.

    The plan artifact key includes controls and topology needed by the
    wrapper. Scalar and one-item array forms stay distinct: the array form
    passes a local one-element array to CUB and adds an ``_x1`` symbol suffix.
    """

    plan: GroupLoweringPlan
    op: str
    value_type: type
    kind: str = "cub_group_reduce"

    def __post_init__(self) -> None:
        """Require a CUB plan with a matching dtype and built-in operator."""

        self.plan.require_supported()
        if self.plan.target not in {
            GroupLoweringTarget.CUB_BLOCK,
            GroupLoweringTarget.CUB_WARP,
        }:
            raise ValueError("CUB reduce request requires a CUB lowering plan")
        if not isinstance(self.plan.implementation, Algorithm):
            raise TypeError("CUB reduce request requires an Algorithm")
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
        """Name the selected block strategy or the warp provider route."""

        if self.plan.target is GroupLoweringTarget.CUB_WARP:
            return "warp"
        algorithm = (
            self.operation.cub_algorithm or BlockReduceAlgorithm.WARP_REDUCTIONS
        )
        return _BLOCK_ALGORITHM_TOKENS[algorithm]

    @property
    def group_instances(self) -> int:
        return (
            _warp_instances(self.plan)[0]
            if self.plan.target is GroupLoweringTarget.CUB_WARP
            else 1
        )

    @property
    def cpp_type(self) -> str:
        implementation = self.plan.implementation
        assert isinstance(implementation, Algorithm)
        arguments = ", ".join(
            _render_cub_template_argument(self, name, value)
            for name, value in implementation.ordered_template_arguments
        )
        return f"::cub::{implementation.struct_name}<{arguments}>"

    @property
    def scratch_requirement_key(self) -> tuple[Any, ...]:
        implementation = self.plan.implementation
        assert isinstance(implementation, Algorithm)
        return (
            "cub_reduce_layout",
            implementation.semantic_key,
            self.group_instances,
        )

    @property
    def symbol_name(self) -> str:
        """Name the CUB call and include a short artifact-key hash."""

        arity = (
            f"_x{self.items_per_thread}"
            if self.operation.primitive.value_kind is ReduceValueKind.ARRAY
            else ""
        )
        signature = hashlib.sha256(
            repr(self.semantic_key).encode()
        ).hexdigest()[:12]
        return (
            "cuda_coop_cutlass_cub_reduce_"
            f"{self.group.symbol_suffix}_{self.op}_"
            f"{TYPE_SPECIFICATIONS[self.value_type].token}{arity}_"
            f"{self.algorithm_suffix}_{self.valid_items_suffix}_{signature}"
        )


def _render_cub_template_argument(
    request: _CubReduceRequest,
    name: str,
    value: Any,
) -> str:
    """Spell a bound CUB argument and verify the payload dtype."""

    if name == "T":
        if value is not request.value_type:
            raise ValueError(
                "CUB reduce template dtype does not match its request"
            )
        return TYPE_SPECIFICATIONS[request.value_type].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(
        f"cannot render CUB reduce template argument {name}={value!r}"
    )


def _warp_instances(plan: GroupLoweringPlan) -> tuple[int, int]:
    """Find complete logical warp instances and their width.

    Use the exact enclosing block and the bound CUB warp width to size
    storage. Each instance needs its own scratch slice.
    """

    participation = plan.participation
    if participation is None:
        raise ValueError("WarpReduce plan requires a participation contract")
    block_dim = participation.exact_block_dim
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    implementation = plan.implementation
    assert isinstance(implementation, Algorithm)
    logical_width = implementation.template_arguments.get(
        "VIRTUAL_WARP_THREADS"
    )
    if not isinstance(logical_width, int) or logical_width < 1:
        raise ValueError("WarpReduce plan requires a static logical warp width")
    if block_threads < 32 or block_threads % 32 != 0:
        raise ValueError("WarpReduce plan requires complete physical warps")
    return (block_threads // 32) * (32 // logical_width), logical_width


def _render_cub_reduce(request: _CubReduceRequest) -> list[str]:
    """Render a CUB call with deferred scratch and root-only results.

    Item scalars come before an optional runtime count in the extern ABI. If
    a runtime count is outside 1 through the group size, the wrapper traps;
    the planner already checked static counts. Select one scratch object per
    block or logical warp from the deferred region, and emit the planned
    barrier before that scratch can be reused.
    """

    implementation = request.plan.implementation
    assert isinstance(implementation, Algorithm)
    type_specification = TYPE_SPECIFICATIONS[request.value_type]
    template_arguments = ", ".join(
        _render_cub_template_argument(request, name, value)
        for name, value in implementation.ordered_template_arguments
    )
    runtime_valid_items = (
        request.operation.valid_items.kind is BindingKind.RUNTIME
    )
    params = [
        *(
            f"{type_specification.cpp_type} item{index}"
            for index in range(request.items_per_thread)
        ),
        *(["int valid_items"] if runtime_valid_items else []),
        "unsigned int temp_storage_smem_addr",
        "int temp_storage_bytes",
        "int temp_storage_auto_sync",
    ]
    input_name = "item0"
    input_lines: list[str] = []
    if request.operation.primitive.value_kind is ReduceValueKind.ARRAY:
        values = ", ".join(
            f"item{index}" for index in range(request.items_per_thread)
        )
        input_name = "thread_data"
        input_lines.append(
            f"  {type_specification.cpp_type} "
            f"thread_data[{request.items_per_thread}] "
            f"= {{{values}}};"
        )

    call_arguments = [input_name]
    if request.operation.operation is ReduceOperation.REDUCE:
        call_arguments.append(operator_expression(request.op))
    if runtime_valid_items:
        call_arguments.append("valid_items")
    elif request.operation.valid_items.kind is BindingKind.STATIC:
        call_arguments.append(str(request.operation.valid_items.value))

    storage_lines = [
        "  using storage_type = typename implementation_type::TempStorage;",
        (
            f"  if (temp_storage_bytes < {request.group_instances}u "
            "* sizeof(storage_type) ||"
        ),
        "      (temp_storage_smem_addr & (alignof(storage_type) - 1)) != 0) {",
        '    asm volatile("trap;");',
        "  }",
        "  unsigned long long generic_addr;",
        (
            '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_addr) : '
            '"l"(static_cast<unsigned long long>(temp_storage_smem_addr)));'
        ),
    ]
    instance = "0"
    if request.plan.target is GroupLoweringTarget.CUB_WARP:
        _, logical_width = _warp_instances(request.plan)
        storage_lines.extend(
            [
                "  unsigned int thread_rank = threadIdx.x + blockDim.x *",
                "      (threadIdx.y + blockDim.y * threadIdx.z);",
                "  unsigned int storage_instance =",
                f"      (thread_rank / 32u) * {32 // logical_width}u +",
                f"      (thread_rank % 32u) / {logical_width}u;",
            ]
        )
        instance = "storage_instance"
    storage_lines.append(
        "  auto& storage = reinterpret_cast<storage_type*>(generic_addr)"
        f"[{instance}];"
    )

    barrier_line = _storage_reuse_barrier_line(request.plan)

    return [
        (
            f"{type_specification.cpp_type} "
            f"{request.symbol_name}({', '.join(params)}) {{"
        ),
        (
            "  using implementation_type = "
            f"::cub::{implementation.struct_name}<"
            f"{template_arguments}>;"
        ),
        *(
            [
                (
                    "  if (valid_items < 1 || valid_items > "
                    f"{request.group.static_size}) {{"
                ),
                '    asm volatile("trap;" : : :);',
                "  }",
            ]
            if runtime_valid_items
            else []
        ),
        *storage_lines,
        *input_lines,
        (
            f"  {type_specification.cpp_type} result "
            f"= implementation_type(storage)."
            f"{implementation.method_name}({', '.join(call_arguments)});"
        ),
        *(
            [f"  if (temp_storage_auto_sync != 0) {{ {barrier_line.strip()} }}"]
            if barrier_line
            else []
        ),
        "  return result;",
        "}",
    ]


def _scratch_layout_probe(request: _CubReduceRequest):
    """Resolve CUB size and alignment for every independent group instance."""

    return _provider_rendering.make_scratch_layout_probe(
        request.scratch_requirement_key,
        f"typename {request.cpp_type}::TempStorage[{request.group_instances}]",
    )


def _scratch_arguments(request: _CubReduceRequest, temp_storage):
    """Record this call's scratch use and return deferred ABI operands."""

    from .._compiler._storage import register_deferred_temp_storage_event
    from .._temp_storage import TempStorage

    if temp_storage is None:
        temp_storage = TempStorage(auto_sync=True)
    elif not isinstance(temp_storage, TempStorage):
        raise TypeError(
            "cuda.coop.cutlass Reduce scratch must be CUTLASS TempStorage"
        )
    arguments = register_deferred_temp_storage_event(
        temp_storage,
        primitive_name="reduce",
        requirement_key=request.scratch_requirement_key,
    )
    return (Uint32, Int32, Int32), arguments


def _register_renderer() -> None:
    """Register CUB reductions and their deferred scratch layouts."""

    _provider_rendering.register_bundle_renderer(
        "cub_group_reduce",
        render=_render_cub_reduce,
        scratch_layout_probe=_scratch_layout_probe,
        include_lines=(
            "#include <cuda/functional>",
            "#include <cuda/std/functional>",
            "#include <cub/block/block_reduce.cuh>",
            "#include <cub/warp/warp_reduce.cuh>",
        ),
        cccl_headers=(
            (
                "#include <cub/block/block_reduce.cuh>",
                "cub/block/block_reduce.cuh",
            ),
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
    valid_items: Any = None,
    valid_items_binding: ArgumentBinding | None = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Plan a reduction and emit its typed CuTe extern call.

    Resolve scalar or initialized ThreadData values to one dtype and validate
    the built-in operator. Each payload element becomes a scalar ABI argument;
    a runtime valid count follows those items. Inputs remain unchanged, and
    the returned scalar follows the plan's result visibility.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.reduce group must be a ThreadGroup")
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
        """Bind one operand shape and emit its planned reduction.

        Convert literals and runtime counts before registration. On
        failure, restore queued requests; this bookkeeping rollback does
        not remove emitted IR.
        """

        plan = _make_group_reduce_plan(
            group=group,
            launch=launch,
            dtype=value_type,
            value_kind=value_kind,
            items_per_thread=len(values),
            op=op,
            valid_items=valid_items_binding,
            algorithm=algorithm,
            temp_storage=temp_storage,
        ).require_supported()
        request = _CubReduceRequest(plan=plan, op=op, value_type=value_type)

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
            scratch_types, scratch_args = _scratch_arguments(
                request, temp_storage
            )
            result = ffi(
                name=request.symbol_name,
                params_types=[
                    *([value_type] * len(values)),
                    *([Int32] if runtime_valid_items else []),
                    *scratch_types,
                ],
                return_type=value_type,
            )(*typed_values, *runtime_valid_args, *scratch_args)
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
    "_CubReduceRequest",
    "provider_reduce",
]
