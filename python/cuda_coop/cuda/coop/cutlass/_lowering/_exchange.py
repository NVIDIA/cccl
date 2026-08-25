# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider renderer for CUTLASS Exchange."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from cutlass import cute as _cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    GroupExchangeSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import BlockExchangeMode, make_block_exchange_semantics

from .._compiler import _rendering as _provider_rendering
from .._compiler import _state as _provider_state
from .._compiler import _types as _provider_types
from .._compiler._types import ALL_PROVIDER_TYPES, RADIX_KEY_TYPES, TYPE_SPECS
from .._thread_data import ThreadData
from .._thread_group import ThreadGroup
from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)
from ._symbols import block_dim_token as _block_dim_token

_ROOT_SCOPE = "cuda.coop.cutlass"
_ROOT_MODES = frozenset(BlockExchangeMode)
_RANK_TYPES = RADIX_KEY_TYPES
_SIGNED_RANK_TYPES = frozenset({Int32, Int64})
_VALID_FLAG_TYPES = RADIX_KEY_TYPES


def _normalize_exchange_mode(mode: Any) -> str:
    try:
        return BlockExchangeMode(mode).value
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in BlockExchangeMode)
        raise ValueError(
            f"{_ROOT_SCOPE}.exchange mode must be one of: {choices}"
        ) from exc


def _make_group_exchange_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    items_per_thread: int,
    mode: str,
    rank_dtype: Any = None,
    valid_flag_dtype: Any = None,
    warp_time_slicing: bool = False,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core exchange plan."""

    primitive = make_block_exchange_semantics(
        dtype=dtype,
        items_per_thread=items_per_thread,
        mode=_normalize_exchange_mode(mode),
        value_form="out_of_place",
        warp_time_slicing=warp_time_slicing,
        rank_dtype=rank_dtype,
        valid_flag_dtype=valid_flag_dtype,
    )
    call = make_group_primitive_call(
        group,
        GroupExchangeSemantics(primitive),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _render_template_argument(
    request: _CubExchangeRequest,
    name: str,
    value: Any,
) -> str:
    if name == "T":
        if value is not request.value_type:
            raise ValueError("CUB Exchange template dtype does not match its request")
        return TYPE_SPECS[value].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render CUB Exchange template argument {name}={value!r}")


def _storage_reuse_barrier_line(request: _CubExchangeRequest) -> str:
    synchronization = request.plan.synchronization
    if synchronization is None:
        raise ValueError("Exchange plan requires a synchronization contract")
    if synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK:
        return "  cuda_coop_cutlass_block_sync();"
    if synchronization.storage_reuse_barrier is SynchronizationScope.WARP:
        return "  cuda_coop_cutlass_warp_sync();"
    raise ValueError("Exchange plan requires a storage reuse barrier")


def _validate_planned_exchange(
    plan: GroupLoweringPlan,
    *,
    value_type: type,
    rank_type: type | None,
    valid_flag_type: type | None,
) -> None:
    plan.require_supported()
    if plan.target not in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }:
        raise ValueError("CUB Exchange request requires a CUB lowering plan")
    if not isinstance(plan.implementation, AlgorithmSpec):
        raise TypeError("CUB Exchange request requires an AlgorithmSpec")
    operation = plan.call.operation
    if not isinstance(operation, GroupExchangeSemantics):
        raise TypeError("group Exchange request requires Exchange semantics")
    if operation.dtype is not value_type:
        raise ValueError("group Exchange dtype does not match its request")
    primitive = operation.primitive
    if primitive.mode not in _ROOT_MODES:
        raise ValueError("group Exchange plan uses an unsupported mode")
    if primitive.rank_dtype is not rank_type:
        raise ValueError("group Exchange rank dtype does not match its request")
    if primitive.valid_flag_dtype is not valid_flag_type:
        raise ValueError("group Exchange flag dtype does not match its request")

    implementation = plan.implementation
    expected_struct = (
        "BlockExchange"
        if plan.target is GroupLoweringTarget.CUB_BLOCK
        else "WarpExchange"
    )
    if implementation.struct_name != expected_struct:
        raise ValueError("CUB Exchange implementation does not match its group")
    if implementation.method_name != primitive.method_name:
        raise ValueError("CUB Exchange method does not match its plan")
    template_arguments = implementation.template_arguments
    if template_arguments.get("T") is not value_type:
        raise ValueError("CUB Exchange template dtype does not match its request")
    if template_arguments.get("ITEMS_PER_THREAD") != primitive.items_per_thread:
        raise ValueError("CUB Exchange item count does not match its request")

    participation = plan.participation
    if participation is None:
        raise ValueError("CUB Exchange plan requires a participation contract")
    block_dim = participation.exact_block_dim
    if plan.target is GroupLoweringTarget.CUB_BLOCK:
        expected_dims = (
            template_arguments.get("BLOCK_DIM_X"),
            template_arguments.get("BLOCK_DIM_Y"),
            template_arguments.get("BLOCK_DIM_Z"),
        )
        if expected_dims != block_dim:
            raise ValueError("CUB BlockExchange dimensions do not match its plan")
        if template_arguments.get("WARP_TIME_SLICING") != int(
            primitive.warp_time_slicing
        ):
            raise ValueError("group BlockExchange time slicing does not match its plan")
    else:
        if template_arguments.get("LOGICAL_WARP_THREADS") != (
            plan.resolved_group.static_size
        ):
            raise ValueError("group WarpExchange width does not match its plan")
        if template_arguments.get("WARP_EXCHANGE_ALGORITHM") != (
            "::cub::WARP_EXCHANGE_SMEM"
        ):
            raise ValueError("group WarpExchange requires the CUB SMEM algorithm")

    temp_storage = plan.temp_storage
    if temp_storage is None:
        raise ValueError("CUB Exchange plan requires a temporary-storage contract")
    if temp_storage.ownership is not StorageOwnership.IMPLEMENTATION:
        raise ValueError("CUB Exchange temporary storage must be implementation-owned")
    result = plan.result
    if result is None or len(result.values) != 1:
        raise ValueError("CUB Exchange plan requires one logical result")
    if result.values[0].items_per_member != primitive.items_per_thread:
        raise ValueError("CUB Exchange result item count does not match its request")


@dataclasses.dataclass(frozen=True, eq=False)
class _CubExchangeRequest:
    plan: GroupLoweringPlan
    value_type: type
    rank_type: type | None = None
    valid_flag_type: type | None = None
    kind: str = "cub_group_exchange"

    def __post_init__(self) -> None:
        if not isinstance(self.plan, GroupLoweringPlan):
            raise TypeError("CUB Exchange request requires a GroupLoweringPlan")
        _validate_planned_exchange(
            self.plan,
            value_type=self.value_type,
            rank_type=self.rank_type,
            valid_flag_type=self.valid_flag_type,
        )

    @property
    def implementation(self) -> AlgorithmSpec:
        assert isinstance(self.plan.implementation, AlgorithmSpec)
        return self.plan.implementation

    @property
    def operation(self):
        operation = self.plan.call.operation
        assert isinstance(operation, GroupExchangeSemantics)
        return operation.primitive

    @property
    def group_kind(self) -> str:
        return self.plan.resolved_group.kind

    @property
    def block_dim(self) -> tuple[int, int, int]:
        participation = self.plan.participation
        if participation is None or participation.exact_block_dim is None:
            raise ValueError("CUB Exchange request requires exact block dimensions")
        return participation.exact_block_dim

    @property
    def items_per_thread(self) -> int:
        return self.operation.items_per_thread

    @property
    def mode(self) -> BlockExchangeMode:
        return self.operation.mode

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        return self.plan.artifact_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CubExchangeRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def symbol_name(self) -> str:
        assert self.plan.artifact_key is not None
        signature = hashlib.sha256(repr(self.plan.artifact_key).encode()).hexdigest()[
            :12
        ]
        suffixes = []
        if self.rank_type is not None:
            suffixes.append(f"rank_{TYPE_SPECS[self.rank_type].token}")
        if self.valid_flag_type is not None:
            suffixes.append(f"flag_{TYPE_SPECS[self.valid_flag_type].token}")
        suffix = f"_{'_'.join(suffixes)}" if suffixes else ""
        return (
            "cuda_coop_cutlass_cub_exchange_"
            f"{self.group_kind}_{_block_dim_token(self.block_dim)}_"
            f"{self.implementation.method_name.lower()}_"
            f"{TYPE_SPECS[self.value_type].token}_x{self.items_per_thread}"
            f"{suffix}_{signature}"
        )


def _warp_instances(request: _CubExchangeRequest) -> tuple[int, int]:
    x, y, z = request.block_dim
    block_threads = x * y * z
    logical_width = request.implementation.template_arguments.get(
        "LOGICAL_WARP_THREADS"
    )
    if not isinstance(logical_width, int) or logical_width < 1:
        raise ValueError("WarpExchange plan requires a static logical warp width")
    if block_threads < logical_width or block_threads % logical_width != 0:
        raise ValueError("WarpExchange plan requires complete logical warps")
    return block_threads // logical_width, logical_width


def _render_cub_exchange(request: _CubExchangeRequest) -> list[str]:
    request.__post_init__()
    implementation = request.implementation
    spec = TYPE_SPECS[request.value_type]
    template_arguments = ", ".join(
        _render_template_argument(request, name, value)
        for name, value in implementation.ordered_template_arguments
    )

    storage = "storage"
    storage_lines = ["  __shared__ typename implementation_type::TempStorage storage;"]
    if request.group_kind in {"warp", "threads_within_warp"}:
        instances, logical_width = _warp_instances(request)
        storage_lines = [
            "  __shared__ typename implementation_type::TempStorage "
            f"storage[{instances}];",
            "  unsigned int storage_instance =",
            f"      cuda_coop_cutlass_linear_tid() / {logical_width}u;",
        ]
        storage = "storage[storage_instance]"

    params = [
        f"{spec.cpp_type} item{index}" for index in range(request.items_per_thread)
    ]
    rank_spec = None
    if request.rank_type is not None:
        rank_spec = TYPE_SPECS[request.rank_type]
        params.extend(
            f"{rank_spec.cpp_type} rank{index}"
            for index in range(request.items_per_thread)
        )
    flag_spec = None
    if request.valid_flag_type is not None:
        flag_spec = TYPE_SPECS[request.valid_flag_type]
        params.extend(
            f"{flag_spec.cpp_type} valid{index}"
            for index in range(request.items_per_thread)
        )
    params.append(f"{spec.cpp_type}* result_items")

    values = ", ".join(f"item{index}" for index in range(request.items_per_thread))
    input_lines = [
        f"  {spec.cpp_type} input_items[{request.items_per_thread}] = {{{values}}};",
        f"  {spec.cpp_type} output_items[{request.items_per_thread}];",
    ]
    call_arguments = ["input_items", "output_items"]
    if rank_spec is not None:
        ranks = ", ".join(f"rank{index}" for index in range(request.items_per_thread))
        input_lines.append(
            f"  {rank_spec.cpp_type} ranks[{request.items_per_thread}] = {{{ranks}}};"
        )
        call_arguments.append("ranks")
    if flag_spec is not None:
        flags = ", ".join(f"valid{index}" for index in range(request.items_per_thread))
        input_lines.append(
            f"  {flag_spec.cpp_type} valid_flags[{request.items_per_thread}] = "
            f"{{{flags}}};"
        )
        call_arguments.append("valid_flags")

    output_lines = [
        f"  result_items[{index}] = output_items[{index}];"
        for index in range(request.items_per_thread)
    ]
    return [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = ::cub::{implementation.struct_name}<"
        f"{template_arguments}>;",
        *storage_lines,
        *input_lines,
        f"  implementation_type({storage}).{implementation.method_name}("
        f"{', '.join(call_arguments)});",
        _storage_reuse_barrier_line(request),
        *output_lines,
        "}",
    ]


def _register_renderer() -> None:
    _provider_rendering.register_bundle_renderer(
        "cub_group_exchange",
        render=_render_cub_exchange,
        include_lines=(
            "#include <cub/block/block_exchange.cuh>",
            "#include <cub/warp/warp_exchange.cuh>",
        ),
        cccl_headers=(
            (
                "#include <cub/block/block_exchange.cuh>",
                "cub/block/block_exchange.cuh",
            ),
            (
                "#include <cub/warp/warp_exchange.cuh>",
                "cub/warp/warp_exchange.cuh",
            ),
        ),
    )


_register_renderer()

_resolve_type = _provider_state.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _resolve_auxiliary_values(
    value: Any,
    *,
    name: str,
    items_per_thread: int,
    allowed: frozenset[type],
) -> tuple[type | None, tuple[Any, ...]]:
    if value is None:
        return None, ()
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_ROOT_SCOPE}.exchange {name} must be ThreadData")
    if value.items_per_thread != items_per_thread:
        raise ValueError(
            f"{_ROOT_SCOPE}.exchange {name} must have matching items_per_thread"
        )
    value_type, values = _provider_types.resolve_thread_data_value_type(
        value,
        allowed=allowed,
        feature="exchange",
        scope=_ROOT_SCOPE,
        resolve_type=_resolve_type,
    )
    return value_type, tuple(values)


def _resolve_exchange_operands(
    *,
    value: ThreadData,
    ranks: ThreadData | None,
    valid_flags: ThreadData | None,
) -> tuple[
    type, tuple[Any, ...], type | None, tuple[Any, ...], type | None, tuple[Any, ...]
]:
    value_type, values = _provider_types.resolve_thread_data_value_type(
        value,
        allowed=ALL_PROVIDER_TYPES,
        feature="exchange",
        scope=_ROOT_SCOPE,
        resolve_type=_resolve_type,
    )
    rank_type, rank_values = _resolve_auxiliary_values(
        ranks,
        name="ranks",
        items_per_thread=value.items_per_thread,
        allowed=_RANK_TYPES,
    )
    valid_flag_type, valid_flag_values = _resolve_auxiliary_values(
        valid_flags,
        name="valid_flags",
        items_per_thread=value.items_per_thread,
        allowed=_VALID_FLAG_TYPES,
    )
    return (
        value_type,
        tuple(values),
        rank_type,
        rank_values,
        valid_flag_type,
        valid_flag_values,
    )


def _resolve_exchange_operand_types(
    *,
    value: ThreadData,
    ranks: ThreadData | None,
    valid_flags: ThreadData | None,
) -> tuple[type, type | None, type | None]:
    value_type, _, rank_type, _, valid_flag_type, _ = _resolve_exchange_operands(
        value=value,
        ranks=ranks,
        valid_flags=valid_flags,
    )
    return value_type, rank_type, valid_flag_type


def provider_exchange(
    *,
    plan: GroupLoweringPlan,
    value: ThreadData,
    ranks: ThreadData | None = None,
    valid_flags: ThreadData | None = None,
) -> ThreadData:
    """Materialize one plan-validated CUB Exchange call."""

    if not isinstance(plan, GroupLoweringPlan):
        raise TypeError(f"{_ROOT_SCOPE}.exchange plan must be a GroupLoweringPlan")
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_ROOT_SCOPE}.exchange value must be ThreadData")
    (
        value_type,
        values,
        rank_type,
        rank_values,
        valid_flag_type,
        valid_flag_values,
    ) = _resolve_exchange_operands(
        value=value,
        ranks=ranks,
        valid_flags=valid_flags,
    )
    request = _CubExchangeRequest(
        plan=plan,
        value_type=value_type,
        rank_type=rank_type,
        valid_flag_type=valid_flag_type,
    )
    mode = request.mode
    if mode.uses_ranks != (rank_type is not None):
        requirement = "requires" if mode.uses_ranks else "does not accept"
        raise ValueError(f"{_ROOT_SCOPE}.exchange {mode.value} {requirement} ranks")
    if (
        mode is BlockExchangeMode.SCATTER_TO_STRIPED_GUARDED
        and rank_type not in _SIGNED_RANK_TYPES
    ):
        raise TypeError(
            f"{_ROOT_SCOPE}.exchange {mode.value} requires signed Int32 or Int64 ranks"
        )
    if mode.uses_valid_flags != (valid_flag_type is not None):
        requirement = "requires" if mode.uses_valid_flags else "does not accept"
        raise ValueError(
            f"{_ROOT_SCOPE}.exchange {mode.value} {requirement} valid_flags"
        )

    validate_operand_domains(
        plan.resolved_group,
        {
            "value": value,
            **({"ranks": ranks} if ranks is not None else {}),
            **({"valid_flags": valid_flags} if valid_flags is not None else {}),
        },
        scope=_ROOT_SCOPE,
        primitive_name="exchange",
    )
    _provider_state.register_request(request)
    result_tensor = _cute.make_rmem_tensor(value.items_per_thread, value_type)
    ffi(
        name=request.symbol_name,
        params_types=[
            *([value_type] * value.items_per_thread),
            *([rank_type] * value.items_per_thread if rank_type is not None else []),
            *(
                [valid_flag_type] * value.items_per_thread
                if valid_flag_type is not None
                else []
            ),
            llvm.PointerType.get(0),
        ],
        return_type=None,
    )(
        *values,
        *rank_values,
        *valid_flag_values,
        result_tensor.iterator.llvm_ptr,
    )
    result_values = tuple(
        result_tensor[index] for index in range(value.items_per_thread)
    )
    assert plan.result is not None
    return attach_thread_data_metadata(
        ThreadData.from_values(
            *result_values,
            dtype=_provider_types.thread_data_output_dtype(value, value_type),
        ),
        metadata_for_group(
            plan.resolved_group,
            visibility=plan.result.primary.visibility,
        ),
    )


__all__ = ["_CubExchangeRequest", "provider_exchange"]
