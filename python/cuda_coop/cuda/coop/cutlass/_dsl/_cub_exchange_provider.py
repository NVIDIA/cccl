# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan-driven public-CUB provider renderer for CUTLASS Exchange."""

from __future__ import annotations

import dataclasses
from typing import Any

from cutlass import cute as _cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Int64, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    GroupExchangeSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
)
from cuda.coop._core.block import (
    BlockExchangeMode,
    BlockExchangeSpec,
)

from .._limits import MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD
from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)
from . import _provider as _provider_support
from ._provider import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from ._provider import RADIX_KEY_TYPES as _RADIX_KEY_TYPES
from ._provider import TYPE_SPECS as _TYPE_SPECS
from ._single_phase import get_active_single_phase_context
from ._symbols import block_dim_token as _block_dim_token
from ._thread_data import ThreadData
from ._thread_group import ThreadGroup

_ROOT_SCOPE = __name__.split("._dsl.", 1)[0]
_ROOT_MODES = frozenset(BlockExchangeMode)
_BLOCK_MODES = frozenset(BlockExchangeMode)
_WARP_STRIPED_BLOCK_MODES = frozenset(
    {
        BlockExchangeMode.WARP_STRIPED_TO_BLOCKED,
        BlockExchangeMode.BLOCKED_TO_WARP_STRIPED,
    }
)
_RANK_TYPES = _RADIX_KEY_TYPES
_SIGNED_RANK_TYPES = frozenset({Int32, Int64})
_VALID_FLAG_TYPES = _RADIX_KEY_TYPES


def _render_template_argument(
    request: _CubExchangeRequest,
    name: str,
    value: Any,
) -> str:
    if name == "T":
        if value is not request.value_type:
            raise ValueError("CUB Exchange template dtype does not match its request")
        return _TYPE_SPECS[value].cpp_type
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot render CUB Exchange template argument {name}={value!r}")


def _storage_reuse_barrier_line(request: _CubExchangeRequest) -> str:
    if request.plan is None:
        return "  cuda_coop_cutlass_block_sync();"
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
    external_scratch: bool,
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
    if primitive.items_per_thread > MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD:
        raise ValueError("group Exchange exceeds the supported item-count limit")
    if primitive.mode not in _ROOT_MODES:
        raise ValueError("group-first Exchange plan uses an unsupported mode")
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
    expected_storage_ownership = (
        StorageOwnership.CALLER if external_scratch else StorageOwnership.IMPLEMENTATION
    )
    if temp_storage.ownership is not expected_storage_ownership:
        raise ValueError(
            "CUB group Exchange temporary-storage ownership does not match its request"
        )
    if external_scratch:
        if plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise ValueError("deferred CUB Exchange scratch is block-scoped only")
        if (
            temp_storage.address_space != "shared"
            or temp_storage.instances != 1
            or temp_storage.instance_index != "cta"
            or not temp_storage.exact_layout_required
        ):
            raise ValueError(
                "deferred CUB BlockExchange requires exact caller-owned shared storage"
            )
    result = plan.result
    if result is None or len(result.values) != 1:
        raise ValueError("CUB Exchange plan requires one logical result")
    logical_result = result.values[0]
    if logical_result.items_per_member != primitive.items_per_thread:
        raise ValueError("CUB Exchange result item count does not match its request")


def _validate_compatibility_spec(
    spec: BlockExchangeSpec,
    *,
    value_type: type,
    rank_type: type | None,
    valid_flag_type: type | None,
) -> None:
    call = spec.call
    if call.items_per_thread > MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD:
        raise ValueError("scoped BlockExchange exceeds the supported item-count limit")
    if call.dtype is not value_type:
        raise ValueError("scoped BlockExchange dtype does not match its request")
    if call.mode not in _BLOCK_MODES:
        raise ValueError("scoped BlockExchange mode is unsupported")
    if call.rank_dtype is not rank_type:
        raise ValueError("scoped BlockExchange rank dtype does not match its request")
    if (
        call.mode is BlockExchangeMode.SCATTER_TO_STRIPED_GUARDED
        and rank_type not in _SIGNED_RANK_TYPES
    ):
        raise ValueError(
            "scoped BlockExchange ScatterToStripedGuarded requires signed "
            "Int32 or Int64 ranks"
        )
    if call.valid_flag_dtype is not valid_flag_type:
        raise ValueError("scoped BlockExchange flag dtype does not match its request")
    if call.value_form.value != "out_of_place":
        raise ValueError("scoped BlockExchange requires out-of-place values")
    if call.warp_time_slicing:
        raise ValueError("scoped BlockExchange does not use warp time slicing")
    block_threads = 1
    for dim in spec.block_dim:
        block_threads *= dim
    if (
        call.mode in _WARP_STRIPED_BLOCK_MODES
        and block_threads > 32
        and block_threads % 32 != 0
    ):
        raise ValueError(
            "scoped BlockExchange warp-striped modes require a CTA no larger "
            "than one warp or a whole number of complete physical warps"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class _CubExchangeRequest:
    value_type: type
    rank_type: type | None = None
    valid_flag_type: type | None = None
    plan: GroupLoweringPlan | None = None
    compatibility_spec: BlockExchangeSpec | None = None
    external_scratch: bool = False
    kind: str = "cub_group_exchange"

    def __post_init__(self) -> None:
        if (self.plan is None) == (self.compatibility_spec is None):
            raise ValueError(
                "CUB Exchange request requires exactly one plan or compatibility spec"
            )
        if self.plan is not None:
            _validate_planned_exchange(
                self.plan,
                value_type=self.value_type,
                rank_type=self.rank_type,
                valid_flag_type=self.valid_flag_type,
                external_scratch=self.external_scratch,
            )
        else:
            assert self.compatibility_spec is not None
            _validate_compatibility_spec(
                self.compatibility_spec,
                value_type=self.value_type,
                rank_type=self.rank_type,
                valid_flag_type=self.valid_flag_type,
            )
        if self.external_scratch and self.group_kind != "block":
            raise ValueError("deferred CUB Exchange scratch is block-scoped only")

    @property
    def implementation(self) -> AlgorithmSpec:
        if self.plan is not None:
            assert isinstance(self.plan.implementation, AlgorithmSpec)
            return self.plan.implementation
        assert self.compatibility_spec is not None
        return self.compatibility_spec.specialization

    @property
    def operation(self):
        if self.plan is not None:
            operation = self.plan.call.operation
            assert isinstance(operation, GroupExchangeSemantics)
            return operation.primitive
        assert self.compatibility_spec is not None
        return self.compatibility_spec.call

    @property
    def group_kind(self) -> str:
        if self.plan is None:
            return "block"
        return self.plan.resolved_group.kind

    @property
    def block_dim(self) -> tuple[int, int, int]:
        if self.plan is not None:
            participation = self.plan.participation
            assert participation is not None
            return participation.exact_block_dim
        assert self.compatibility_spec is not None
        return self.compatibility_spec.block_dim

    @property
    def items_per_thread(self) -> int:
        return self.operation.items_per_thread

    @property
    def mode(self) -> BlockExchangeMode:
        return self.operation.mode

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        if self.plan is not None:
            assert self.plan.artifact_key is not None
            if not self.external_scratch:
                return self.plan.artifact_key
            return ("external_scratch", self.plan.artifact_key)
        assert self.compatibility_spec is not None
        key = (
            "scoped_block_exchange",
            self.compatibility_spec.semantic_key,
        )
        if not self.external_scratch:
            return key
        return ("external_scratch", key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CubExchangeRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def symbol_name(self) -> str:
        implementation = self.implementation
        suffixes = []
        if self.rank_type is not None:
            suffixes.append(f"rank_{_TYPE_SPECS[self.rank_type].token}")
        if self.valid_flag_type is not None:
            suffixes.append(f"flag_{_TYPE_SPECS[self.valid_flag_type].token}")
        suffix = f"_{'_'.join(suffixes)}" if suffixes else ""
        symbol = (
            "cuda_coop_cutlass_cub_exchange_"
            f"{self.group_kind}_{_block_dim_token(self.block_dim)}_"
            f"{implementation.method_name.lower()}_"
            f"{_TYPE_SPECS[self.value_type].token}_x{self.items_per_thread}"
            f"{suffix}"
        )
        if self.external_scratch:
            return f"{symbol}_external_scratch"
        return symbol

    @property
    def scratch_requirement_key(self) -> tuple[Any, ...]:
        """Identity of the instantiated CUB class whose scratch layout is needed."""

        implementation = self.implementation
        return (
            "cub_temp_storage_layout",
            implementation.struct_name,
            tuple(
                (name, _render_template_argument(self, name, value))
                for name, value in implementation.ordered_template_arguments
            ),
        )

    @property
    def scratch_cpp_type(self) -> str:
        implementation = self.implementation
        template_arguments = ", ".join(
            _render_template_argument(self, name, value)
            for name, value in implementation.ordered_template_arguments
        )
        return (
            f"typename ::cub::{implementation.struct_name}<"
            f"{template_arguments}>::TempStorage"
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
    # Re-run validation at source materialization rather than trusting a
    # previously constructed request.
    request.__post_init__()
    implementation = request.implementation
    spec = _TYPE_SPECS[request.value_type]
    template_arguments = ", ".join(
        _render_template_argument(request, name, value)
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
    elif request.group_kind == "warp":
        instances, logical_width = _warp_instances(request)
        storage_lines = [
            "  __shared__ typename implementation_type::TempStorage "
            f"storage[{instances}];",
            "  unsigned int storage_instance =",
            f"      cuda_coop_cutlass_linear_tid() / {logical_width}u;",
        ]
        storage = "storage[storage_instance]"
    elif request.group_kind == "threads_within_warp":
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
        rank_spec = _TYPE_SPECS[request.rank_type]
        params.extend(
            f"{rank_spec.cpp_type} rank{index}"
            for index in range(request.items_per_thread)
        )
    flag_spec = None
    if request.valid_flag_type is not None:
        flag_spec = _TYPE_SPECS[request.valid_flag_type]
        params.extend(
            f"{flag_spec.cpp_type} valid{index}"
            for index in range(request.items_per_thread)
        )
    if request.external_scratch:
        params.extend(
            (
                "unsigned int temp_storage_smem_addr",
                "int temp_storage_bytes",
                "int temp_storage_auto_sync",
            )
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
        *input_lines,
        f"  implementation_type({storage}).{implementation.method_name}("
        f"{', '.join(call_arguments)});",
        *barrier_lines,
        *output_lines,
        "}",
    ]


def _cub_exchange_scratch_layout_probe(
    request: _CubExchangeRequest,
) -> _provider_support.ScratchLayoutProbe | None:
    if not request.external_scratch:
        return None
    return _provider_support.make_scratch_layout_probe(
        request.scratch_requirement_key,
        request.scratch_cpp_type,
    )


def _register_renderer() -> None:
    _provider_support.register_bundle_renderer(
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
        scratch_layout_probe=_cub_exchange_scratch_layout_probe,
    )


_register_renderer()

_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="thread_group",
)


def _validate_output(
    output: Any,
    *,
    value: ThreadData,
    value_type: type,
) -> ThreadData | None:
    return _provider_support.validate_thread_data_output(
        output=output,
        expected_items_per_thread=value.items_per_thread,
        resolved_dtype=value_type,
        scope=_ROOT_SCOPE,
        primitive_name="exchange",
        output_name="output",
        resolve_type=_resolve_type,
        assigned_dtype=_provider_support.thread_data_output_dtype(value, value_type),
        type_label=f"{_ROOT_SCOPE}.ThreadData",
        item_count_message=(
            f"{_ROOT_SCOPE}.exchange output must have matching items_per_thread"
        ),
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
    value_type, values = _provider_support.resolve_thread_data_value_type(
        value,
        allowed=allowed,
        feature="exchange",
        scope=_ROOT_SCOPE,
        resolve_type=_resolve_type,
    )
    return value_type, tuple(values)


def _make_request(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value_type: type,
    items_per_thread: int,
    mode: BlockExchangeMode,
    rank_type: type | None,
    valid_flag_type: type | None,
    warp_time_slicing: bool,
    source: str,
    external_scratch: bool = False,
) -> _CubExchangeRequest:
    from .. import _group_exchange as _group_frontend

    plan = _group_frontend._make_group_exchange_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        items_per_thread=items_per_thread,
        mode=mode.value,
        rank_dtype=rank_type,
        valid_flag_dtype=valid_flag_type,
        warp_time_slicing=warp_time_slicing,
        source=source,
    ).require_supported()
    if external_scratch:
        temp_storage = plan.temp_storage
        assert temp_storage is not None
        plan = dataclasses.replace(
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
    return _CubExchangeRequest(
        plan=plan,
        value_type=value_type,
        rank_type=rank_type,
        valid_flag_type=valid_flag_type,
        external_scratch=external_scratch,
    )


def _deferred_temp_storage_for_exchange(
    *,
    group: ThreadGroup,
    source: str,
) -> Any | None:
    context = get_active_single_phase_context()
    temp_storage = context.temp_storage if context is not None else None
    if temp_storage is None or not getattr(temp_storage, "is_deferred", False):
        return None
    if group.kind != "block" or not source.startswith("scoped_block"):
        raise ValueError(
            f"{_ROOT_SCOPE}.exchange deferred TempStorage is supported only by "
            "scoped block exchange calls"
        )
    return temp_storage


def provider_exchange(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    value: Any,
    mode: str = "striped_to_blocked",
    output: Any = None,
    ranks: Any = None,
    valid_flags: Any = None,
    warp_time_slicing: bool = False,
    source: str = "cutlass_group_exchange_provider",
) -> ThreadData:
    """Materialize one whole-register-array CUB Exchange call."""

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_ROOT_SCOPE}.exchange group must be a ThreadGroup")
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_ROOT_SCOPE}.exchange value must be ThreadData")
    if value.items_per_thread > MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD:
        raise NotImplementedError(
            f"{_ROOT_SCOPE}.exchange supports at most "
            f"{MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD} items per thread"
        )
    if not isinstance(warp_time_slicing, bool):
        raise TypeError(f"{_ROOT_SCOPE}.exchange warp_time_slicing must be a bool")
    try:
        mode_enum = BlockExchangeMode(mode)
    except ValueError:
        raise ValueError(
            f"{_ROOT_SCOPE}.exchange mode must be a supported Exchange mode"
        ) from None

    value_type, values = _provider_support.resolve_thread_data_value_type(
        value,
        allowed=_ALL_PROVIDER_TYPES,
        feature="exchange",
        scope=_ROOT_SCOPE,
        resolve_type=_resolve_type,
    )
    values = tuple(values)
    output_td = _validate_output(
        output,
        value=value,
        value_type=value_type,
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
    if mode_enum.uses_ranks != (rank_type is not None):
        requirement = "requires" if mode_enum.uses_ranks else "does not accept"
        raise ValueError(
            f"{_ROOT_SCOPE}.exchange {mode_enum.value} {requirement} ranks"
        )
    if (
        mode_enum is BlockExchangeMode.SCATTER_TO_STRIPED_GUARDED
        and rank_type not in _SIGNED_RANK_TYPES
    ):
        raise TypeError(
            f"{_ROOT_SCOPE}.exchange {mode_enum.value} requires signed "
            "Int32 or Int64 ranks"
        )
    if mode_enum.uses_valid_flags != (valid_flag_type is not None):
        requirement = "requires" if mode_enum.uses_valid_flags else "does not accept"
        raise ValueError(
            f"{_ROOT_SCOPE}.exchange {mode_enum.value} {requirement} valid_flags"
        )

    deferred_temp_storage = _deferred_temp_storage_for_exchange(
        group=group,
        source=source,
    )
    request = _make_request(
        group=group,
        launch=launch,
        value_type=value_type,
        items_per_thread=value.items_per_thread,
        mode=mode_enum,
        rank_type=rank_type,
        valid_flag_type=valid_flag_type,
        warp_time_slicing=warp_time_slicing,
        source=source,
        external_scratch=deferred_temp_storage is not None,
    )
    result_metadata = None
    if request.plan is not None:
        validate_operand_domains(
            request.plan.resolved_group,
            {
                "value": value,
                **({"ranks": ranks} if ranks is not None else {}),
                **({"valid_flags": valid_flags} if valid_flags is not None else {}),
            },
            scope=_ROOT_SCOPE,
            primitive_name="exchange",
        )
        assert request.plan.result is not None
        result_metadata = metadata_for_group(
            request.plan.resolved_group,
            visibility=request.plan.result.visibility,
        )
    _provider_support.register_request(request)
    scratch_args: tuple[Any, ...] = ()
    scratch_param_types: list[type] = []
    if deferred_temp_storage is not None:
        scratch_addr, scratch_size, scratch_auto_sync = (
            _provider_support.register_deferred_temp_storage_event(
                deferred_temp_storage,
                primitive_name="exchange",
                requirement_key=request.scratch_requirement_key,
            )
        )
        scratch_args = (scratch_addr, scratch_size, scratch_auto_sync)
        scratch_param_types = [Uint32, Int32, Int32]
    result_tensor = _cute.make_rmem_tensor(value.items_per_thread, value_type)
    params_types = [
        *([value_type] * value.items_per_thread),
        *([rank_type] * value.items_per_thread if rank_type is not None else []),
        *(
            [valid_flag_type] * value.items_per_thread
            if valid_flag_type is not None
            else []
        ),
        *scratch_param_types,
        llvm.PointerType.get(0),
    ]
    ffi(
        name=request.symbol_name,
        params_types=params_types,
        return_type=None,
    )(
        *values,
        *rank_values,
        *valid_flag_values,
        *scratch_args,
        result_tensor.iterator.llvm_ptr,
    )
    result_values = tuple(
        result_tensor[index] for index in range(value.items_per_thread)
    )
    if output_td is None:
        return attach_thread_data_metadata(
            ThreadData.from_values(
                *result_values,
                dtype=_provider_support.thread_data_output_dtype(value, value_type),
            ),
            result_metadata,
        )
    for index, result in enumerate(result_values):
        output_td[index] = result
    return attach_thread_data_metadata(output_td, result_metadata)


__all__ = ["_CubExchangeRequest", "provider_exchange"]
