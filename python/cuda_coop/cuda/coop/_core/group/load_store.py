# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan a group's memory transfer using a CUB Block or Warp implementation.

The operation record describes the requested transfer without compiler values.
After group resolution, the planner selects a CUB specialization and records
participation, argument bounds, and scratch requirements. Compiler adapters
use that plan to generate the call and provide any required storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any

from .._bindings import (
    ArgumentBinding,
    BindingKind,
    normalize_i32_binding,
    normalize_i64_binding,
)
from .._symbols import semantic_token
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.load_store import (
    BlockLoadStoreAlgorithm,
    make_block_load_specialization,
    make_block_store_specialization,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.load_store import (
    WarpLoadStoreAlgorithm,
    make_warp_load_specialization,
    make_warp_store_specialization,
)
from ._dispatch import _register_group_operation_family
from ._execution_requirements import (
    _build_execution_requirements,
    _unsupported,
    _unsupported_cub_warp_width,
)
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ImplementationProvenance,
    PreconditionEnforcement,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)


class GroupLoadStoreKind(str, Enum):
    """Distinguish a load into per-thread items from a store to memory."""

    LOAD = "load"
    STORE = "store"


class GroupLoadStoreAlgorithm(str, Enum):
    """Name transfer algorithms before the planner selects Block or Warp CUB.

    The set includes both families' choices. The planner checks whether the
    resolved group supports the requested choice, including warp width limits.
    """

    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"
    WARP_TRANSPOSE = "warp_transpose"
    WARP_TRANSPOSE_TIMESLICED = "warp_transpose_timesliced"


_STORAGE_FREE_ALGORITHMS = frozenset(
    {
        GroupLoadStoreAlgorithm.DIRECT,
        GroupLoadStoreAlgorithm.STRIPED,
        GroupLoadStoreAlgorithm.VECTORIZE,
    }
)


@dataclass(frozen=True, eq=False)
class GroupLoadStoreSemantics:
    """Describe one transfer and the values available when planning it.

    ``dtype`` and ``items_per_thread`` determine each thread's payload.
    Argument bindings distinguish omitted arguments, known constants, and
    runtime inputs. Storage fields carry the caller's requests until the
    selected algorithm's scratch needs are known. Direct, striped, and
    vectorized transfers need no scratch, so their semantic keys exclude those
    storage requests.
    """

    kind: GroupLoadStoreKind
    dtype: Any
    items_per_thread: int
    algorithm: GroupLoadStoreAlgorithm = GroupLoadStoreAlgorithm.DIRECT
    valid_items: ArgumentBinding = field(
        default_factory=ArgumentBinding.omitted
    )
    oob_default: ArgumentBinding = field(
        default_factory=ArgumentBinding.omitted
    )
    offset: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    storage_ownership: StorageOwnership = StorageOwnership.IMPLEMENTATION
    storage_sharing: str | None = None
    storage_size_in_bytes: int | None = None
    storage_alignment: int | None = None
    storage_auto_sync: bool = True

    def __post_init__(self) -> None:
        """Normalize selectors and check argument and storage requests.

        Check integer representation here. Bounds that depend on group size,
        such as ``valid_items``, are checked after the group is resolved.
        """

        object.__setattr__(self, "kind", GroupLoadStoreKind(self.kind))
        object.__setattr__(
            self,
            "algorithm",
            GroupLoadStoreAlgorithm(self.algorithm),
        )
        object.__setattr__(
            self,
            "storage_ownership",
            StorageOwnership(self.storage_ownership),
        )
        if (
            not isinstance(self.items_per_thread, int)
            or isinstance(self.items_per_thread, bool)
            or self.items_per_thread <= 0
        ):
            raise ValueError("items_per_thread must be a positive integer")
        for name in ("valid_items", "oob_default", "offset"):
            if not isinstance(getattr(self, name), ArgumentBinding):
                raise TypeError(f"{name} must be an ArgumentBinding")
        object.__setattr__(
            self,
            "valid_items",
            normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        object.__setattr__(
            self,
            "offset",
            normalize_i64_binding(self.offset, name="offset"),
        )
        if (
            self.offset.kind is BindingKind.STATIC
            and int(self.offset.value) < 0
        ):
            raise ValueError("static offset must be nonnegative")
        if self.kind is GroupLoadStoreKind.STORE and (
            self.oob_default.kind is not BindingKind.OMITTED
        ):
            raise ValueError("oob_default is valid only for group load")
        if (
            self.oob_default.kind is not BindingKind.OMITTED
            and self.valid_items.kind is BindingKind.OMITTED
        ):
            raise ValueError("oob_default requires valid_items")
        if self.storage_sharing not in {None, "shared", "exclusive"}:
            raise ValueError("storage_sharing must be shared or exclusive")
        for name in ("storage_size_in_bytes", "storage_alignment"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(self.storage_auto_sync, bool):
            raise TypeError("storage_auto_sync must be a bool")
        if self.storage_ownership is StorageOwnership.NONE:
            if self.algorithm not in _STORAGE_FREE_ALGORITHMS:
                raise ValueError(
                    "storage-free group Load/Store is valid only for direct, "
                    "striped, or vectorize algorithms"
                )
            if any(
                value is not None
                for value in (
                    self.storage_sharing,
                    self.storage_size_in_bytes,
                    self.storage_alignment,
                )
            ):
                raise ValueError(
                    "storage-free operations cannot carry storage layout"
                )
            if self.storage_auto_sync:
                raise ValueError(
                    "storage-free operations cannot request automatic sync"
                )
        elif self.storage_ownership is StorageOwnership.IMPLEMENTATION:
            if any(
                value is not None
                for value in (
                    self.storage_sharing,
                    self.storage_size_in_bytes,
                    self.storage_alignment,
                )
            ):
                raise ValueError(
                    "implementation-owned storage cannot carry caller requests"
                )
        elif self.storage_sharing is None:
            raise ValueError("caller-owned storage requires storage_sharing")

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def has_oob_default(self) -> bool:
        return self.oob_default.kind is not BindingKind.OMITTED

    @property
    def has_offset(self) -> bool:
        return self.offset.kind is not BindingKind.OMITTED

    @property
    def result_visibility(self) -> ResultVisibility:
        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self) -> bool:
        """Report that the call writes through its output argument."""

        return False

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify the transfer, excluding unused scratch requests."""

        common = (
            f"group_{self.kind.value}",
            semantic_token(self.dtype),
            self.items_per_thread,
            self.algorithm.value,
            self.valid_items.semantic_key,
            self.oob_default.semantic_key,
            self.offset.semantic_key,
        )
        if self.algorithm in _STORAGE_FREE_ALGORITHMS:
            return (*common, StorageOwnership.NONE.value)
        return (
            *common,
            self.storage_ownership.value,
            self.storage_sharing,
            self.storage_size_in_bytes,
            self.storage_alignment,
            self.storage_auto_sync,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupLoadStoreSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _call_classifications(
    operation: GroupLoadStoreSemantics,
) -> tuple[ParameterClassification, ...]:
    """Describe each argument's role and when its value is available.

    Both transfers take a memory pointer and per-thread items, but load and
    store reverse their input/output roles. Omitted optional arguments do not
    enter the call. Record constants separately from runtime inputs.
    """

    classifications = [
        ParameterClassification(
            "source"
            if operation.kind is GroupLoadStoreKind.LOAD
            else "destination",
            ArgumentKind.RUNTIME,
            (
                ParameterRole.INPUT
                if operation.kind is GroupLoadStoreKind.LOAD
                else ParameterRole.OUTPUT
            ),
        )
    ]
    classifications.append(
        ParameterClassification(
            "output" if operation.kind is GroupLoadStoreKind.LOAD else "value",
            ArgumentKind.RUNTIME,
            (
                ParameterRole.OUTPUT
                if operation.kind is GroupLoadStoreKind.LOAD
                else ParameterRole.INPUT
            ),
        )
    )
    for name, binding in (
        ("valid_items", operation.valid_items),
        ("oob_default", operation.oob_default),
        ("offset", operation.offset),
    ):
        if binding.argument_kind is None:
            continue
        classifications.append(
            ParameterClassification(
                name,
                binding.argument_kind,
                (
                    ParameterRole.CONSTANT
                    if binding.kind is BindingKind.STATIC
                    else ParameterRole.INPUT
                ),
            )
        )
    classifications.append(
        ParameterClassification(
            "algorithm",
            ArgumentKind.STATIC,
            ParameterRole.CONSTANT,
        )
    )
    return tuple(classifications)


def _plan_load_store(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupLoadStoreSemantics,
) -> GroupLoweringPlan:
    """Choose a CUB Load/Store specialization for a resolved thread group.

    Called by ``plan_group_primitive`` after group resolution. Select the
    block or warp implementation and describe its participation, scratch
    storage, and storage-reuse synchronization requirements for later
    compilation and lowering. Direct, striped, and vectorized algorithms
    require no scratch storage or storage-reuse barrier; other algorithms
    retain the requested storage ownership, layout constraints, sharing, and
    automatic sync policy.

    Validate static ``valid_items`` against the group tile size and check that
    pointer offsets fit signed 64-bit arithmetic. Runtime bounds remain caller
    preconditions recorded in the plan. Each physical or logical warp handles
    a consecutive tile, so its specialization always takes a runtime pointer
    offset. The backend combines that tile's origin with the user offset
    preserved in ``operation``; the plan limits the user offset accordingly.

    Parameters
    ----------
    call : GroupPrimitiveCall
        Original group call, retained in the plan and unsupported diagnostics.
    resolved : ThreadGroup
        Group resolved against ``launch``. Supported block, physical-warp, and
        logical-warp groups must have a static size and complete membership.
    launch : LaunchFacts
        Launch facts used during group resolution, including exact block
        dimensions for specialization and counting warp-group instances.
    operation : GroupLoadStoreSemantics
        Normalized load or store semantics from ``call.operation``, including
        dtype, item count, algorithm, argument bindings, and storage requests.

    Returns
    -------
    GroupLoweringPlan
        CUB specialization, execution requirements, and implementation
        provenance, or an ``UNSUPPORTED`` plan with a reason when the group
        kind, warp width, or algorithm variant is unsupported. This builds
        metadata; compilation and device-storage allocation happen during
        later lowering.

    Raises
    ------
    TypeError
        Static ``valid_items`` is not an integer or is a boolean.
    ValueError
        Static ``valid_items`` is outside the group tile, a warp tile origin
        or its sum with the user offset exceeds signed 64-bit range, or the
        specialization builder rejects an argument binding.
    AssertionError
        A supported group lacks a static size or exact block dimensions are
        missing from ``launch``; group resolution must establish these first.
    """

    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "cuda.coop Load/Store supports this_block(), complete physical "
            "this_warp(), and power-of-two logical-warp groups",
        )
    group_size = resolved.static_size
    assert group_size is not None
    tile_items = group_size * operation.items_per_thread
    if operation.valid_items.kind is BindingKind.STATIC:
        valid_items = operation.valid_items.value
        if isinstance(valid_items, bool) or not isinstance(
            valid_items, Integral
        ):
            raise TypeError("static valid_items must be an integer")
        valid_items = int(valid_items)
        if not 0 <= valid_items <= tile_items:
            raise ValueError(
                "static valid_items must be between zero and the group tile "
                f"size ({tile_items})"
            )
    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    maximum_user_offset = (1 << 63) - 1
    if resolved.kind == "block":
        algorithm = BlockLoadStoreAlgorithm(operation.algorithm.value)
        if (
            algorithm
            in {
                BlockLoadStoreAlgorithm.WARP_TRANSPOSE,
                BlockLoadStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
            }
            and block_threads % 32 != 0
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                f"cub::Block{operation.kind.value.title()} algorithm "
                f"{operation.algorithm.value!r} requires a block size that is "
                "a multiple of 32",
            )
        make_specialization = (
            make_block_load_specialization
            if operation.kind is GroupLoadStoreKind.LOAD
            else make_block_store_specialization
        )
        specialization = make_specialization(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            algorithm=algorithm,
            valid_items=operation.valid_items,
            oob_default=operation.oob_default,
            include_full_tile=False,
            include_pointer_offset=operation.offset,
        )
        target = GroupLoweringTarget.CUB_BLOCK
        header = f"cub/block/block_{operation.kind.value}.cuh"
    else:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        try:
            algorithm = WarpLoadStoreAlgorithm(operation.algorithm.value)
        except ValueError:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                f"cub::Warp{operation.kind.value.title()} does not support "
                f"algorithm {operation.algorithm.value!r}",
            )
        make_specialization = (
            make_warp_load_specialization
            if operation.kind is GroupLoadStoreKind.LOAD
            else make_warp_store_specialization
        )
        specialization = make_specialization(
            dtype=operation.dtype,
            items_per_thread=operation.items_per_thread,
            algorithm=algorithm,
            threads_in_warp=warp_width,
            valid_items=operation.valid_items,
            oob_default=operation.oob_default,
            include_full_tile=False,
            # Every physical or logical Warp group receives a consecutive tile.
            # The backend combines this runtime ABI argument with the preserved
            # user offset recorded on ``operation``.
            include_pointer_offset=ArgumentBinding.runtime(),
        )
        target = GroupLoweringTarget.CUB_WARP
        header = f"cub/warp/warp_{operation.kind.value}.cuh"
        group_instances = block_threads // warp_width
        maximum_tile_origin = (group_instances - 1) * tile_items
        if maximum_tile_origin > maximum_user_offset:
            raise ValueError(
                "warp-group tile origin must fit a signed 64-bit offset"
            )
        maximum_user_offset -= maximum_tile_origin
    if operation.offset.kind is BindingKind.STATIC and (
        int(operation.offset.value) > maximum_user_offset
    ):
        raise ValueError(
            "static offset plus the warp-group tile origin must fit a "
            "signed 64-bit integer"
        )
    cpp_class = f"cub::{specialization.struct_name}"
    storage_free = operation.algorithm in _STORAGE_FREE_ALGORITHMS
    requirements = _build_execution_requirements(
        resolved,
        launch,
        storage_ownership=(
            StorageOwnership.NONE
            if storage_free
            else operation.storage_ownership
        ),
        cpp_type=None,
        storage_sharing=None if storage_free else operation.storage_sharing,
        requested_size_in_bytes=(
            None if storage_free else operation.storage_size_in_bytes
        ),
        requested_alignment=(
            None if storage_free else operation.storage_alignment
        ),
        auto_sync=False if storage_free else operation.storage_auto_sync,
        uniform_arguments=(
            *(("valid_items",) if operation.has_valid_items else ()),
            *(("oob_default",) if operation.has_oob_default else ()),
            *(("offset",) if operation.has_offset else ()),
        ),
        valid_member_selection=(
            "first valid_items tile elements"
            if operation.has_valid_items
            else None
        ),
        argument_preconditions=(
            *(
                (
                    ArgumentPrecondition(
                        name="valid_items",
                        minimum=0,
                        maximum=tile_items,
                        enforcement=(
                            PreconditionEnforcement.PLANNER_VALIDATED
                            if operation.valid_items.kind is BindingKind.STATIC
                            else PreconditionEnforcement.CALLER
                        ),
                    ),
                )
                if operation.has_valid_items
                else ()
            ),
            *(
                (
                    ArgumentPrecondition(
                        name="offset",
                        minimum=0,
                        maximum=maximum_user_offset,
                        enforcement=(
                            PreconditionEnforcement.PLANNER_VALIDATED
                            if operation.offset.kind is BindingKind.STATIC
                            else PreconditionEnforcement.CALLER
                        ),
                    ),
                )
                if operation.has_offset
                else ()
            ),
        ),
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=specialization,
        topology=requirements.topology,
        participation=requirements.participation,
        result=None,
        synchronization=requirements.synchronization,
        temp_storage=requirements.temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=specialization.method_name,
        ),
    )


_register_group_operation_family(
    GroupLoadStoreSemantics,
    classifications=_call_classifications,
    planner=_plan_load_store,
    group_kinds=frozenset({"block", "warp", "threads_within_warp"}),
    unsupported_group_message=(
        "cuda.coop Load/Store supports this_block(), complete physical "
        "this_warp(), and power-of-two logical-warp groups"
    ),
)


__all__ = [
    "GroupLoadStoreAlgorithm",
    "GroupLoadStoreKind",
    "GroupLoadStoreSemantics",
]
