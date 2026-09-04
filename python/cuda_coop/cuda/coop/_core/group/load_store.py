# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load/store semantics and block, physical-warp, or logical-warp lowering.

This module owns portable load/store algorithm normalization and selects the
corresponding CUB specialization after group resolution. It does not own
ThreadData allocation, backend activation, or compiler rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any

from .._bindings import (
    ArgumentBinding,
    BindingKind,
    _normalize_i32_binding,
    _normalize_i64_binding,
)
from .._symbols import semantic_token
from .._types import ArgumentKind, ParameterClassification, ParameterRole
from ..block.load_store import (
    BlockLoadStoreAlgorithm,
    make_block_load_spec,
    make_block_store_spec,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ..warp.load_store import (
    WarpLoadStoreAlgorithm,
    make_warp_load_spec,
    make_warp_store_spec,
)
from ._contracts import _contracts, _unsupported, _unsupported_cub_warp_width
from ._dispatch import _register_group_operation_family
from ._model import (
    ArgumentPrecondition,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    PreconditionEnforcement,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    UnsupportedReasonCode,
)


class GroupLoadStoreKind(str, Enum):
    LOAD = "load"
    STORE = "store"


class GroupLoadStoreAlgorithm(str, Enum):
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
    kind: GroupLoadStoreKind
    dtype: Any
    items_per_thread: int
    algorithm: GroupLoadStoreAlgorithm = GroupLoadStoreAlgorithm.DIRECT
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    oob_default: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    offset: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    storage_ownership: StorageOwnership = StorageOwnership.IMPLEMENTATION
    storage_sharing: str | None = None
    storage_size_in_bytes: int | None = None
    storage_alignment: int | None = None
    storage_auto_sync: bool = True

    def __post_init__(self) -> None:
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
            _normalize_i32_binding(self.valid_items, name="valid_items"),
        )
        object.__setattr__(
            self,
            "offset",
            _normalize_i64_binding(self.offset, name="offset"),
        )
        if self.offset.kind is BindingKind.STATIC and int(self.offset.value) < 0:
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
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(self.storage_auto_sync, bool):
            raise TypeError("storage_auto_sync must be a bool")
        if self.storage_sharing == "exclusive" and self.storage_auto_sync:
            raise ValueError("exclusive storage cannot request automatic sync")
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
                raise ValueError("storage-free operations cannot carry storage layout")
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
        return self.kind is GroupLoadStoreKind.LOAD

    @property
    def semantic_key(self) -> tuple[Any, ...]:
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
    classifications = [
        ParameterClassification(
            "source" if operation.kind is GroupLoadStoreKind.LOAD else "destination",
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
        if isinstance(valid_items, bool) or not isinstance(valid_items, Integral):
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
        make_spec = (
            make_block_load_spec
            if operation.kind is GroupLoadStoreKind.LOAD
            else make_block_store_spec
        )
        spec = make_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            algorithm=algorithm,
            valid_items=operation.valid_items,
            oob_default=operation.oob_default,
            include_full_tile=False,
            include_pointer_offset=operation.offset,
        ).specialization
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
        make_spec = (
            make_warp_load_spec
            if operation.kind is GroupLoadStoreKind.LOAD
            else make_warp_store_spec
        )
        spec = make_spec(
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
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        header = f"cub/warp/warp_{operation.kind.value}.cuh"
        group_instances = block_threads // warp_width
        maximum_tile_origin = (group_instances - 1) * tile_items
        if maximum_tile_origin > maximum_user_offset:
            raise ValueError("warp-group tile origin must fit a signed 64-bit offset")
        maximum_user_offset -= maximum_tile_origin
    if operation.offset.kind is BindingKind.STATIC and (
        int(operation.offset.value) > maximum_user_offset
    ):
        raise ValueError(
            "static offset plus the warp-group tile origin must fit a "
            "signed 64-bit integer"
        )
    cpp_class = f"cub::{spec.struct_name}"
    result = None
    if operation.kind is GroupLoadStoreKind.LOAD:
        result = ResultContract(
            (
                LogicalResultContract(
                    name="value",
                    dtype=operation.dtype,
                    visibility=ResultVisibility.PER_MEMBER,
                    ownership=ResultOwnership.EACH_MEMBER,
                    operand_kind=GroupOperandKind.ARRAY,
                    items_per_member=operation.items_per_thread,
                ),
            )
        )
    storage_free = operation.algorithm in _STORAGE_FREE_ALGORITHMS
    contracts = _contracts(
        resolved,
        launch,
        result=result,
        storage_ownership=(
            StorageOwnership.NONE if storage_free else operation.storage_ownership
        ),
        cpp_type=None,
        storage_sharing=None if storage_free else operation.storage_sharing,
        requested_size_in_bytes=(
            None if storage_free else operation.storage_size_in_bytes
        ),
        requested_alignment=(None if storage_free else operation.storage_alignment),
        auto_sync=False if storage_free else operation.storage_auto_sync,
        uniform_arguments=(
            *(("valid_items",) if operation.has_valid_items else ()),
            *(("oob_default",) if operation.has_oob_default else ()),
            *(("offset",) if operation.has_offset else ()),
        ),
        valid_member_selection=(
            "first valid_items tile elements" if operation.has_valid_items else None
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
        implementation=spec,
        topology=contracts[0],
        participation=contracts[1],
        result=result,
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
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
