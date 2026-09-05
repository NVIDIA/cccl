# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load/store semantics and block or warp lowering.

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
    LOAD = "load"
    STORE = "store"


class GroupLoadStoreAlgorithm(str, Enum):
    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"
    WARP_TRANSPOSE = "warp_transpose"
    WARP_TRANSPOSE_TIMESLICED = "warp_transpose_timesliced"


@dataclass(frozen=True, eq=False)
class GroupLoadStoreSemantics:
    kind: GroupLoadStoreKind
    dtype: Any
    items_per_thread: int
    algorithm: GroupLoadStoreAlgorithm = GroupLoadStoreAlgorithm.DIRECT
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    oob_default: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    offset: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GroupLoadStoreKind(self.kind))
        object.__setattr__(
            self,
            "algorithm",
            GroupLoadStoreAlgorithm(self.algorithm),
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
        if self.kind is GroupLoadStoreKind.STORE and (
            self.oob_default.kind is not BindingKind.OMITTED
        ):
            raise ValueError("oob_default is valid only for group load")
        if (
            self.oob_default.kind is not BindingKind.OMITTED
            and self.valid_items.kind is BindingKind.OMITTED
        ):
            raise ValueError("oob_default requires valid_items")

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
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            f"group_{self.kind.value}",
            semantic_token(self.dtype),
            self.items_per_thread,
            self.algorithm.value,
            self.valid_items.semantic_key,
            self.oob_default.semantic_key,
            self.offset.semantic_key,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupLoadStoreSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


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
            "group load/store supports complete physical block, physical-warp, "
            "and logical-warp groups",
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
    if resolved.kind == "block":
        algorithm = BlockLoadStoreAlgorithm(operation.algorithm.value)
        block_threads = launch.exact_block_threads
        assert block_threads is not None
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
        cpp_class = (
            "cub::BlockLoad"
            if operation.kind is GroupLoadStoreKind.LOAD
            else "cub::BlockStore"
        )
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
            threads_in_warp=warp_width,
            algorithm=algorithm,
            valid_items=operation.valid_items,
            oob_default=operation.oob_default,
            include_full_tile=False,
            include_pointer_offset=operation.offset,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = (
            "cub::WarpLoad"
            if operation.kind is GroupLoadStoreKind.LOAD
            else "cub::WarpStore"
        )
        header = f"cub/warp/warp_{operation.kind.value}.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *(("valid_items",) if operation.has_valid_items else ()),
            *(("oob_default",) if operation.has_oob_default else ()),
            *(("offset",) if operation.has_offset else ()),
        ),
        valid_member_selection=(
            "first valid_items tile elements" if operation.has_valid_items else None
        ),
        argument_preconditions=(
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
        returns_value=operation.kind is GroupLoadStoreKind.LOAD,
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


__all__ = [
    "GroupLoadStoreAlgorithm",
    "GroupLoadStoreKind",
    "GroupLoadStoreSemantics",
]
