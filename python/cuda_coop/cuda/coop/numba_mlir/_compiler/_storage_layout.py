# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared scratch domains and placement for both Numba compiler pipelines.

The caller supplies validated requirements and core lowering contracts.
Layout never inspects or changes IR and performs no lifetime inference.
"""

from __future__ import annotations

from dataclasses import replace

from cuda.coop._core import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupTopologyRequirements,
    StorageOwnership,
    SynchronizationScope,
)

from ._rewrite_support import (
    _DEFAULT_STATIC_SHARED_MEMORY_BYTES,
    _MIN_TEMP_STORAGE_ALIGNMENT,
    CoopSinglePhaseRewriteError,
    _align_up,
    _TempStorageSlice,
    _TempStorageUseRequirement,
)


def apply_caller_storage(
    plan: GroupLoweringPlan,
    descriptor: tuple[int | None, int | None, bool, str],
    *,
    operation: str,
) -> GroupLoweringPlan:
    """Apply an explicit descriptor to a supported single-block plan.

    Preserve the provider's storage requirement while recording caller-owned
    capacity, alignment, sharing, and synchronization. Allocation is separate.
    """

    if plan.target is not GroupLoweringTarget.CUB_BLOCK:
        raise ValueError(
            f"cuda.coop.numba_mlir {operation} temp_storage applies only to "
            "block groups"
        )
    storage = plan.temp_storage
    synchronization = plan.synchronization
    assert storage is not None and synchronization is not None
    size_in_bytes, alignment, auto_sync, sharing = descriptor
    return replace(
        plan,
        temp_storage=replace(
            storage,
            ownership=StorageOwnership.CALLER,
            exact_layout_required=True,
            sharing=sharing,
            requested_size_in_bytes=size_in_bytes,
            requested_alignment=alignment,
            auto_sync=auto_sync,
        ),
        synchronization=replace(
            synchronization,
            storage_reuse_barrier=(
                SynchronizationScope.BLOCK
                if auto_sync
                else SynchronizationScope.NONE
            ),
        ),
    )


def get_device_shared_memory_limits(
    required_bytes: int, query
) -> tuple[int, int]:
    """Get the default and opt-in byte limits needed for this allocation.

    Requests up to the conservative static limit need no device query.
    Larger ``required_bytes`` values require valid limits from the active
    CUDA device; a failed query becomes a rewrite error.
    """

    conservative_default = _DEFAULT_STATIC_SHARED_MEMORY_BYTES
    if required_bytes <= conservative_default:
        return (conservative_default, conservative_default)
    try:
        limits = query()
        max_default = int(limits["max_default_shared_memory_per_block"])
        max_optin = int(limits["max_optin_shared_memory_per_block"])
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise CoopSinglePhaseRewriteError(
            "TempStorage requirements above the conservative 49152-byte "
            "static shared-memory limit require an exact current-device "
            "shared-memory query"
        ) from exc
    if max_default <= 0 or max_optin <= 0 or max_optin < max_default:
        raise CoopSinglePhaseRewriteError(
            "The current device reported invalid shared-memory limits: "
            f"default={max_default}, opt-in={max_optin}."
        )
    return (max_default, max_optin)


def temp_storage_domain_key(
    entry: _TempStorageUseRequirement,
) -> tuple[object, ...]:
    """Identify storage uses that may reuse one region.

    Legacy block providers and caller-owned storage use one domain,
    preserving the explicit reuse contract with typed reservations; its
    block-only restriction is checked by storage-plan validation.
    Implementation-owned storage instead partitions uses by group topology
    and reuse-barrier scope so incompatible group instances cannot alias.

    When a group has an execution scope but no reuse barrier, append the
    call order to its key. Such calls receive distinct domains even if
    their topologies match: completion before scratch reuse is uncertain.

    Parameters
    ----------
    entry : _TempStorageUseRequirement
        Primitive use with its lowering plan and scan-order position.

    Returns
    -------
    tuple of object
        Placement key for ``_layout_temp_storage_uses``.

    Raises
    ------
    CoopSinglePhaseRewriteError
        The lowering plan is unsupported, lacks storage contracts, or
        its storage instances disagree with its topology.
    """

    if entry.reservation:
        return ("caller-storage",)
    lowering_plan = entry.lowering_plan
    if lowering_plan is None:
        # Explicit descriptors and implicit scratch are laid out
        # separately. A legacy block provider using a descriptor shares
        # its region with that descriptor's other callers.
        return ("caller-storage",)
    if lowering_plan.unsupported is not None:
        raise CoopSinglePhaseRewriteError(
            "cooperative provider storage received an unsupported group "
            "lowering plan."
        )
    topology = lowering_plan.topology
    synchronization = lowering_plan.synchronization
    storage = lowering_plan.temp_storage
    if topology is None or synchronization is None or storage is None:
        raise CoopSinglePhaseRewriteError(
            "cooperative provider storage requires complete group "
            "topology, synchronization, and storage contracts."
        )
    if storage.ownership is StorageOwnership.NONE:
        raise CoopSinglePhaseRewriteError(
            "a storage-bearing cooperative provider received a "
            "storage-free lowering plan."
        )
    if storage.instances != topology.instances or (
        storage.instance_index != topology.instance_index
    ):
        raise CoopSinglePhaseRewriteError(
            "cooperative provider storage layout disagrees with its "
            "group topology."
        )
    if storage.ownership is StorageOwnership.CALLER:
        # Explicit single-block storage keeps the caller's reuse layout.
        # The backend separates topology domains only when it owns the
        # storage and controls every slice.
        return ("caller-storage",)
    domain = (
        topology.group_kind,
        topology.execution_scope.value,
        topology.logical_width,
        topology.instances,
        topology.instance_index,
        topology.thread_rank,
        synchronization.storage_reuse_barrier.value,
    )
    if (
        synchronization.storage_reuse_barrier is SynchronizationScope.NONE
        and topology.execution_scope is not SynchronizationScope.NONE
    ):
        return (*domain, "non-reusable", entry.order)
    return domain


def layout_temp_storage_uses(
    uses: list[_TempStorageUseRequirement],
    *,
    sharing: str,
) -> tuple[int, int, dict[int, _TempStorageSlice]]:
    """Lay out scratch for each call and group instance.

    Exclusive sharing gives each use a distinct domain. Shared placement
    reuses a domain only when ``_temp_storage_domain_key`` permits it;
    typed reservations join the descriptor's caller-owned scratch domain.
    Within a domain, reserve the largest per-instance requirement and
    align its stride for every consumer. Multiple group instances receive
    separate strides; compatible calls reuse those same instance slots.

    Offsets are relative to the region, before the function-wide backing's
    base offset is assigned. Views retain each call's actual byte count
    even when another call determines the larger shared stride. No
    lifetime or control-flow overlap analysis is performed here.

    Parameters
    ----------
    uses : list of _TempStorageUseRequirement
        Nonempty list of validated requirements and lowering plans.
    sharing : str
        Validated sharing policy: ``"exclusive"`` separates every
        call; ``"shared"`` permits reuse within compatible domains.

    Returns
    -------
    required_size : int
        Region size in bytes, including alignment gaps and all instances.
    required_alignment : int
        Maximum required alignment, at least that of the storage pointer.
    slices_by_call_id : dict of int to _TempStorageSlice
        Region-relative offset, byte count, instance stride, and lowering
        plan keyed by the object identity of each call assignment.
    """

    ordered_uses = sorted(
        uses, key=lambda entry: (entry.reservation, entry.order)
    )
    required_alignment = max(
        _MIN_TEMP_STORAGE_ALIGNMENT,
        *(max(1, int(entry.alignment)) for entry in ordered_uses),
    )
    domains: dict[tuple[object, ...], list[_TempStorageUseRequirement]] = {}
    for index, entry in enumerate(ordered_uses):
        domain_key = (
            ("exclusive", index)
            if sharing == "exclusive"
            else temp_storage_domain_key(entry)
        )
        domains.setdefault(domain_key, []).append(entry)

    required_size = 0
    slices_by_call_id: dict[int, _TempStorageSlice] = {}
    for domain_uses in domains.values():
        domain_alignment = max(
            _MIN_TEMP_STORAGE_ALIGNMENT,
            *(max(1, int(entry.alignment)) for entry in domain_uses),
        )
        per_instance_size = max(
            int(entry.size_in_bytes) for entry in domain_uses
        )
        instance_stride = _align_up(per_instance_size, domain_alignment)
        # Topology tells us how many independent groups need scratch: four
        # 32-thread warps in a 128-thread block need four separate slots.
        # Calls in a reuse domain agree on that count, so the first plan
        # is sufficient; an exclusive domain contains only one call.
        # Legacy block providers and typed reservations have no topology
        # and need one slot. The aligned instance_stride spaces groups'
        # slots apart, while compatible calls reuse the same set of slots.
        first_plan = domain_uses[0].lowering_plan
        instances = (
            1
            if first_plan is None or first_plan.topology is None
            else first_plan.topology.instances
        )
        required_size = _align_up(required_size, domain_alignment)
        domain_offset = required_size
        for entry in domain_uses:
            slices_by_call_id[id(entry.call_assign)] = _TempStorageSlice(
                offset=domain_offset,
                size_in_bytes=int(entry.size_in_bytes),
                stride=instance_stride,
                instances=instances,
                lowering_plan=entry.lowering_plan,
            )
        required_size += (
            per_instance_size if instances == 1 else instance_stride * instances
        )
    return required_size, required_alignment, slices_by_call_id


def validate_emittable_topology(
    lowering_plan: GroupLoweringPlan | None,
) -> GroupTopologyRequirements | None:
    """Require group rank formulas that the storage emitters support.

    A plan must cover every complete group in the exact block dimensions.
    Accept a single block, contiguous logical groups within each physical
    warp, or individual threads. Check the symbolic instance and rank
    expressions against those forms; emitters implement
    these specific formulas rather than evaluating arbitrary topology
    expression strings.

    Parameters
    ----------
    lowering_plan : GroupLoweringPlan or None
        Plan whose topology and participation requirements govern
        storage. None preserves the legacy block-provider path.

    Returns
    -------
    GroupTopologyRequirements or None
        Validated topology, or None when no plan was supplied.

    Raises
    ------
    CoopSinglePhaseRewriteError
        Dimensions or execution requirements are missing, coverage is
        inconsistent, or no emitter supports the scope and rank formulas.
    """

    if lowering_plan is None:
        return None
    topology = lowering_plan.topology
    participation = lowering_plan.participation
    if topology is None or participation is None:
        raise CoopSinglePhaseRewriteError(
            "cooperative provider storage requires group topology and "
            "participation contracts."
        )
    exact_block_dim = participation.exact_block_dim
    if exact_block_dim is None:
        raise CoopSinglePhaseRewriteError(
            "cooperative provider storage requires exact block dimensions."
        )
    block_threads = exact_block_dim[0] * exact_block_dim[1] * exact_block_dim[2]
    participating_threads = block_threads
    nonexhaustive_warp = (
        participation.group_kind == "threads_within_warp"
        and not participation.complete_parent_partition
        and 1 <= topology.logical_width <= 32
    )
    if nonexhaustive_warp:
        participating_threads = (
            (block_threads // 32)
            * (32 // topology.logical_width)
            * topology.logical_width
        )
    if topology.logical_width * topology.instances != participating_threads:
        raise CoopSinglePhaseRewriteError(
            "cooperative provider topology does not cover the exact "
            "block dimensions."
        )
    scope = topology.execution_scope
    if scope is SynchronizationScope.BLOCK:
        if (
            topology.instances != 1
            or topology.logical_width != block_threads
            or topology.instance_index != "cta"
            or topology.thread_rank != "linear_thread_rank"
        ):
            raise CoopSinglePhaseRewriteError(
                "block-scoped cooperative storage requires canonical "
                "single-CTA ranks."
            )
    elif scope is SynchronizationScope.WARP:
        width = topology.logical_width
        if (
            width < 1
            or width > 32
            or (width & (width - 1) and block_threads % 32 != 0)
        ):
            raise CoopSinglePhaseRewriteError(
                "warp-scoped cooperative storage requires a logical width "
                "from 1 through 32; non-power-of-two widths require "
                "complete physical warps."
            )
        if nonexhaustive_warp:
            instance_index = (
                f"(linear_thread_rank / 32) * {32 // width} + "
                f"((linear_thread_rank % 32) / {width})"
            )
            thread_rank = f"(linear_thread_rank % 32) % {width}"
        else:
            instance_index = f"linear_thread_rank / {width}"
            thread_rank = f"linear_thread_rank % {width}"
        if (
            topology.instance_index != instance_index
            or topology.thread_rank != thread_rank
        ):
            raise CoopSinglePhaseRewriteError(
                "warp-scoped cooperative storage requires canonical "
                "ranks within each physical warp."
            )
    elif scope is SynchronizationScope.NONE:
        if (
            topology.logical_width != 1
            or topology.instance_index != "linear_thread_rank"
            or topology.thread_rank != "0"
        ):
            raise CoopSinglePhaseRewriteError(
                "thread-scoped cooperative storage requires canonical "
                "per-thread ranks."
            )
    else:
        raise CoopSinglePhaseRewriteError(
            "cuda.coop.numba_mlir provider execution scope "
            f"{scope.value!r} has no storage emitter"
        )
    return topology


__all__ = [
    "apply_caller_storage",
    "get_device_shared_memory_limits",
    "layout_temp_storage_uses",
    "temp_storage_domain_key",
    "validate_emittable_topology",
]
