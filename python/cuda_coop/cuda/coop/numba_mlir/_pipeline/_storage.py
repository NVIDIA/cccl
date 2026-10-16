# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Size each site's scratch and lay it out in shared memory.

Storage analysis compiles the kernel's operation providers in one bundle
(``_link``), which also reports each provider's scratch size and alignment.
Shared layout policy assigns descriptor and group-instance regions in one
backing. Allocations above the static limit use the launch's dynamic window
and publish the required byte count to the compiler and dispatcher.

Which sites take scratch was decided before the compile, from each
provider's storage ABI; the compiled layouts must agree. ``layout`` is pure
and takes plain sizes. This stage does not change the IR.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..._core import StorageOwnership, SynchronizationScope
from .._compiler._rewrite_support import (
    _DYNAMIC_SHARED_MEMORY_ALIGNMENT,
    _MIN_TEMP_STORAGE_ALIGNMENT,
    CoopSinglePhaseRewriteError,
    _align_up,
    _query_device_shared_memory_limits,
    _TempStorageUseRequirement,
)
from .._compiler._storage_layout import (
    get_device_shared_memory_limits,
    layout_temp_storage_uses,
    validate_emittable_topology,
)
from . import _link
from ._errors import TypedPipelineError
from ._plan import Backing, Region
from ._specialize import Specialization
from ._temp_storage import Descriptors

STATIC_SHARED_LIMIT = 48 * 1024
IMPLICIT_REGION = "implicit.0"
_BACKING_ALIGNMENT = 16


@dataclass(frozen=True)
class ScratchSlice:
    offset: int
    size: int
    alignment: int
    stride: int
    instances: int
    lowering_plan: Any = None


@dataclass(frozen=True)
class StorageLayout:
    """One backing, descriptor regions and each consumer's byte view."""

    backing: Backing | None
    regions: tuple[Region, ...]
    slices: dict[str, ScratchSlice]
    scratch: tuple[bool, ...]
    barrier_scopes: tuple[SynchronizationScope, ...]
    reservations: dict = field(default_factory=dict)
    required_dynamic_bytes: int = 0

    @property
    def region(self):
        return self.regions[0] if self.regions else None


def layout(
    needs: Sequence[tuple[int, int] | None], user_shared_bytes: int
) -> StorageLayout:
    """Lay out the sites' scratch ``needs``, one ``(size, alignment)`` each.

    A site without scratch needs ``None``. Every site shares one region, so
    the region is as large as the largest need, not their sum.
    """

    scratch = tuple(need is not None for need in needs)
    used = [need for need in needs if need is not None]
    if not used:
        return StorageLayout(
            None,
            (),
            {},
            scratch,
            tuple(SynchronizationScope.NONE for _ in scratch),
        )
    size = max(s for s, _ in used)
    alignment = max(a for _, a in used)
    backing_alignment = max(_BACKING_ALIGNMENT, alignment)
    backing_size = -(-size // backing_alignment) * backing_alignment
    if backing_size + user_shared_bytes > STATIC_SHARED_LIMIT:
        raise TypedPipelineError(
            f"cooperative scratch needs {backing_size} bytes of shared "
            f"memory, which with the kernel's own {user_shared_bytes} bytes "
            "exceeds "
            f"the {STATIC_SHARED_LIMIT}-byte static limit. Scratch above "
            "48 KB is not supported on the typed pipeline; use a smaller "
            "block, fewer items per thread, or a storage-free algorithm "
            "(direct, striped, vectorize)."
        )
    return StorageLayout(
        Backing("static", backing_size, backing_alignment),
        (Region(IMPLICIT_REGION, 0, size, alignment),),
        {},
        scratch,
        tuple(
            SynchronizationScope.BLOCK if used else SynchronizationScope.NONE
            for used in scratch
        ),
    )


def _whole_kernel_layout(specialization, sites, descriptors, needs, user_bytes):
    """Apply shared descriptor/domain rules to typed whole-kernel uses."""
    from .._compiler._operations import (
        expected_storage_reuse_barrier,
        provider_synchronization_matches,
    )

    uses = {}
    scopes = []
    for order, (site, need, plan, algorithm) in enumerate(
        zip(
            sites,
            needs,
            specialization.lowering_plans,
            specialization.site_algorithms,
            strict=True,
        )
    ):
        if need is None:
            scopes.append(SynchronizationScope.NONE)
            continue
        topology, storage, synchronization = (
            plan.topology,
            plan.temp_storage,
            plan.synchronization,
        )
        validate_emittable_topology(plan)
        if topology is None or storage is None or synchronization is None:
            raise TypedPipelineError(
                "cooperative storage needs complete group contracts"
            )
        storage_var = dict(site.stmt.value.kws).get("temp_storage")
        owner = (
            None
            if storage_var is None
            else descriptors.owners.get(storage_var.name)
        )
        caller = storage.ownership is StorageOwnership.CALLER
        if caller != (owner is not None):
            raise TypedPipelineError(
                "TempStorage ownership disagrees with its typed descriptor"
            )
        if caller and (
            topology.execution_scope is not SynchronizationScope.BLOCK
            or topology.instances != 1
        ):
            raise TypedPipelineError(
                "caller-owned TempStorage is supported only for "
                "single-instance block-scoped cooperative primitives"
            )
        if (
            storage.address_space != "shared"
            or algorithm.execution_scope is not topology.execution_scope
        ):
            raise TypedPipelineError(
                "cooperative storage scope or address space "
                "disagrees with its provider"
            )
        expected = expected_storage_reuse_barrier(topology, storage)
        if (
            synchronization.storage_reuse_barrier is not expected
            or not provider_synchronization_matches(
                algorithm, topology, synchronization, storage
            )
        ):
            raise TypedPipelineError(
                "cooperative storage synchronization "
                "disagrees with its provider"
            )
        scopes.append(expected)
        uses.setdefault(owner, []).append(
            _TempStorageUseRequirement(
                site.stmt,
                order,
                need[0],
                need[1],
                lowering_plan=plan,
            )
        )
    for order, reservation in enumerate(descriptors.reservations.values()):
        uses.setdefault(reservation.owner, []).append(
            _TempStorageUseRequirement(
                reservation.stmt,
                order,
                reservation.size_in_bytes,
                reservation.alignment,
                reservation=True,
            )
        )
    offset, maximum = 0, _MIN_TEMP_STORAGE_ALIGNMENT
    regions, slices = [], {}
    ordered = [owner for owner in descriptors.constructors if owner in uses]
    if None in uses:
        ordered.append(None)
    for owner in ordered:
        descriptor = descriptors.constructors.get(owner)
        size, alignment, planned = layout_temp_storage_uses(
            uses[owner],
            sharing=descriptor.sharing if descriptor else "shared",
        )
        if descriptor is not None:
            if descriptor.size_in_bytes is not None:
                if descriptor.size_in_bytes < size:
                    raise TypedPipelineError(
                        "TempStorage size_in_bytes is smaller than required "
                        "by primitive and reserve uses "
                        f"({descriptor.size_in_bytes} < {size})"
                    )
                size = descriptor.size_in_bytes
            alignment = max(alignment, descriptor.alignment or 1)
        offset = _align_up(offset, alignment)
        name = (
            IMPLICIT_REGION if owner is None else f"descriptor.{len(regions)}"
        )
        regions.append(Region(name, offset, size, alignment))
        for use in uses[owner]:
            region = planned[id(use.call_assign)]
            slices[use.call_assign.target.name] = ScratchSlice(
                offset + region.offset,
                region.size_in_bytes,
                use.alignment,
                region.stride or region.size_in_bytes,
                region.instances,
                region.lowering_plan,
            )
        offset += size
        maximum = max(maximum, alignment)
    if not regions:
        return StorageLayout(
            None, (), {}, specialization.scratch, tuple(scopes)
        )
    total = _align_up(offset, maximum)
    if (
        total <= STATIC_SHARED_LIMIT
        and total + user_bytes > STATIC_SHARED_LIMIT
    ):
        raise TypedPipelineError(
            f"cooperative scratch needs {total} bytes of shared memory, "
            f"which with the kernel's own {user_bytes} bytes exceeds "
            f"the {STATIC_SHARED_LIMIT}-byte static limit. The supported "
            "compiler cannot combine cooperative dynamic backing with user "
            "shared allocations."
        )
    try:
        default, max_optin = get_device_shared_memory_limits(
            total, _query_device_shared_memory_limits
        )
    except CoopSinglePhaseRewriteError as exc:
        raise TypedPipelineError(str(exc)) from exc
    dynamic = total > default
    if dynamic:
        if maximum > _DYNAMIC_SHARED_MEMORY_ALIGNMENT:
            raise TypedPipelineError(
                f"TempStorage requires {maximum}-byte alignment, but "
                f"the {total}-byte backing exceeds the {default}-byte "
                "static shared-memory limit and dynamic shared memory "
                "guarantees only "
                f"{_DYNAMIC_SHARED_MEMORY_ALIGNMENT}-byte alignment"
            )
        if total > max_optin:
            raise TypedPipelineError(
                f"TempStorage requires {total} bytes dynamic shared memory, "
                f"but device max opt-in is {max_optin} bytes."
            )
        if user_bytes:
            raise TypedPipelineError(
                f"cooperative scratch needs a {total}-byte dynamic "
                "shared-memory backing, but this kernel also declares "
                "static cuda.shared.array allocations. The supported "
                "compiler does not separate these allocations; they would "
                "alias. Move the user data out of shared memory."
            )
    return StorageLayout(
        Backing("dynamic" if dynamic else "static", total, maximum),
        tuple(regions),
        slices,
        specialization.scratch,
        tuple(scopes),
        descriptors.reservations,
        total if dynamic else 0,
    )


def analyze(
    specialization: Specialization,
    user_shared_bytes: int,
    identity: Any,
    context: Any,
    *,
    user_dynamic_shared_at: str | None = None,
    sites=None,
    descriptors: Descriptors | None = None,
) -> tuple[StorageLayout, _link.Bundle]:
    """Compile the specialization's bundle and lay out its scratch."""

    if user_dynamic_shared_at is not None and (
        any(specialization.scratch)
        or (descriptors is not None and descriptors.reservations)
    ):
        raise TypedPipelineError(
            "cooperative scratch requires its own shared-memory backing, but "
            "this kernel also declares dynamic/runtime-sized cuda.shared.array "
            f"at {user_dynamic_shared_at}. The supported compiler does not "
            "separate these allocations; they would alias. Use statically "
            "sized user shared arrays or a storage-free algorithm."
        )
    bundle = (
        _link.compile_bundle(specialization.algorithms, identity, context)
        if specialization.algorithms
        else _link.Bundle((), b"", {}, 0)
    )
    needs = []
    for algorithm, scratch in zip(
        specialization.site_algorithms, specialization.scratch, strict=True
    ):
        need = bundle.layouts[algorithm._private_symbol_digest]
        if (need is not None) != scratch:
            raise AssertionError(
                f"{algorithm.c_name}: its {algorithm.storage_abi} storage "
                f"ABI disagrees with its compiled scratch layout {need}"
            )
        needs.append(need)
    if sites is None:
        return layout(needs, user_shared_bytes), bundle
    return _whole_kernel_layout(
        specialization,
        sites,
        descriptors or Descriptors(),
        needs,
        user_shared_bytes,
    ), bundle


__all__ = [
    "IMPLICIT_REGION",
    "STATIC_SHARED_LIMIT",
    "StorageLayout",
    "analyze",
    "layout",
]
