# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Allocate shared scratch after the C++ layouts become known.

Tracing records each scratch-using call with placeholder address and size
operands. Finalization obtains exact layouts from the provider compilation,
plans each kernel's allocations, then replaces those placeholders in MLIR.
Descriptor identity controls reuse; equal descriptor fields alone do not
make two descriptors share an allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import Any, NoReturn

from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.typing import (
    Int32,
    Uint32,
)

from ._state import _SESSION_SCOPE, BundleSession, active_bundle_session
from ._types import (
    DeferredTempStorageBinding,
    DeferredTempStorageEvent,
    DeferredTempStoragePlan,
    ScratchLayout,
)


def _deferred_temp_storage_capability_error(
    cause: Exception | None = None,
) -> NoReturn:
    raise DSLRuntimeError(
        f"{_SESSION_SCOPE} deferred TempStorage requires "
        "a CUTLASS DSL with trace finalization, SmemAllocator, and MLIR value "
        "replacement support. Install a compatible CUTLASS DSL runtime.",
        cause=cause,
    )


def _active_cuda_kernel_op() -> Any:
    """Find the kernel that must own scratch for the active insertion point.

    Calls can occur in nested regions, so walk outward to ``cuda.kernel`` or
    a ``lir.func`` marked ``gpu.kernel`` directly inside ``gpu.module``. A
    device helper or a function with launch attributes alone is not enough:
    scratch planning and allocation must be attached to the enclosing kernel.
    """

    try:
        from cutlass._mlir import ir

        current_ip = ir.InsertionPoint.current
    except Exception as exc:
        raise DSLRuntimeError(
            f"{_SESSION_SCOPE} deferred TempStorage requires an active CuTe "
            "kernel trace."
        ) from exc

    if current_ip is None or current_ip.block is None:
        raise DSLRuntimeError(
            f"{_SESSION_SCOPE} deferred TempStorage requires an active CuTe "
            "kernel trace."
        )

    op = current_ip.block.owner
    while op is not None:
        operation = getattr(op, "operation", op)
        if getattr(operation, "name", None) == "cuda.kernel":
            return operation
        if (
            getattr(operation, "name", None) == "lir.func"
            and "gpu.kernel" in operation.attributes
            and getattr(getattr(operation, "parent", None), "name", None)
            == "gpu.module"
        ):
            return operation
        op = getattr(op, "parent_op", None) or getattr(op, "parent", None)

    raise DSLRuntimeError(
        f"{_SESSION_SCOPE} deferred TempStorage could not find the enclosing "
        "CUDA kernel operation."
    )


def _cuda_kernel_name(kernel_op: Any) -> str:
    try:
        return str(kernel_op.attributes["sym_name"])
    except Exception:  # noqa: BLE001
        # A diagnostic name must not break scratch planning.
        return f"cuda.kernel@{id(kernel_op):x}"


def _fresh_i32_placeholder() -> Any:
    """Emit a distinct operand for replacement during finalization."""

    from cutlass._mlir.dialects import arith
    from cutlass.cutlass_dsl import T

    return arith.constant(T.i32(), 0)


def register_deferred_temp_storage_event(
    temp_storage: Any,
    *,
    primitive_name: str,
    requirement_key: Hashable,
    reservation_layout: ScratchLayout | None = None,
    active_session_getter: Callable[[], BundleSession] = active_bundle_session,
) -> tuple[Any, Any, Any]:
    """Record one traced scratch use before its C++ layout is available.

    Each call gets fresh address and size placeholders. The session preserves
    call order because exclusive storage assigns a slice to each traced use.
    A runtime loop reuses those operands on each iteration.

    Parameters
    ----------
    temp_storage : TempStorage
        Descriptor whose identity and current policies are recorded.
    primitive_name : str
        Operation name for diagnostics.
    requirement_key : hashable
        Key that matches this call to its compiled scratch-layout probe.
    reservation_layout : ScratchLayout or None, optional
        Known size and alignment for a typed reservation. Primitive calls
        leave this unset and obtain their layout from the C++ probe.
    active_session_getter : callable, optional
        Supplies the active bundle session; tests can substitute a session.

    Returns
    -------
    tuple
        Provider operands: a Uint32 shared-memory address, an Int32 capacity,
        and an Int32 automatic-barrier flag. Finalization replaces the first
        two; the flag is fixed by the descriptor's ``auto_sync`` policy.

    Raises
    ------
    TypeError
        ``requirement_key`` is not hashable.
    DSLRuntimeError
        No kernel trace is active, or no enclosing kernel owns the call.
    """

    try:
        hash(requirement_key)
    except TypeError as exc:
        raise TypeError("scratch requirement keys must be hashable") from exc

    session = active_session_getter()
    kernel_op = _active_cuda_kernel_op()
    smem_addr_placeholder = _fresh_i32_placeholder()
    size_placeholder = _fresh_i32_placeholder()
    try:
        location = str(smem_addr_placeholder.owner.location)
    except Exception:  # noqa: BLE001
        # Missing diagnostic locations do not invalidate scratch.
        location = "unknown location"

    session.add_deferred_temp_storage_event(
        DeferredTempStorageEvent(
            kernel_op=kernel_op,
            kernel_name=_cuda_kernel_name(kernel_op),
            temp_storage=temp_storage,
            primitive_name=primitive_name,
            requirement_key=requirement_key,
            sharing=temp_storage.sharing,
            auto_sync=temp_storage.auto_sync,
            capacity_size_in_bytes=temp_storage.capacity_size_in_bytes,
            capacity_alignment=temp_storage.alignment,
            smem_addr_placeholder=smem_addr_placeholder,
            size_placeholder=size_placeholder,
            location=location,
            reservation_layout=reservation_layout,
        )
    )
    return (
        Uint32(smem_addr_placeholder),
        Int32(size_placeholder),
        Int32(1 if temp_storage.auto_sync else 0),
    )


def reserve_temp_storage_tensor(
    temp_storage: Any,
    num_elems: Any,
    dtype: Any,
    *,
    alignment: Any = None,
) -> Any:
    """Expose a typed shared view whose address is resolved at finalization."""

    import numpy as np
    from cutlass import BFloat16, Float16, cute

    from ..._core.api._payload import _normalize_alignment
    from .._thread_data import _normalize_index_int
    from ._types import ALL_PROVIDER_TYPES, canonical_dsl_type

    count = (
        None
        if isinstance(num_elems, np.bool_)
        else _normalize_index_int(num_elems)
    )
    if count is None:
        raise TypeError(
            "TempStorage.reserve num_elems must be a compile-time integer."
        )
    if count <= 0:
        raise ValueError("TempStorage.reserve num_elems must be positive.")
    if not isinstance(dtype, (type, np.dtype)):
        raise TypeError(
            "TempStorage.reserve dtype must be a compile-time scalar type."
        )
    value_type = canonical_dsl_type(dtype)
    if dtype is np.float16 or dtype == np.dtype(np.float16):
        value_type = Float16
    if dtype in (Float16, BFloat16):
        value_type = dtype
    if value_type not in ALL_PROVIDER_TYPES | {Float16, BFloat16}:
        raise TypeError(
            "TempStorage.reserve dtype must be a supported fixed-width "
            "integer or floating-point scalar type."
        )
    itemsize = value_type.width // 8
    alignment = max(itemsize, _normalize_alignment(alignment) or 1)
    if temp_storage.auto_sync:
        raise ValueError(
            "TempStorage.reserve requires auto_sync=False; callers must "
            "synchronize reserved arrays and complete library operations "
            "before reuse."
        )
    address, _, _ = register_deferred_temp_storage_event(
        temp_storage,
        primitive_name="reserve",
        requirement_key=None,
        reservation_layout=ScratchLayout(count * itemsize, alignment),
    )
    pointer = cute.make_ptr(
        value_type,
        address,
        cute.AddressSpace.smem,
        assumed_align=alignment,
    )
    return cute.make_tensor(pointer, cute.make_layout(count))


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def plan_deferred_temp_storage_events(
    events: list[DeferredTempStorageEvent],
    layouts: Mapping[Hashable, ScratchLayout],
) -> tuple[DeferredTempStoragePlan, ...]:
    """Resolve kernel-local allocations without changing MLIR.

    Events are grouped by kernel and descriptor identity. Shared storage uses
    one slice large enough for every use. Exclusive storage assigns an aligned
    slice to each event in trace order. Sharing selects the layout, not
    synchronization. Unless ``auto_sync`` is true, the kernel must synchronize
    before a shared slice serves another call and before any call site reuses
    its own slice.

    Parameters
    ----------
    events : list of DeferredTempStorageEvent
        Traced uses with captured descriptor policies and ABI placeholders.
    layouts : mapping
        Exact C++ byte layouts, indexed by each event's requirement key.

    Returns
    -------
    tuple of DeferredTempStoragePlan
        Allocations and per-call slices. Explicit capacity must cover the
        whole plan. Allocation alignment satisfies both the descriptor's
        requested minimum and every operation's alignment.

    Raises
    ------
    DSLRuntimeError
        Uses of one descriptor disagree on its configuration, a layout is
        missing, or explicit capacity is too small.
    """

    grouped: dict[tuple[Any, int], list[DeferredTempStorageEvent]] = {}
    for event in events:
        grouped.setdefault(
            (event.kernel_op, id(event.temp_storage)),
            [],
        ).append(event)

    plans: list[DeferredTempStoragePlan] = []
    for group_events in grouped.values():
        first = group_events[0]
        if any(
            event.reservation_layout is not None for event in group_events
        ) and any(event.auto_sync for event in group_events):
            raise DSLRuntimeError(
                "TempStorage.reserve requires auto_sync=False for every "
                "use of its descriptor."
            )
        for event in group_events[1:]:
            if (
                event.sharing != first.sharing
                or event.auto_sync != first.auto_sync
                or event.capacity_size_in_bytes != first.capacity_size_in_bytes
                or event.capacity_alignment != first.capacity_alignment
            ):
                raise DSLRuntimeError(
                    "Deferred TempStorage configuration changed during tracing "
                    f"for {first.kernel_name} ({event.location})."
                )
        event_layouts = []
        for event in group_events:
            layout = event.reservation_layout or layouts.get(
                event.requirement_key
            )
            if layout is None:
                raise DSLRuntimeError(
                    "No exact C++ scratch layout was registered for "
                    f"{event.primitive_name} ({event.location})."
                )
            event_layouts.append((event, layout))

        required_plan_alignment = max(
            layout.alignment for _, layout in event_layouts
        )
        if first.sharing == "shared":
            planned_size = max(
                layout.size_in_bytes for _, layout in event_layouts
            )
            planned_bindings = tuple(
                (event, 0, planned_size, required_plan_alignment)
                for event, _ in event_layouts
            )
        else:
            planned_size = 0
            exclusive_bindings = []
            for event, layout in event_layouts:
                offset = _align_up(planned_size, layout.alignment)
                exclusive_bindings.append(
                    (event, offset, layout.size_in_bytes, layout.alignment)
                )
                planned_size = offset + layout.size_in_bytes
            planned_bindings = tuple(exclusive_bindings)

        capacity_size = first.capacity_size_in_bytes
        if capacity_size is not None and capacity_size < planned_size:
            raise DSLRuntimeError(
                "Deferred TempStorage capacity is smaller than "
                "its resolved plan "
                f"in {first.kernel_name} ({capacity_size} < {planned_size})."
            )
        resolved_size_in_bytes = (
            capacity_size if capacity_size is not None else planned_size
        )
        plan_alignment = max(
            first.capacity_alignment or 1, required_plan_alignment
        )

        bindings = tuple(
            DeferredTempStorageBinding(
                event=event,
                byte_offset_in_bytes=offset,
                size_in_bytes=(
                    binding_size_in_bytes
                    if first.sharing == "exclusive"
                    else resolved_size_in_bytes
                ),
                alignment=(
                    binding_alignment
                    if first.sharing == "exclusive"
                    else plan_alignment
                ),
            )
            for event, offset, binding_size_in_bytes, binding_alignment in (
                planned_bindings
            )
        )
        plans.append(
            DeferredTempStoragePlan(
                kernel_op=first.kernel_op,
                kernel_name=first.kernel_name,
                temp_storage=first.temp_storage,
                size_in_bytes=resolved_size_in_bytes,
                alignment=plan_alignment,
                bindings=bindings,
            )
        )
    return tuple(plans)


def _replace_all_uses(old_value: Any, new_value: Any) -> None:
    for method_name in ("replace_all_uses_with", "replaceAllUsesWith"):
        replace = getattr(old_value, method_name, None)
        if replace is None:
            continue
        try:
            replace(new_value)
            return
        except Exception:  # noqa: BLE001, S112
            # Try the alternate MLIR spelling before reporting incompatibility.
            continue
    _deferred_temp_storage_capability_error()


def materialize_deferred_temp_storage_plans(
    plans: tuple[DeferredTempStoragePlan, ...],
) -> None:
    """Allocate each plan at kernel entry and replace its call operands.

    ``plans`` comes from :func:`plan_deferred_temp_storage_events`. One
    ``SmemAllocator`` per kernel places all of that kernel's allocations.
    Each binding supplies its slice's shared-memory address and capacity to
    the recorded call, including calls nested in loops or conditional regions.
    This step emits no barriers. The provider wrapper adds a trailing barrier
    when the call's ``auto_sync`` flag is set; otherwise the kernel must
    synchronize before reuse.

    The function returns ``None``. Missing allocation or value-replacement
    support raises ``DSLRuntimeError``. Capabilities are checked before
    allocation. A later replacement failure does not roll back MLIR.
    """

    if not plans:
        return

    try:
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import arith, llvm
        from cutlass.cute.typing import Pointer
        from cutlass.cutlass_dsl import T
        from cutlass.memory import SmemAllocator
    except (AttributeError, ImportError) as exc:
        _deferred_temp_storage_capability_error(exc)

    required_capabilities = (
        getattr(ir.InsertionPoint, "at_block_begin", None),
        getattr(SmemAllocator, "allocate", None),
        getattr(Pointer, "to_llvm_ptr", None),
        getattr(arith, "constant", None),
        getattr(arith, "addi", None),
        getattr(llvm, "ptrtoint", None),
    )
    if not all(callable(capability) for capability in required_capabilities):
        _deferred_temp_storage_capability_error()

    kernel_groups: dict[Any, tuple[Any, list[DeferredTempStoragePlan]]] = {}
    for plan in plans:
        try:
            entry_block = plan.kernel_op.regions[0].blocks[0]
        except Exception as exc:
            raise DSLRuntimeError(
                "Deferred TempStorage could not locate the entry block for "
                f"{plan.kernel_name}."
            ) from exc
        kernel_key = plan.kernel_op
        existing_group = kernel_groups.get(kernel_key)
        if existing_group is None:
            kernel_groups[kernel_key] = (entry_block, [plan])
        else:
            existing_group[1].append(plan)
        for binding in plan.bindings:
            for placeholder in (
                binding.event.smem_addr_placeholder,
                binding.event.size_placeholder,
            ):
                if not any(
                    callable(getattr(placeholder, method_name, None))
                    for method_name in (
                        "replace_all_uses_with",
                        "replaceAllUsesWith",
                    )
                ):
                    _deferred_temp_storage_capability_error()

    for entry_block, kernel_plans in kernel_groups.values():
        with ir.InsertionPoint.at_block_begin(entry_block):
            allocator = SmemAllocator()
            for plan in kernel_plans:
                smem_ptr = allocator.allocate(
                    plan.size_in_bytes,
                    plan.alignment,
                )
                base_addr = llvm.ptrtoint(T.i32(), smem_ptr.to_llvm_ptr())
                for binding in plan.bindings:
                    smem_addr = base_addr
                    if binding.byte_offset_in_bytes:
                        offset = arith.constant(
                            T.i32(),
                            binding.byte_offset_in_bytes,
                        )
                        smem_addr = arith.addi(base_addr, offset)
                    size = arith.constant(T.i32(), binding.size_in_bytes)
                    _replace_all_uses(
                        binding.event.smem_addr_placeholder,
                        smem_addr,
                    )
                    _replace_all_uses(binding.event.size_placeholder, size)
