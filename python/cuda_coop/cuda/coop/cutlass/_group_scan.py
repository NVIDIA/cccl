# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in group Scan and Sum entry points for CuTe kernels."""

from enum import Enum

from cuda.coop._core.api._payload import _validate_common_temp_storage
from cuda.coop._core.thread_group import ThreadGroup

from ._thread_data import _coerce_thread_payload
from ._thread_group import _require_complete_warp_partition

_SCOPE = "cuda.coop.cutlass"
_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})


def _selector(value, *, name, choices):
    if not isinstance(value, str) or isinstance(value, Enum):
        raise TypeError(f"{_SCOPE}.scan {name} must be a string")
    value = value.strip().lower().replace("-", "_")
    if value not in choices:
        raise ValueError(
            f"{_SCOPE}.scan {name} must be one of: {', '.join(sorted(choices))}"
        )
    return value


def scan(
    group,
    value,
    /,
    *,
    mode="exclusive",
    scan_op=None,
    initial_value=None,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Compute built-in prefixes in linear group order, preserving the input.

    Blocks accept scalars or fixed per-thread payloads in blocked order;
    physical and logical warps accept scalars. Every group member participates.
    A payload input returns a fresh payload of the same extent and dtype.
    Exclusive Sum defaults to zero; other exclusive operators require a seed.
    Inclusive scans reject a seed. Typed seeds must match the input dtype;
    Python literals must be finite and representable.

    The qualified ``valid_items`` control selects a prefix of warp lanes from
    one through group size. Only those lanes have defined scan results; all
    lanes must still participate. ``aggregate_output`` is a writable one-item
    ThreadData receiving the input aggregate on every member, excluding the
    seed and any invalid tail. These two controls are backend-qualified.

    Only blocks accept algorithm selection and explicit TempStorage. Its
    automatic trailing barrier protects scratch reuse unless disabled by the
    caller. Custom operators and prefix callbacks are not supported.
    """
    from ._compiler._launch import current_kernel_launch_facts
    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.scan group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(f"{_SCOPE}.scan requires a block or warp group")
    mode = _selector(mode, name="mode", choices={"inclusive", "exclusive"})
    if algorithm is not None:
        algorithm = _selector(algorithm, name="algorithm", choices=_ALGORITHMS)
    op = normalize_operator(scan_op, primitive="scan")
    if mode == "inclusive" and initial_value is not None:
        raise ValueError(f"{_SCOPE}.scan inclusive mode does not accept initial_value")
    if mode == "exclusive" and op != "sum" and initial_value is None:
        raise ValueError(f"{_SCOPE}.scan exclusive {op} requires initial_value")
    if group.kind != "block":
        if algorithm is not None:
            raise ValueError(f"{_SCOPE}.scan algorithm selection requires a block")
        if temp_storage is not None:
            raise ValueError(f"{_SCOPE}.scan temp_storage is supported only for blocks")
    elif temp_storage is not None:
        _validate_common_temp_storage("scan", temp_storage)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="scan",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
    )
    launch = current_kernel_launch_facts()
    _require_complete_warp_partition(
        group, feature="scan", exact_block_dim=launch.exact_block_dim
    )
    from ._lowering._scan import provider_scan

    return provider_scan(
        group=group,
        launch=launch,
        value=value,
        mode=mode,
        op=op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_scan(
    group,
    value,
    /,
    *,
    scan_op=None,
    initial_value=None,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return exclusive prefixes using the contracts of :func:`scan`."""
    return scan(
        group,
        value,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_scan(
    group,
    value,
    /,
    *,
    scan_op=None,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return inclusive prefixes using the contracts of :func:`scan`."""
    return scan(
        group,
        value,
        mode="inclusive",
        scan_op=scan_op,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_sum(
    group,
    value,
    /,
    *,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return exclusive sums starting at zero, with qualified prefix controls."""
    return scan(
        group,
        value,
        mode="exclusive",
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_sum(
    group,
    value,
    /,
    *,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return inclusive sums using the contracts of :func:`scan`."""
    return scan(
        group,
        value,
        mode="inclusive",
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


__all__ = ["scan", "exclusive_scan", "inclusive_scan", "exclusive_sum", "inclusive_sum"]
