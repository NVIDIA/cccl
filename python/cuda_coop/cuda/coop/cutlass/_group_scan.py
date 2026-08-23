# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first scan entrypoint."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cuda.coop._core import (
    ArgumentBinding,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupScanSemantics,
    LaunchFacts,
    Reference,
    ScanMode,
    ScanValueKind,
    make_group_primitive_call,
    make_scan_semantics,
    plan_group_primitive,
)
from cuda.coop._core.block import BlockScanAlgorithm

from ._internal._thread_data import _coerce_thread_payload
from ._thread_group import (
    ThreadGroup,
    _require_complete_warp_partition,
    _resolve_collective_group,
    _resolve_collective_group_from_launch,
)

_SCOPE = __name__.rsplit(".", 1)[0]

_SCAN_OPERATOR_CPP = {
    "sum": "::cuda::std::plus<T>",
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}

_BLOCK_SCAN_ALGORITHM_ALIASES = {
    "raking": BlockScanAlgorithm.RAKING,
    "raking_memoize": BlockScanAlgorithm.RAKING_MEMOIZE,
    "warp_scans": BlockScanAlgorithm.WARP_SCANS,
}


def _normalize_scan_op(scan_op: Any) -> str:
    from ._dsl._scope import normalize_scan_op

    return normalize_scan_op(scan_op, scope=_SCOPE)


def _normalize_scan_mode(mode: Any) -> str:
    try:
        return ScanMode(mode).value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_SCOPE}.scan mode must be 'exclusive' or 'inclusive'"
        ) from exc


def _normalize_block_scan_algorithm(algorithm: Any) -> Any:
    if isinstance(algorithm, str):
        return _BLOCK_SCAN_ALGORITHM_ALIASES.get(algorithm, algorithm)
    return algorithm


def _validate_group_for_scan(group: ThreadGroup) -> None:
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.scan currently lowers CUB scans only for this_block "
            "and physical or logical warp groups"
        )


def _classify_valid_items(valid_items: Any) -> ArgumentBinding:
    if valid_items is None:
        return ArgumentBinding.omitted()
    if isinstance(valid_items, bool):
        raise TypeError(f"{_SCOPE}.scan valid_items must be an integer, not bool")
    if isinstance(valid_items, Integral):
        return ArgumentBinding.static(int(valid_items))
    return ArgumentBinding.runtime()


def _make_group_scan_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    dtype: Any,
    value_kind: ScanValueKind,
    items_per_thread: int,
    mode: str,
    op: str,
    initial_value: Any = None,
    aggregate: bool = False,
    valid_items: Any = None,
    algorithm: Any = None,
    source: str = "cutlass_root",
) -> GroupLoweringPlan:
    """Build the canonical shared-core plan for one CUTLASS group scan."""

    mode = _normalize_scan_mode(mode)
    if group.kind == "block" and algorithm is None:
        algorithm = BlockScanAlgorithm.RAKING
    if mode == ScanMode.INCLUSIVE.value and initial_value is not None:
        raise ValueError(
            f"{_SCOPE}.scan initial_value is not supported for inclusive scans"
        )
    if mode == ScanMode.EXCLUSIVE.value and op != "sum" and initial_value is None:
        raise ValueError(
            f"{_SCOPE}.scan requires initial_value for non-default exclusive scans"
        )

    initial_descriptor = (
        Reference(Dependency("T"), name="initial_value")
        if initial_value is not None
        else None
    )
    scan_operator = None
    if op != "sum" or initial_descriptor is not None:
        try:
            cpp = _SCAN_OPERATOR_CPP[op]
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported group scan operator {op!r}"
            ) from exc
        scan_operator = CxxOperator(
            cpp=cpp,
            dtype=Dependency("T"),
            name="scan_op",
        )

    primitive = make_scan_semantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        scan_operator=scan_operator,
        initial_value=initial_descriptor,
        aggregate=aggregate,
    )
    call = make_group_primitive_call(
        group,
        GroupScanSemantics(
            primitive=primitive,
            cub_algorithm=algorithm,
            valid_items=_classify_valid_items(valid_items),
        ),
        source=source,
    )
    return plan_group_primitive(call, launch)


def _resolve_group_for_scan(
    group: ThreadGroup,
    launch_kwargs: dict[str, Any],
) -> ThreadGroup:
    return _resolve_collective_group(
        group,
        launch_kwargs,
        feature="scan",
    )


def _scan(
    group: ThreadGroup,
    value: Any,
    /,
    *args: Any,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    aggregate_output: Any = None,
    valid_items: Any = None,
    temp_storage: Any = None,
    source: str = "cutlass_root",
    **kwargs: Any,
) -> Any:
    """Internal group scan entrypoint shared by root and scoped APIs."""

    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import merge_payload, validate_no_extra_args

    payload = merge_payload(
        _SCOPE,
        "scan",
        {
            "group": group,
            "value": value,
            "args": args,
            "mode": mode,
            "scan_op": scan_op,
            "initial_value": initial_value,
            "algorithm": algorithm,
            "aggregate_output": aggregate_output,
            "valid_items": valid_items,
            "temp_storage": temp_storage,
            "source": source,
        },
        kwargs,
    )
    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "scan",
        args=payload.pop("args"),
        kwargs=kwargs,
        expected=(
            "expects a ThreadGroup and one positional value, with optional "
            "mode, scan operator, initial value, and block algorithm selectors"
        ),
    )
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.scan group must be a ThreadGroup")

    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="scan",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
    )

    mode = _normalize_scan_mode(mode)
    op = _normalize_scan_op(scan_op)
    algorithm = _normalize_block_scan_algorithm(algorithm)
    if mode == ScanMode.INCLUSIVE.value and initial_value is not None:
        raise ValueError(
            f"{_SCOPE}.scan initial_value is not supported for inclusive scans"
        )
    if mode == ScanMode.EXCLUSIVE.value and op != "sum" and initial_value is None:
        raise ValueError(
            f"{_SCOPE}.scan requires initial_value for non-default exclusive scans"
        )

    _validate_group_for_scan(group)
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="scan",
    )
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="scan",
    )
    assert validated_group.hierarchy is not None
    _require_complete_warp_partition(
        validated_group,
        feature="scan",
        exact_block_dim=validated_group.hierarchy.block_dim,
    )

    from ._dsl import _cub_scan_provider as _provider

    return _provider.provider_scan(
        group=group,
        launch=launch,
        value=value,
        mode=mode,
        op=op,
        initial_value=initial_value,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
        source=source,
    )


def scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Scan values across a complete CUDA block, physical warp, or logical warp.

    ``group`` always names the participating threads explicitly.
    Block scans accept scalars, ``ThreadData``, rmem tensors, and ``TensorSSA``.
    Physical- and logical-warp scans are scalar-per-lane. The default operator is sum;
    non-default exclusive scans require ``initial_value``. Inclusive scans do
    not accept an initial value. Explicit ``algorithm`` selectors apply only
    to block scans and accept ``raking``, ``raking_memoize``, or ``warp_scans``.
    ``warp_scans`` requires a block size that is a multiple of 32 threads.
    A runtime ``initial_value`` must be uniform across every group member.
    ``valid_items`` selects a leading partial logical warp and is accepted only
    for physical or logical warp groups. ``aggregate_output`` receives the
    reduction of the input values and excludes ``initial_value``.

    Every group member must invoke the collective in converged control flow.
    A size-less ``coop.TempStorage()`` requests inferred caller-owned scratch
    for block scans. Shared storage inserts the reuse barrier by default;
    ``auto_sync=False`` leaves synchronization to the caller. Warp scans and
    explicitly sized legacy storage retain implementation-owned scratch.

    This is the preferred alpha group-first Scan surface; capability readiness
    remains blocked until focused same-toolchain direct-CUB comparisons cover
    the supported routes.
    """

    return _scan(
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an exclusive sum across a block, physical warp, or logical warp."""

    return _scan(
        group,
        value,
        mode="exclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an inclusive sum across a block, physical warp, or logical warp."""

    return _scan(
        group,
        value,
        mode="inclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an exclusive scan using a supported built-in operator."""

    return _scan(
        group,
        value,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an inclusive scan using a supported built-in operator."""

    return _scan(
        group,
        value,
        mode="inclusive",
        scan_op=scan_op,
        initial_value=None,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
