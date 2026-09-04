# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange payload inference and pre-provider validation."""

from enum import Enum

from numba_cuda_mlir import types

from ._group_rewriting import GroupRewriteContext
from ._parameters import _validate_common_numeric_dtype, normalize_dtype_param
from ._rewrite_payload import PayloadInference
from ._rewrite_support import CoopSinglePhaseRewriteError, _dtype_values_match

_SCATTER_MODES = frozenset(
    {
        "scatter_to_blocked",
        "scatter_to_striped",
        "scatter_to_striped_guarded",
        "scatter_to_striped_flagged",
    }
)


def _mode_token(value: object) -> str:
    if isinstance(value, str) and not isinstance(value, Enum):
        return value.strip().lower().replace("-", "_")
    raise CoopSinglePhaseRewriteError(
        "coop exchange mode must be a compile-time string"
    )


def _require_array(
    inference: PayloadInference,
    index: int,
    name: str,
):
    value, spec = inference.array_candidate(index)
    if value is None or spec is None or spec.items_per_thread is None:
        raise CoopSinglePhaseRewriteError(
            "coop exchange requires "
            f"{name} to be a fixed-size ThreadData or local array"
        )
    return value, spec


def _require_matching_extent(
    *,
    expected: int,
    actual: int,
    name: str,
) -> None:
    if actual != expected:
        raise CoopSinglePhaseRewriteError(
            "coop exchange requires value and "
            f"{name} arrays to have matching items_per_thread"
        )


def _validate_rank_dtype(dtype: object) -> object:
    try:
        dtype = normalize_dtype_param(dtype)
    except (TypeError, ValueError) as exc:
        raise CoopSinglePhaseRewriteError(
            "coop exchange ranks must have a signed integer dtype"
        ) from exc
    literal_type = getattr(dtype, "literal_type", dtype)
    if (
        isinstance(literal_type, types.Boolean)
        or not isinstance(literal_type, types.Integer)
        or not literal_type.signed
    ):
        raise CoopSinglePhaseRewriteError(
            "coop exchange ranks must have a signed integer dtype"
        )
    return literal_type


def _validate_flag_dtype(dtype: object) -> object:
    try:
        dtype = normalize_dtype_param(dtype)
    except (TypeError, ValueError) as exc:
        raise CoopSinglePhaseRewriteError(
            "coop exchange valid_flags must have an integral non-bool dtype"
        ) from exc
    literal_type = getattr(dtype, "literal_type", dtype)
    if isinstance(literal_type, types.Boolean) or not isinstance(
        literal_type, types.Integer
    ):
        raise CoopSinglePhaseRewriteError(
            "coop exchange valid_flags must have an integral non-bool dtype"
        )
    return literal_type


def infer_exchange_payload(
    context: GroupRewriteContext,
    inference: PayloadInference,
) -> None:
    """Infer one out-of-place Exchange provider specialization."""

    input_var, input_spec = _require_array(inference, 0, "value")
    output_var, output_spec = _require_array(inference, 1, "result")
    extent = input_spec.items_per_thread
    assert extent is not None
    assert output_spec.items_per_thread is not None
    _require_matching_extent(
        expected=extent,
        actual=output_spec.items_per_thread,
        name="result",
    )

    input_dtype = inference.inferred_array_dtype(input_var, input_spec)
    output_dtype = inference.inferred_array_dtype(output_var, output_spec)
    if input_dtype is None:
        input_dtype = output_dtype
    if output_dtype is None:
        output_dtype = input_dtype
    if input_dtype is None:
        input_dtype = inference.factory_value("dtype")
    if input_dtype is None:
        raise CoopSinglePhaseRewriteError("coop exchange could not infer value dtype")
    try:
        input_dtype = _validate_common_numeric_dtype(
            input_dtype,
            operation="exchange",
            parameter="value",
        )
    except (TypeError, ValueError) as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc
    if output_dtype is not None and not _dtype_values_match(input_dtype, output_dtype):
        raise CoopSinglePhaseRewriteError(
            "coop exchange requires value and result arrays to have matching dtype"
        )

    mode_value = inference.factory_value("mode")
    mode = _mode_token("striped_to_blocked" if mode_value is None else mode_value)
    uses_ranks = mode in _SCATTER_MODES
    uses_valid_flags = mode == "scatter_to_striped_flagged"
    expected_count = 2 + int(uses_ranks) + int(uses_valid_flags)
    if len(inference.runtime_args) != expected_count:
        raise CoopSinglePhaseRewriteError(
            f"coop exchange mode {mode!r} expects {expected_count} runtime "
            f"arguments; got {len(inference.runtime_args)}"
        )

    if uses_ranks:
        ranks_var, ranks_spec = _require_array(inference, 2, "ranks")
        assert ranks_spec.items_per_thread is not None
        _require_matching_extent(
            expected=extent,
            actual=ranks_spec.items_per_thread,
            name="ranks",
        )
        rank_dtype = inference.inferred_array_dtype(ranks_var, ranks_spec)
        if rank_dtype is None:
            rank_dtype = inference.factory_value("rank_dtype")
        if rank_dtype is None:
            raise CoopSinglePhaseRewriteError(
                "coop exchange could not infer rank_dtype from ranks"
            )
        inference.infer_kwarg("rank_dtype", _validate_rank_dtype(rank_dtype))

    if uses_valid_flags:
        flags_var, flags_spec = _require_array(inference, 3, "valid_flags")
        assert flags_spec.items_per_thread is not None
        _require_matching_extent(
            expected=extent,
            actual=flags_spec.items_per_thread,
            name="valid_flags",
        )
        flag_dtype = inference.inferred_array_dtype(flags_var, flags_spec)
        if flag_dtype is None:
            flag_dtype = inference.factory_value("valid_flag_dtype")
        if flag_dtype is None:
            raise CoopSinglePhaseRewriteError(
                "coop exchange could not infer valid_flag_dtype from valid_flags"
            )
        inference.infer_kwarg(
            "valid_flag_dtype",
            _validate_flag_dtype(flag_dtype),
        )

    inference.infer_kwarg("items_per_thread", extent)
    inference.infer_kwarg("dtype", input_dtype)
    context.record_thread_data_dtype(input_var, input_dtype)
    context.record_thread_data_dtype(output_var, input_dtype)


__all__ = ["infer_exchange_payload"]
