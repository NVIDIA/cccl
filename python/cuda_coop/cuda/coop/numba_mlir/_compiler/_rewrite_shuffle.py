# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle payload inference and runtime-distance validation."""

from numbers import Integral

from cuda.coop._core import ArgumentBinding, BindingKind

from ._group_rewriting import GroupRewriteContext
from ._parameters import (
    _validate_common_numeric_dtype,
    _validate_runtime_integer_dtype,
)
from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
    ir,
)

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


def _mode_token(value: object) -> str:
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_")
    raise CoopSinglePhaseRewriteError("coop shuffle mode must be a compile-time string")


def _block_threads(value: object) -> int | None:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    if any(
        not isinstance(component, Integral) or isinstance(component, bool)
        for component in value
    ):
        return None
    return int(value[0]) * int(value[1]) * int(value[2])


def _numeric_dtype(dtype: object, *, parameter: str) -> object:
    try:
        return _validate_common_numeric_dtype(
            dtype,
            operation="shuffle",
            parameter=parameter,
        )
    except (TypeError, ValueError) as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc


def infer_shuffle_scalar_payload(
    context: GroupRewriteContext,
    inference: PayloadInference,
) -> None:
    """Infer the scalar Shuffle provider dtype."""

    if not inference.runtime_args or not isinstance(inference.runtime_args[0], ir.Var):
        raise CoopSinglePhaseRewriteError(
            "coop shuffle scalar value must be a runtime variable"
        )
    dtype = context.dtype(inference.runtime_args[0])
    if dtype is None:
        dtype = inference.factory_value("dtype")
    if dtype is None:
        raise CoopSinglePhaseRewriteError(
            "coop shuffle could not infer scalar value dtype"
        )
    inference.infer_kwarg("dtype", _numeric_dtype(dtype, parameter="value"))


def infer_shuffle_array_payload(
    context: GroupRewriteContext,
    inference: PayloadInference,
) -> None:
    """Infer matching array payload metadata for Up or Down."""

    input_var, input_spec = inference.array_candidate(0)
    output_var, output_spec = inference.array_candidate(1)
    if (
        input_var is None
        or input_spec is None
        or input_spec.items_per_thread is None
        or output_var is None
        or output_spec is None
        or output_spec.items_per_thread is None
    ):
        raise CoopSinglePhaseRewriteError(
            "coop shuffle array value and result must be fixed-size "
            "ThreadData or local arrays"
        )
    if input_spec.items_per_thread != output_spec.items_per_thread:
        raise CoopSinglePhaseRewriteError(
            "coop shuffle requires value and result arrays to have matching "
            "items_per_thread"
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
        raise CoopSinglePhaseRewriteError(
            "coop shuffle could not infer array value dtype"
        )
    input_dtype = _numeric_dtype(input_dtype, parameter="value")
    if output_dtype is not None and not _dtype_values_match(input_dtype, output_dtype):
        raise CoopSinglePhaseRewriteError(
            "coop shuffle requires value and result arrays to have matching dtype"
        )

    inference.infer_kwarg("items_per_thread", input_spec.items_per_thread)
    inference.infer_kwarg("dtype", input_dtype)
    context.record_thread_data_dtype(input_var, input_dtype)
    context.record_thread_data_dtype(output_var, input_dtype)


def validate_shuffle_scalar_runtime_controls(
    context: GroupRewriteContext,
    *,
    op_name: str,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
) -> None:
    """Validate scalar Offset or Rotate distance before specialization."""

    del op_name
    mode = _mode_token(factory_kwargs.get("mode", "offset"))
    if mode not in {"offset", "rotate"}:
        raise CoopSinglePhaseRewriteError(
            "coop scalar shuffle mode must be 'offset' or 'rotate'"
        )
    distance = factory_kwargs.get("distance")
    if not isinstance(distance, ArgumentBinding):
        raise CoopSinglePhaseRewriteError(
            "coop scalar shuffle requires an integer distance"
        )
    if distance.kind is BindingKind.STATIC:
        value = distance.value
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise CoopSinglePhaseRewriteError(
                "coop shuffle distance must be an integer, not bool or a "
                "noninteger scalar"
            )
        value = int(value)
        if not _I32_MIN <= value <= _I32_MAX:
            raise CoopSinglePhaseRewriteError(
                "coop shuffle distance must fit a signed 32-bit integer"
            )
        if mode == "rotate":
            block_threads = _block_threads(factory_kwargs.get("threads_per_block"))
            if block_threads is not None and not 1 <= value < block_threads:
                raise CoopSinglePhaseRewriteError(
                    "coop shuffle rotate distance must satisfy "
                    "1 <= distance < block_threads"
                )
        return
    if distance.kind is not BindingKind.RUNTIME:
        raise CoopSinglePhaseRewriteError(
            "coop scalar shuffle requires an integer distance"
        )
    if len(runtime_args) != 2 or not isinstance(runtime_args[1], ir.Var):
        raise CoopSinglePhaseRewriteError(
            "coop shuffle runtime distance is missing its runtime value"
        )
    dtype = context.numba_type(runtime_args[1])
    if dtype is None:
        dtype = context.dtype(runtime_args[1])
    if dtype is None:
        return
    try:
        _validate_runtime_integer_dtype(
            dtype,
            operation="shuffle",
            parameter="distance",
        )
    except TypeError as exc:
        raise CoopSinglePhaseRewriteError(str(exc)) from exc


def validate_shuffle_array_runtime_controls(
    context: GroupRewriteContext,
    *,
    op_name: str,
    runtime_args: list[ir.Var],
    factory_kwargs: dict[str, object],
) -> None:
    """Keep array Shuffle on CUB's unit-distance Up or Down ABI."""

    del context, op_name, runtime_args
    mode = _mode_token(factory_kwargs.get("mode", "down"))
    if mode not in {"up", "down"}:
        raise CoopSinglePhaseRewriteError(
            "coop array shuffle mode must be 'up' or 'down'"
        )
    if "distance" in factory_kwargs:
        raise CoopSinglePhaseRewriteError(
            "coop array shuffle uses a unit distance and does not pass a "
            "distance to the provider"
        )


__all__ = [
    "infer_shuffle_array_payload",
    "infer_shuffle_scalar_payload",
    "validate_shuffle_array_runtime_controls",
    "validate_shuffle_scalar_runtime_controls",
]
