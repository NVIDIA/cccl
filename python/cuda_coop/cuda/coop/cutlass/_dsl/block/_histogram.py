# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from numbers import Integral
from typing import Any

from cuda.coop._core.block import normalize_block_histogram_algorithm

from .._scope import merge_block_payload as merge_payload
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from ._dispatch import dispatch_primitive, register_primitive_impl


def _normalize_histogram_algorithm(algorithm: Any) -> str:
    try:
        return normalize_block_histogram_algorithm(algorithm).value
    except ValueError as exc:
        raise ValueError(
            "cuda.coop.cutlass._block.histogram algorithm must be 'atomic' or 'sort'"
        ) from exc


def _validate_static_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise NotImplementedError(
            f"cuda.coop.cutlass._block.histogram {name} must be trace-time "
            "static for public CUB BlockHistogram lowering"
        )
    value = int(value)
    if value < 1:
        raise ValueError(
            f"cuda.coop.cutlass._block.histogram {name} must be a positive int"
        )
    return value


def _histogram_provider(
    *,
    samples: Any,
    args: tuple[Any, ...] = (),
    bins: Any,
    bins_per_thread: int = 1,
    counter_dtype: Any = None,
    algorithm: Any = "atomic",
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "histogram",
            args=args,
            kwargs={},
            expected="does not accept extra positional args",
        )

    from ... import _group_histogram as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._histogram(
        this_block(),
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
        source="scoped_block",
        **kwargs,
    )


_histogram_provider._supports_native_thread_data = True
_histogram_provider._preserves_launch_metadata = True
_histogram_provider._uses_planned_temp_storage = True


def histogram(
    samples: Any,
    /,
    *args: Any,
    bins: Any,
    bins_per_thread: int = 1,
    counter_dtype: Any = None,
    algorithm: Any = "atomic",
    **kwargs: Any,
) -> Any:
    """Return per-thread block histogram counters for the input samples.

    ``bins`` is the total histogram width. ``bins_per_thread`` controls how
    many counter bins each thread returns, and ``algorithm`` selects the CUB
    atomic or sort-backed block histogram path.
    """
    structural_payload = {
        "samples": samples,
        "args": args,
        "bins": _validate_static_positive_int("bins", bins),
        "bins_per_thread": _validate_static_positive_int(
            "bins_per_thread",
            bins_per_thread,
        ),
        "algorithm": _normalize_histogram_algorithm(algorithm),
    }
    if counter_dtype is not None:
        structural_payload["counter_dtype"] = counter_dtype
    payload = merge_payload(
        "histogram",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("histogram", kwargs=payload)


register_primitive_impl("histogram", impl=_histogram_provider)
