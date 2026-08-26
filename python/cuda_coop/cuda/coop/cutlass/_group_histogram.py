# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first static-width histogram entrypoint."""

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import (
    normalize_block_histogram_algorithm,
    normalize_block_histogram_positive_int,
)

from ._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]


def _static_positive_int(name: str, value: Any) -> int:
    return normalize_block_histogram_positive_int(
        name,
        value,
        scope=f"{_SCOPE}.histogram",
    )


def _normalize_histogram_algorithm(algorithm: Any) -> Any:
    try:
        return normalize_block_histogram_algorithm(algorithm)
    except ValueError as exc:
        raise ValueError(
            f"{_SCOPE}.histogram algorithm must be 'atomic' or 'sort'"
        ) from exc


def _histogram(
    group: ThreadGroup,
    samples: Any,
    /,
    *,
    bins: Any,
    bins_per_thread: Any = 1,
    counter_dtype: Any = None,
    algorithm: Any = "atomic",
    source: str = "cutlass_root",
) -> Any:
    """Lower one qualified or common-root group-first Histogram call."""

    from ._compiler._launch import infer_launch_facts

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.histogram group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.histogram currently lowers only this_block groups"
        )
    samples = _coerce_thread_payload(
        samples,
        scope=_SCOPE,
        primitive_name="histogram",
        arg_name="samples",
    )
    bins = _static_positive_int("bins", bins)
    bins_per_thread = _static_positive_int("bins_per_thread", bins_per_thread)
    algorithm = _normalize_histogram_algorithm(algorithm)
    launch = infer_launch_facts({}, scope=_SCOPE, primitive_name="histogram")
    if not launch.is_verified("exact_block_dim"):
        raise NotImplementedError(
            f"{_SCOPE}.histogram requires exact block dimensions from "
            "verified compiler launch facts"
        )
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="histogram",
    )

    from ._lowering import _histogram as _provider

    return _provider.provider_histogram(
        group=validated_group,
        launch=launch,
        samples=samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
        source=source,
    )


def histogram(
    group: ThreadGroup,
    samples: Any,
    /,
    *,
    bins: Any,
    bins_per_thread: Any = 1,
    counter_dtype: Any = None,
    algorithm: Any = "atomic",
) -> Any:
    """Construct a static-width histogram across an explicit CUDA block.

    Samples may be a supported integral scalar, ``ThreadData``, an rmem tensor,
    or ``TensorSSA``; tracing validates tensor address space and element dtype.
    ``bins`` and ``bins_per_thread`` are trace-time constants. Each member
    receives striped counters for bin indices ``rank + i * block_size``, and
    their product with the block size must provide capacity for every bin.
    Samples must satisfy CUB's public ``0 <= sample < bins`` precondition.
    """

    return _histogram(
        group,
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
        source="cutlass_root",
    )


__all__ = ["histogram"]
