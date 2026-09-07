# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first public-CUB run-length decode entrypoint."""

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import (
    normalize_positive_int,
)

from ._run_length_controls import validate_decoded_window_offset
from ._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]


def _positive_int(name: str, value: Any) -> int:
    try:
        return normalize_positive_int(name, value)
    except ValueError as exc:
        raise ValueError(f"{_SCOPE}.run_length_decode: {exc}") from exc


def _run_length_decode(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    /,
    *,
    decoded_items_per_thread: Any,
    decoded_window_offset: Any = 0,
    relative_offsets: Any = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
    source: str = "cutlass_root",
) -> Any:
    """Lower one qualified or common-root group-first decode call."""

    from ._compiler._launch import infer_launch_facts

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.run_length_decode group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.run_length_decode currently lowers only this_block groups"
        )
    run_values = _coerce_thread_payload(
        run_values,
        scope=_SCOPE,
        primitive_name="run_length_decode",
        arg_name="run_values",
    )
    run_lengths = _coerce_thread_payload(
        run_lengths,
        scope=_SCOPE,
        primitive_name="run_length_decode",
        arg_name="run_lengths",
    )
    decoded_items_per_thread = _positive_int(
        "decoded_items_per_thread",
        decoded_items_per_thread,
    )
    decoded_window_offset = validate_decoded_window_offset(
        decoded_window_offset,
        scope=_SCOPE,
    )
    launch = infer_launch_facts(
        {},
        scope=_SCOPE,
        primitive_name="run_length_decode",
    )
    if not launch.is_verified("exact_block_dim"):
        raise NotImplementedError(
            f"{_SCOPE}.run_length_decode requires exact block dimensions "
            "from verified compiler launch facts"
        )
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="run_length_decode",
    )

    from ._lowering import _run_length_decode as _provider

    return _provider.provider_run_length_decode(
        group=validated_group,
        launch=launch,
        run_values=run_values,
        run_lengths=run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
        source=source,
    )


def run_length_decode(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    /,
    *,
    decoded_items_per_thread: Any,
    decoded_window_offset: Any = 0,
    relative_offsets: Any = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
) -> Any:
    """Decode a run-length stream across an explicit CUDA block group.

    Run values and lengths may be scalar, ``ThreadData``, rmem tensors, or
    ``TensorSSA``. Mutable output payloads remain explicit ``ThreadData``.
    Actual run lengths must be positive and may be followed by one suffix of
    zero-length padding entries; their block-wide sum must be positive and
    representable in the run-length dtype.
    Targets beyond that sum return zero values and all-ones relative offsets
    (``-1`` for signed run-length dtypes). Window offsets must be nonnegative
    and representable in the run-length dtype. Dynamic offsets retain those
    uniform caller preconditions and do not affect artifact identity.
    """

    return _run_length_decode(
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
        source="cutlass_root",
    )


__all__ = ["run_length_decode"]
