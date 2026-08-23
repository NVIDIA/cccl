# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS run-length decoding."""

from __future__ import annotations

from typing import Any, overload

from typing_extensions import TypeVar

from .._typing import _IntegerValue as _IntegerValue
from .._typing import _PortableRunLength as _PortableRunLength
from .._typing import _PortableRunValue as _PortableRunValue
from .._typing import _TraceInteger as _TraceInteger
from ._thread_data import ThreadData
from ._thread_data import _CutlassRunTensor as _CutlassRunTensor
from ._thread_group import _BlockGroup as _BlockGroup

_RunValueT = TypeVar("_RunValueT", bound=_PortableRunValue)
_RunLengthT = TypeVar("_RunLengthT", bound=_PortableRunLength)

@overload
def run_length_decode(
    group: _BlockGroup,
    run_values: ThreadData[_RunValueT],
    run_lengths: ThreadData[_RunLengthT] | _CutlassRunTensor,
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
    relative_offsets: ThreadData[_RunLengthT] | None = None,
    total_decoded_size: ThreadData[_RunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> ThreadData[_RunValueT]:
    """Decode a typed ``ThreadData`` run-value payload.

    ``group`` is the complete block. ``run_values`` and ``run_lengths`` are
    register payloads with matching positive extents.
    ``decoded_items_per_thread`` fixes the positive result extent, while the
    uniform ``decoded_window_offset`` selects a nonnegative stream position
    representable in the run-length dtype; dynamic callers guarantee its range.
    ``relative_offsets`` receives offsets within each run and
    ``total_decoded_size`` receives the block-wide stream size; both use the
    run-length dtype. Out-of-range relative offsets are all-ones (``-1`` for
    signed dtypes). ``decoded_offset_dtype`` may spell that compiler dtype
    explicitly.
    Actual runs have positive lengths; zeros are allowed only as one trailing
    padding suffix, and the block-wide sum is positive and representable in the
    run-length dtype. Out-of-range decoded values are zero and the inputs remain
    unchanged.
    """

@overload
def run_length_decode(
    group: _BlockGroup,
    run_values: _CutlassRunTensor,
    run_lengths: ThreadData[_RunLengthT] | _CutlassRunTensor,
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
    relative_offsets: ThreadData[_RunLengthT] | None = None,
    total_decoded_size: ThreadData[_RunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> ThreadData[Any]:
    """Decode a CUTLASS register-memory tensor payload.

    ``group`` is the complete block. ``run_values`` and ``run_lengths`` are
    register payloads with matching positive extents.
    ``decoded_items_per_thread`` fixes the positive result extent, while the
    uniform ``decoded_window_offset`` selects a nonnegative stream position
    representable in the run-length dtype; dynamic callers guarantee its range.
    ``relative_offsets`` receives offsets within each run and
    ``total_decoded_size`` receives the block-wide stream size; both use the
    run-length dtype. Out-of-range relative offsets are all-ones (``-1`` for
    signed dtypes). ``decoded_offset_dtype`` may spell that compiler dtype
    explicitly.
    Tensor element types are compiler-owned, so the result falls back to
    ``ThreadData[Any]``. Actual runs have positive lengths; zeros are allowed
    only as one trailing padding suffix, and the block-wide sum is positive and
    representable in the run-length dtype. Out-of-range values are zero and
    inputs are unchanged.
    """

@overload
def run_length_decode(
    group: _BlockGroup,
    run_values: _RunValueT,
    run_lengths: _RunLengthT,
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
    relative_offsets: ThreadData[_RunLengthT] | None = None,
    total_decoded_size: ThreadData[_RunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> ThreadData[_RunValueT]:
    """Decode one scalar run per CUTLASS block member.

    ``group`` is the complete block. ``run_values`` and ``run_lengths`` are
    scalar integer registers. ``decoded_items_per_thread`` fixes the positive
    result extent, while uniform ``decoded_window_offset`` selects a
    nonnegative stream position representable in the run-length dtype; dynamic
    callers guarantee its range. ``relative_offsets`` receives offsets within
    each run and ``total_decoded_size`` receives the block-wide stream size;
    both use the run-length dtype. Out-of-range relative offsets are all-ones
    (``-1`` for signed dtypes). ``decoded_offset_dtype`` may spell that compiler
    dtype explicitly. Actual runs have positive lengths; zeros are allowed only as one
    trailing padding suffix, and the block-wide sum is positive and
    representable in the run-length dtype. Out-of-range values are zero and
    inputs are unchanged.
    """

__all__ = [
    "run_length_decode",
]
