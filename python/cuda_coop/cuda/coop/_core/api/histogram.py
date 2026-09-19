# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fresh block histogram counters."""

from __future__ import annotations

from typing import Any

from ..block._common import normalize_positive_int
from ..block.histogram import normalize_histogram_algorithm, validate_histogram_dtype
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _group_primitive_marker,
)
from ._payload import (
    _common_payload_dtype,
    _common_thread_data_extent,
    _validate_common_thread_data_payload,
)


@_common_group_operation("histogram", group_kinds=("block",))
def histogram(
    group: ThreadGroup,
    samples: Any,
    /,
    *,
    bins: Any,
    bins_per_thread: Any = 1,
    counter_dtype: Any = None,
    algorithm: str = "atomic",
    temp_storage: Any = None,
) -> Any:
    """Return fresh striped bin counts, preserving the input samples.

    All members of a complete one-dimensional block participate. Samples
    must be integral bin indices in ``[0, bins)``. ``bins`` and
    ``bins_per_thread`` are positive compile-time integers, with enough
    per-member output slots for every bin. Member ``t`` receives bin
    ``t + i * block_size`` in slot ``i``; slots beyond ``bins`` are zero.
    Samples must be a fixed-size readable ThreadData payload.
    Samples support uint8, int32, uint32, int64 and uint64. Counters default
    to int32; int32, uint32, int64 and uint64 are supported, independently
    of the sample dtype. The Python ``int`` dtype spelling means int32.
    The returned payload has ``bins_per_thread`` counters per member,
    including when each member provides only one sample. Use the striped
    Store algorithm to write counters in bin order.

    ``algorithm`` selects ``"atomic"`` or ``"sort"``. Each call starts at zero.
    Accumulate multiple tiles by adding their returned counters per member;
    choose a counter dtype wide enough for that accumulated total. Every
    input slot contributes a sample: zero-padding an incomplete input tile
    adds counts to bin zero. This operation has no ``valid_items`` control.

    ``temp_storage`` optionally supplies a shared scratch descriptor. It
    holds CUB scratch and intermediate counters for this call. The normal
    storage-reuse synchronization rules apply; it does not retain a running
    histogram between calls. The CUB counterpart is ``cub::BlockHistogram``.
    """
    normalize_positive_int("bins", bins)
    normalize_positive_int("bins_per_thread", bins_per_thread)
    normalize_histogram_algorithm(algorithm)
    if counter_dtype is not None:
        validate_histogram_dtype(counter_dtype, counter=True)
    if _backend_module_name() is not None:
        _validate_common_thread_data_payload(
            "histogram", "samples", samples, allow_readonly=True
        )
        _common_thread_data_extent("histogram", "samples", samples)
        validate_histogram_dtype(_common_payload_dtype("histogram", "samples", samples))
    return _group_primitive_marker(
        "histogram",
        group,
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


__all__ = ["histogram"]
