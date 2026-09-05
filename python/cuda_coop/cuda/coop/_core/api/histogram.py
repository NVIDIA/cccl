# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable fixed-width histogram entry point.

The frontend validates payload, integer counter dtype, static capacity, and the
shared algorithm selector before delegation. Complete-block CUB planning and
backend code generation remain separate.
"""

from __future__ import annotations

from typing import Any

from ..block import validate_block_histogram_output_capacity
from ..thread_group import ThreadGroup
from ._dispatch import (
    _HISTOGRAM_ALGORITHMS,
    _backend_module_name,
    _group_primitive_marker,
    _portable_selector,
    _validate_portable_operation_group,
)
from ._payload import (
    _common_payload_dtype,
    _common_thread_data_extent,
    _validate_common_integer_dtype,
    _validate_common_thread_data_payload,
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
    """Compute a groupwise histogram through the compiler-selected backend.

    ``samples`` is a compiler-produced, fixed-size per-thread payload. Each
    member receives ``bins_per_thread`` striped counters for bin indices
    ``rank + i * group_size``; slots beyond ``bins`` are zero. The static
    projection must have enough capacity for all bins. The portable API accepts
    uint8/int32/uint32/int64/uint64 samples and int32/uint32/int64/uint64
    counters; the Python ``int`` dtype spelling maps to int32. Every sample
    must satisfy CUB's ``0 <= sample < bins`` precondition; violating it is
    undefined behavior.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("histogram", group)
        _validate_common_thread_data_payload("histogram", "samples", samples)
        _common_thread_data_extent("histogram", "samples", samples)
        _validate_common_integer_dtype(
            "histogram",
            "sample",
            _common_payload_dtype("histogram", "samples", samples),
            allow_uint8=True,
        )
        if counter_dtype is not None:
            _validate_common_integer_dtype(
                "histogram",
                "counter",
                counter_dtype,
                allow_uint8=False,
            )
        if (
            isinstance(group, ThreadGroup)
            and group.kind == "block"
            and group.static_size is not None
        ):
            validate_block_histogram_output_capacity(
                bins=bins,
                bins_per_thread=bins_per_thread,
                block_threads=group.static_size,
                scope=f"{_backend_module_name()}.histogram",
            )

    algorithm = _portable_selector(
        "histogram", "algorithm", algorithm, _HISTOGRAM_ALGORITHMS
    )

    return _group_primitive_marker(
        "histogram",
        group,
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
    )


__all__ = ["histogram"]
