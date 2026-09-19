# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common cooperative exchange entry point."""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _common_selector,
    _group_primitive_marker,
)
from ._payload import (
    ThreadDataLike,
    _ReadableThreadDataLike,
    _validate_common_numeric_value,
)

_COMMON_EXCHANGE_MODES = frozenset(
    {
        "striped_to_blocked",
        "blocked_to_striped",
    }
)


@_common_group_operation(
    "exchange",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def exchange(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "striped_to_blocked",
) -> ThreadDataLike[Any]:
    """Convert between blocked and striped per-thread layouts.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating :ref:`thread group <coop-thread-groups>`: a complete
        block, physical warp, or logical warp. Logical warp widths must be
        powers of two between 1 and 32. Every member must call the primitive;
        warp operations require an enclosing block size divisible by 32.
    value : cuda.coop.ThreadDataLike
        Readable :ref:`per-thread payload <coop-thread-data>` with a fixed
        number of items. All members must use the same dtype and extent.
        Supports signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``. Scalar inputs are unsupported.
    mode : str, optional
        Compile-time layout conversion, default ``"striped_to_blocked"``.
        For a group of ``G`` threads with ``K`` items per thread, blocked
        order assigns logical element ``rank * K + item`` to each slot;
        striped order assigns ``item * G + rank``.
        ``"striped_to_blocked"`` gives each thread consecutive elements.
        ``"blocked_to_striped"`` gives neighboring threads neighboring
        elements at each item index. Warp exchanges apply independently
        within each physical or logical warp.

    Returns
    -------
    cuda.coop.ThreadDataLike
        New writable payload with the input dtype and extent in the requested
        layout. The input payload is preserved.

    Notes
    -----
    The call rearranges values already held by the group; it does not load or
    store a memory tile. The implementation manages
    :ref:`temporary storage <coop-temp-storage>` automatically. For ranked
    scatter and other backend-specific modes, use a qualified
    ``cuda.coop.<backend>`` API where supported.

    See Also
    --------
    :cpp:struct:`cub::BlockExchange`, :cpp:struct:`cub::WarpExchange`
        C++ layout-exchange primitives.

    Examples
    --------
    Load two values per thread in striped order, then exchange them into
    blocked order before storing. The output has the original array order.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_rearrangement_examples.py
        :language: python
        :start-after: # exchange-example-begin
        :end-before: # exchange-example-end
        :dedent: 4
    """

    mode = _common_selector(
        "exchange",
        "mode",
        mode,
        _COMMON_EXCHANGE_MODES,
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "exchange",
            "value",
            value,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
    return _group_primitive_marker(
        "exchange",
        group,
        value,
        mode=mode,
    )


__all__ = ["exchange"]
