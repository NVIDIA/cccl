# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Define the common Exchange call for blocked and striped layouts.

Exchange converts a group payload between blocked and striped order. The
decorator registers the function so a compiler can recognize calls to it. A
host Python call raises an error.
"""

from __future__ import annotations

from typing import TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ExchangeMode,
)

from ..thread_group import CoopCompilerContextRequiredError
from ._dispatch import (
    _common_group_operation,
)
from ._payload import (
    ThreadDataLike,
)
from .thread_group import MemoryGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


@_common_group_operation(
    "exchange",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def exchange(
    group: MemoryGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: ExchangeMode = "striped_to_blocked",
) -> ThreadDataLike[_ItemT]:
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
    Load ``items_per_thread`` values per thread in striped order, then exchange
    them into blocked order before storing. The output has the original
    array order.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_rearrangement_examples.py
        :language: python
        :start-after: # exchange-example-begin
        :end-before: # exchange-example-end
        :dedent: 4
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.exchange must be called from a supported GPU kernel."
    )


__all__ = ["exchange"]
