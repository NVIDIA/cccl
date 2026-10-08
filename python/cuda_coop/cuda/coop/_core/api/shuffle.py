# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Define the common block Shuffle call for shifting items by one place.

The common API takes a payload, an up/down direction, and a static distance
of one element. A tracing compiler, such as the CuTe DSL, runs this body in
Python. The body checks these limits, then calls the active backend.
Numba-CUDA-MLIR replaces the call during compilation and applies its own
checks. A call outside a compiler environment raises an error.

The qualified APIs also provide scalar offset and rotate forms.
"""

from __future__ import annotations

from enum import Enum
from numbers import Integral
from typing import Literal, TypeVar

from cuda.coop._typing import (
    CommonNumericScalar,
    CommonShuffleMode,
    CommonThreadDataLike,
)

from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _common_selector,
    _group_primitive_marker,
)
from ._payload import (
    ThreadDataLike,
    _validate_common_numeric_value,
)
from .thread_group import BlockGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


_COMMON_SHUFFLE_MODES = frozenset({"down", "up"})


@_common_group_operation(
    "shuffle",
    group_kinds=("block",),
)
def shuffle(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT],
    /,
    *,
    mode: CommonShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadDataLike[_ItemT]:
    """Shift a block's flattened payload by one element.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Complete block whose members all call the primitive; see
        :ref:`thread groups <coop-common-groups>`. Warp, mapped-warp, cluster,
        and grid groups are unsupported.
    value : cuda.coop.ThreadDataLike
        Readable :ref:`per-thread payload <coop-common-payloads>` in blocked
        order. All threads must use the same dtype and fixed extent.
        Supports signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``. Scalar inputs are unsupported.
    mode : str, optional
        Compile-time direction, ``"down"`` (the default) or ``"up"``.
        Flatten the payloads in linear thread-rank order, with each thread's
        items consecutive. ``"down"`` places input element ``i + 1`` at
        output position ``i``; the final output element is undefined.
        ``"up"`` places input element ``i - 1`` at output position ``i``;
        the first output element is undefined.
    distance : int, optional
        Compile-time shift distance, which must be exactly ``1``. The shift
        crosses thread boundaries as needed. Both
        :func:`cuda.coop.numba_mlir.shuffle` and
        :func:`cuda.coop.cutlass.shuffle` support scalar offset and rotate
        operations with compile-time or runtime distances.

    Returns
    -------
    cuda.coop.ThreadDataLike
        New writable payload with the input dtype and extent. The input is
        preserved. Initialize the undefined boundary slot before reading it,
        or exclude it from subsequent processing.

    Notes
    -----
    The shift has no wraparound. Its unit is one element in the flattened
    block tile. The implementation manages
    :ref:`temporary storage <coop-common-storage>` automatically.

    See Also
    --------
    :cpp:struct:`cub::BlockShuffle`
        C++ block shift, offset, and rotate primitive.

    Examples
    --------
    Shift a block tile in both directions and fill the exposed boundary with
    zero before storing the results.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_rearrangement_examples.py
        :language: python
        :start-after: # shuffle-example-begin
        :end-before: # shuffle-example-end
        :dedent: 4

    For payload shifts in CuTe, see
    :ref:`CUTLASS Shuffle <coop-cutlass-shuffle>`. Both
    :func:`cuda.coop.numba_mlir.shuffle` and :func:`cuda.coop.cutlass.shuffle`
    also support scalar offset and rotate operations.
    """

    mode = _common_selector(
        "shuffle",
        "mode",
        mode,
        _COMMON_SHUFFLE_MODES,
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "shuffle",
            "value",
            value,
            allow_readonly_thread_data=True,
            require_thread_data=True,
        )
        if (
            isinstance(distance, (bool, Enum))
            or not isinstance(distance, Integral)
            or int(distance) != 1
        ):
            raise ValueError(
                "cuda.coop.shuffle distance must be exactly 1 in the common "
                "API; use cuda.coop.numba_mlir for scalar Shuffle"
            )
        distance = 1
    return _group_primitive_marker(
        "shuffle",
        group,
        value,
        mode=mode,
        distance=distance,
    )


__all__ = ["shuffle"]
