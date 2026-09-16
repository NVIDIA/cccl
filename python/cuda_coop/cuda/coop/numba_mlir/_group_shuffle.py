# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Shuffle marker for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from .._core.api._payload import ThreadDataLike, _ReadableThreadDataLike
from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation(
    "shuffle",
    family_module="cuda.coop.numba_mlir._compiler._group_shuffle",
)
def shuffle(
    group: ThreadGroup,
    value: _ReadableThreadDataLike[Any],
    /,
    *,
    mode: Any = "down",
    distance: Any = 1,
) -> ThreadDataLike[Any]:
    """Shift arrays or select another thread's scalar within a block.

    See :func:`cuda.coop.shuffle` for the shared block participation rules,
    supported dtypes, and unit ``"up"``/``"down"`` shifts. The qualified API
    also accepts fixed-size local arrays and scalar offset or rotate calls.

    Parameters
    ----------
    value : numeric scalar, cuda.coop.ThreadDataLike, or local array
        Array payloads can be :ref:`ThreadData <coop-thread-data>` or a
        one-dimensional ``cuda.local.array`` with a fixed extent. Each thread
        supplies one scalar for offset or rotate modes.
    mode : str, optional
        Compile-time mode, default ``"down"``. Array payloads support only
        ``"up"`` and ``"down"``. Scalar payloads require ``"offset"`` or
        ``"rotate"``: thread rank ``r`` receives the value from rank
        ``r + distance``. Offset results are undefined when the source rank
        lies outside the block; rotate wraps the source rank modulo the block
        size.
    distance : int or integer scalar, optional
        Default ``1``. Arrays require a compile-time unit distance. Scalar
        modes accept a compile-time or runtime integer that may differ between
        threads. Offset distances must fit a signed 32-bit integer and may
        be negative or zero. Rotate distances must be between one and the
        block size minus one, inclusive; rotate requires at least two threads.

    Returns
    -------
    numeric scalar, cuda.coop.ThreadDataLike, or local array
        Scalar input produces a scalar with the input dtype. Array input
        produces a new writable payload with the input dtype and extent and
        preserves the input. Observe the boundary rules of the chosen mode
        before using the result.

    Notes
    -----
    The implementation manages :ref:`temporary storage <coop-temp-storage>`
    automatically. Every member of the :ref:`block <coop-thread-groups>` must
    call the collective, including threads with out-of-range offset sources.

    See Also
    --------
    cuda.coop.shuffle
        Shared array shift contract and executable example.
    :cpp:struct:`cub::BlockShuffle`
        C++ shift, offset, and rotate primitive.

    Examples
    --------
    Rotate one scalar per thread by seven positions, including wraparound.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_movement_examples.py
        :language: python
        :start-after: # rotate-example-begin
        :end-before: # rotate-example-end
        :dedent: 4
    """

    return group_primitive_marker(
        "shuffle",
        group,
        value,
        mode=mode,
        distance=distance,
    )


__all__ = ["shuffle"]
