# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from cuda.coop import ThreadGroup

from ._thread_data import ThreadData

def load(
    group: ThreadGroup,
    source: object,
    items: ThreadData,
    /,
    *,
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
) -> ThreadData:
    """Collectively load one block tile into a per-thread payload.

    Every thread in ``group`` must participate in converged control flow. The
    payload size determines the number of consecutive values loaded per thread.
    Contiguous operands are traversed in linear storage order; multidimensional
    logical indexing is not applied.

    Args:
        group: The current CUDA thread block.
        source: Contiguous pointer-backed input memory.
        items: Payload whose size determines the values owned by each thread.
        valid_items: Optional valid element count for a partial block tile.
        oob_default: Optional value assigned to invalid Load positions.
        offset: Optional element offset from the input pointer.

    Returns:
        ``items`` after the active compiler backend populates it.

    Raises:
        TypeError: If ``group`` is invalid or ``oob_default`` is supplied
            without ``valid_items``.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel loads a partial block tile:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

def store(
    group: ThreadGroup,
    destination: object,
    items: ThreadData,
    /,
    *,
    valid_items: Any = None,
    offset: Any = None,
) -> None:
    """Collectively store one per-thread payload as one block tile.

    Every thread in ``group`` must participate in converged control flow. The
    payload size determines the number of consecutive values stored per thread.
    Contiguous operands are traversed in linear storage order; multidimensional
    logical indexing is not applied.

    Args:
        group: The current CUDA thread block.
        destination: Contiguous pointer-backed output memory.
        items: Fixed-size payload stored by each thread.
        valid_items: Optional valid element count for a partial block tile.
        offset: Optional element offset from the output pointer.

    Returns:
        ``None``.

    Raises:
        TypeError: If ``group`` is not a ``ThreadGroup``.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel stores a partial block tile:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """
