# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

class ThreadData:
    """Create an uninitialized per-thread register payload.

    The active backend owns the concrete payload type. When ``dtype`` is
    omitted, a primitive may infer it from its inputs for use by later
    primitives.

    Args:
        items_per_thread: Number of consecutive values owned by each thread.
        dtype: Optional portable numeric dtype. A primitive may infer it from
            its inputs.

    Returns:
        The active compiler backend's fixed-size payload object.

    Raises:
        ValueError: If ``items_per_thread`` is not positive.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel creates and uses a per-thread payload:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    dtype: Any | None

    def __init__(self, items_per_thread: int, dtype: Any = None) -> None: ...
    @property
    def items_per_thread(self) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...
    def __setitem__(self, index: int, value: Any) -> None: ...
