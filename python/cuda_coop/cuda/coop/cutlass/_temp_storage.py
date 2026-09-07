# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical temporary-storage identity for CUTLASS cooperative primitives."""

from __future__ import annotations

from ._dsl._temp_storage import TempStorageBase


class TempStorage(TempStorageBase):
    """Identity-scoped scratch planner for CUTLASS block collectives.

    ``coop.TempStorage`` is the canonical public spelling. The scoped
    ``coop._block.TempStorage`` name remains an identity alias for compatibility.
    """

    scope = "cuda.coop.cutlass"

    def sync(self) -> None:
        """Synchronize the threads that may reuse this block scratch."""

        from ._thread_group import this_block

        this_block().sync()


__all__ = ["TempStorage"]
