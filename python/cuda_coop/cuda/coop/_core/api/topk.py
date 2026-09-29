# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Out-of-place block TopK operations."""

from __future__ import annotations

from typing import Any

from ..thread_group import CoopCompilerContextRequiredError, ThreadGroup
from ._dispatch import (
    _portable_group_operation,
)
from ._payload import TempStorageLike


@_portable_group_operation("topk_min_keys", group_kinds=("block",))
def topk_min_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    k: Any,
    valid_items: object = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return the minimum keys in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_min_keys must be called from a supported GPU kernel."
    )


@_portable_group_operation("topk_min_pairs", group_kinds=("block",))
def topk_min_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    k: Any,
    valid_items: object = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return the minimum pairs in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_min_pairs must be called from a supported GPU kernel."
    )


@_portable_group_operation("topk_max_keys", group_kinds=("block",))
def topk_max_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    k: Any,
    valid_items: object = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return the maximum keys in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_max_keys must be called from a supported GPU kernel."
    )


@_portable_group_operation("topk_max_pairs", group_kinds=("block",))
def topk_max_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    k: Any,
    valid_items: object = None,
    temp_storage: TempStorageLike | None = None,
) -> Any:
    """Return the maximum pairs in an unsorted blocked prefix.

    Inputs are preserved. Only the first ``min(k, valid_items)`` positions
    are defined; omitted ``valid_items`` means the full tile. Ties have
    unspecified order and selection. Every block member participates with
    uniform integer controls in ``[0, block_threads * items_per_thread]``.
    Invalid runtime controls trap before narrowing to CUB's integer ABI.
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_max_pairs must be called from a supported GPU kernel."
    )


__all__ = ["topk_max_keys", "topk_max_pairs", "topk_min_keys", "topk_min_pairs"]
