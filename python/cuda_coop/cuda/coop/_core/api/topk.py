# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Define block TopK calls and their selected-prefix contract.

The registered functions select minimum or maximum keys, optionally with
paired values. Compiler backends preserve the input payloads and return
new ones. The full payload shape is retained, but only the selected
prefix is defined. Python execution raises an error because these calls
require a GPU kernel.
"""

from __future__ import annotations

from typing import TypeVar

from ..._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegralScalar,
    ThreadDataLike,
)
from ..thread_group import CoopCompilerContextRequiredError
from ._dispatch import (
    _common_group_operation,
)
from .temp_storage import TempStorageLike as TempStorage
from .thread_group import BlockGroup

_K = TypeVar("_K", bound=CommonNumericScalar)

_V = TypeVar("_V", bound=CommonNumericScalar)


@_common_group_operation("topk_min_keys", group_kinds=("block",))
def topk_min_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_K]:
    """Select the smallest keys in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys : per-thread payload
        A new payload with the input dtype and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR implementation accepts signed
    runtime counts up to 64 bits and unsigned counts up to 32 bits.
    It rejects invalid static counts during compilation and traps
    on invalid runtime counts.

    Examples
    --------
    Select the smallest and largest eight keys from a partial tile. If
    fewer than eight keys are valid, store only that many results. The
    selected keys are unordered.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_topk_examples.py
        :language: python
        :start-after: # topk-keys-example-begin
        :end-before: # topk-keys-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_min_keys must be called from a supported GPU kernel."
    )


@_common_group_operation("topk_min_pairs", group_kinds=("block",))
def topk_min_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    values: CommonThreadDataLike[_V],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadDataLike[_K], ThreadDataLike[_V]]:
    """Select the smallest key/value pairs in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``.
    values : ThreadData
        Values paired with ``keys``, with the same per-thread extent.
        The value dtype may differ from the key dtype. It must be one
        of the numeric dtypes supported for keys.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys, selected_values : tuple of per-thread payloads
        New payloads with the input dtypes and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Each selected value remains paired with its original key.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR implementation accepts signed
    runtime counts up to 64 bits and unsigned counts up to 32 bits.
    It rejects invalid static counts during compilation and traps
    on invalid runtime counts.

    Examples
    --------
    Select the smallest eight keys and their original positions from a
    partial tile. Each selected position still identifies its key; the
    selected pairs are unordered.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_topk_examples.py
        :language: python
        :start-after: # topk-min-pairs-example-begin
        :end-before: # topk-min-pairs-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_min_pairs must be called from a supported GPU kernel."
    )


@_common_group_operation("topk_max_keys", group_kinds=("block",))
def topk_max_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_K]:
    """Select the largest keys in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys : per-thread payload
        A new payload with the input dtype and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR implementation accepts signed
    runtime counts up to 64 bits and unsigned counts up to 32 bits.
    It rejects invalid static counts during compilation and traps
    on invalid runtime counts.

    Examples
    --------
    Select the smallest and largest eight keys from a partial tile. If
    fewer than eight keys are valid, store only that many results. The
    selected keys are unordered.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_topk_examples.py
        :language: python
        :start-after: # topk-keys-example-begin
        :end-before: # topk-keys-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_max_keys must be called from a supported GPU kernel."
    )


@_common_group_operation("topk_max_pairs", group_kinds=("block",))
def topk_max_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    values: CommonThreadDataLike[_V],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadDataLike[_K], ThreadDataLike[_V]]:
    """Select the largest key/value pairs in a block.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block, obtained with ``this_block()``.
        Every thread in the block must call this operation.
    keys : ThreadData
        Fixed-size per-thread keys in blocked order. Supported dtypes
        are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``float32``, and ``float64``.
    values : ThreadData
        Values paired with ``keys``, with the same per-thread extent.
        The value dtype may differ from the key dtype. It must be one
        of the numeric dtypes supported for keys.
    k : int
        Requested number of selected items. May be static or runtime,
        must be uniform across the block, and must lie in ``[0, N]``,
        where ``N = block_threads * items_per_thread``.
    valid_items : int, optional
        Number of valid input items in the blocked tile prefix.
        Defaults to ``N`` and has the same range and uniformity
        requirements as ``k``. If ``k > valid_items``, all valid
        items are selected.
    temp_storage : TempStorage, optional
        Shared scratch descriptor. Omit it to let the compiler allocate
        storage. With ``auto_sync=False``, synchronize the block before
        reusing the descriptor in another primitive.

    Returns
    -------
    selected_keys, selected_values : tuple of per-thread payloads
        New payloads with the input dtypes and per-thread extent.
        Only the first ``min(k, valid_items)`` blocked tile positions
        are defined. Position ``thread_rank * items_per_thread + i``
        belongs to element ``i`` of that thread. Remaining positions
        must not be read or stored.

    Notes
    -----
    Selection preserves the input payloads. Results are unsorted;
    selection and ordering among equal keys are unspecified.
    Each selected value remains paired with its original key.
    Zero ``k`` or ``valid_items`` produces no defined output items.
    Positive and negative floating-point zero compare as equal, and
    selected keys retain their original bits. NaNs have no guaranteed
    numeric ordering.

    Call this operation inside a kernel compiled by a registered
    backend. The Numba-CUDA-MLIR implementation accepts signed
    runtime counts up to 64 bits and unsigned counts up to 32 bits.
    It rejects invalid static counts during compilation and traps
    on invalid runtime counts.

    Examples
    --------
    Select the largest eight keys and their original positions from a
    partial tile. Store only the selected prefix; the pairs are unordered.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_topk_examples.py
        :language: python
        :start-after: # topk-example-begin
        :end-before: # topk-example-end
        :dedent: 4
    """
    raise CoopCompilerContextRequiredError(
        "cuda.coop.topk_max_pairs must be called from a supported GPU kernel."
    )


__all__ = ["topk_max_keys", "topk_max_pairs", "topk_min_keys", "topk_min_pairs"]
