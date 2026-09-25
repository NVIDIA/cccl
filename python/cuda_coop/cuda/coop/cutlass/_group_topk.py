# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Input-preserving TopK selection for complete one-dimensional CuTe blocks."""

from cuda.coop._core.thread_group import ThreadGroup

from ._temp_storage import TempStorage
from ._thread_data import _snapshot_readable_payload


def _topk(group, keys, values, *, selection, k, valid_items, temp_storage):
    primitive = f"topk_{selection}_{'keys' if values is None else 'pairs'}"
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.cutlass.{primitive} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError("TopK supports only complete this_block() groups")
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("TopK temp_storage must be CUTLASS TempStorage")
    keys = _snapshot_readable_payload(keys, name="keys", primitive=primitive)
    if values is not None:
        values = _snapshot_readable_payload(values, name="values", primitive=primitive)
        if keys.items_per_thread != values.items_per_thread:
            raise ValueError("TopK keys and values must have matching items_per_thread")

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._topk import provider_topk

    return provider_topk(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        values=values,
        selection=selection,
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_min_keys(group, keys, /, *, k, valid_items=None, temp_storage=None):
    """Select the block's smallest keys without changing the input.

    This qualified form of :func:`cuda.coop.topk_min_keys` also accepts CuTe
    register payloads. All block threads participate, including those whose
    input items are outside the valid prefix.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block from ``this_block()``.
    keys : ThreadData or CuTe register payload
        Per-thread keys in blocked order with a positive, compile-time extent.
        Read-only payloads, CuTe register-memory tensors, and ``TensorSSA``
        values are accepted. Supported element types are signed and unsigned
        8-, 16-, 32-, and 64-bit integers, ``Float32``, and ``Float64``.
        Register inputs are converted through
        :meth:`ThreadData.from_payload <cuda.coop.cutlass.ThreadData.from_payload>`.
    k : integer
        Block-uniform number of requested items in ``[0, N]``, where
        ``N = block_threads * items_per_thread``. May be static or runtime.
    valid_items : integer, optional
        Block-uniform length of the valid blocked input prefix, also in
        ``[0, N]``. Omit it to use the entire tile. If ``k > valid_items``,
        all valid items are selected.
    temp_storage : TempStorage, optional
        Explicit block scratch. Omit it for automatic allocation. Requested
        alignment is a minimum. With ``auto_sync=False``, synchronize the
        block before reusing the storage.

    Returns
    -------
    ThreadData
        Fresh selected keys with the input element type and per-thread extent,
        including for CuTe register inputs. Only the first
        ``min(k, valid_items)`` blocked output positions are defined; omit
        ``valid_items`` to use ``N`` in this expression. Thread ``t`` owns
        positions ``t * items_per_thread + i``. Remaining positions must not
        be read or stored. The selected keys are unsorted.

    Notes
    -----
    Selection and ordering among equal keys are unspecified. Positive and
    negative floating-point zero compare equally, and selected keys retain
    their original bits. NaNs have no guaranteed numeric ordering.

    Runtime ``k`` and ``valid_items`` accept signed integers up to 64 bits or
    unsigned integers up to 32 bits. Invalid static counts fail compilation;
    invalid runtime counts trap before conversion to CUB's count type.
    Zero ``k`` or ``valid_items`` leaves no defined output positions.

    See Also
    --------
    cuda.coop.cutlass.topk_max_keys
    cuda.coop.cutlass.topk_min_pairs
    """
    return _topk(
        group,
        keys,
        None,
        selection="min",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_max_keys(group, keys, /, *, k, valid_items=None, temp_storage=None):
    """Select the block's largest keys without changing the input.

    This qualified form of :func:`cuda.coop.topk_max_keys` accepts the same
    inputs and controls as :func:`cuda.coop.cutlass.topk_min_keys`, selecting
    the largest keys. All threads in the block must participate.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block from ``this_block()``.
    keys : ThreadData or CuTe register payload
        Fixed-size per-thread keys in blocked order. Read-only payloads,
        register-memory tensors, and ``TensorSSA`` values are accepted. The
        element types are those supported by ``topk_min_keys``.
    k : integer
        Block-uniform count in ``[0, block_threads * items_per_thread]``.
        Static and runtime counts are accepted.
    valid_items : integer, optional
        Block-uniform length of the valid input prefix. Defaults to the whole
        tile and has the same range as ``k``.
    temp_storage : TempStorage, optional
        Explicit block scratch. Omit it for automatic allocation. Synchronize
        before reuse if its descriptor specifies ``auto_sync=False``.

    Returns
    -------
    ThreadData
        Fresh selected keys with the input element type and extent. Only the
        first ``min(k, valid_items)`` blocked positions are defined, or
        ``min(k, N)`` when ``valid_items`` is omitted, where
        ``N = block_threads * items_per_thread``. The selected keys are
        unsorted; remaining positions must not be consumed.
        Ties, floating-point ordering, and count validation follow
        ``topk_min_keys``.

    See Also
    --------
    cuda.coop.cutlass.topk_min_keys
    cuda.coop.cutlass.topk_max_pairs
    """
    return _topk(
        group,
        keys,
        None,
        selection="max",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_min_pairs(group, keys, values, /, *, k, valid_items=None, temp_storage=None):
    """Select the smallest keys and their associated values.

    This qualified form of :func:`cuda.coop.topk_min_pairs` accepts CuTe
    register payloads for either operand. Both inputs remain unchanged.
    Participation, count validation, and key ordering follow
    :func:`cuda.coop.cutlass.topk_min_keys`.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block. All block threads must participate.
    keys, values : ThreadData or CuTe register payload
        Fixed-size payloads in blocked order, with equal per-thread extents.
        Either operand may be read-only or use a CuTe register representation.
        Key and value types are independent: signed and unsigned 8-, 16-,
        32-, and 64-bit integers, ``Float32``, and ``Float64`` are supported.
    k : integer
        Block-uniform count in ``[0, block_threads * items_per_thread]``.
        Static and runtime counts are accepted.
    valid_items : integer, optional
        Block-uniform length of the valid blocked input prefix. Defaults to
        the entire tile and has the same range as ``k``. If ``k`` exceeds
        this count, all valid key/value pairs are selected.
    temp_storage : TempStorage, optional
        Explicit block scratch. Omit it for automatic allocation. Synchronize
        before reuse if its descriptor specifies ``auto_sync=False``.

    Returns
    -------
    tuple[ThreadData, ThreadData]
        Fresh selected keys and associated values, each retaining its input
        element type and extent. Only the first ``min(k, valid_items)`` blocked
        positions are defined, or ``min(k, N)`` when ``valid_items`` is omitted,
        where ``N = block_threads * items_per_thread``. Each selected value
        stays with its original key. Results are unsorted, with no stability
        guarantee for ties; undefined tail positions must not be read or stored.

    See Also
    --------
    cuda.coop.cutlass.topk_min_keys
    cuda.coop.cutlass.topk_max_pairs
    """
    if values is None:
        raise TypeError("TopK values must be a numeric ThreadData payload")
    return _topk(
        group,
        keys,
        values,
        selection="min",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def topk_max_pairs(group, keys, values, /, *, k, valid_items=None, temp_storage=None):
    """Select the largest keys and their associated values.

    This qualified form of :func:`cuda.coop.topk_max_pairs` accepts the same
    inputs and controls as :func:`cuda.coop.cutlass.topk_min_pairs`, selecting
    the largest keys. Both inputs remain unchanged.

    Parameters
    ----------
    group : ThreadGroup
        Complete one-dimensional block. All block threads must participate.
    keys, values : ThreadData or CuTe register payload
        Fixed-size payloads in blocked order, with equal per-thread extents.
        Read-only payloads and CuTe register representations are accepted.
        Each operand independently uses a numeric element type supported by
        ``topk_min_pairs``.
    k : integer
        Block-uniform count in ``[0, block_threads * items_per_thread]``.
        Static and runtime counts are accepted.
    valid_items : integer, optional
        Block-uniform length of the valid input prefix. Defaults to the entire
        tile and has the same range as ``k``. If ``k`` exceeds this count,
        all valid key/value pairs are selected.
    temp_storage : TempStorage, optional
        Explicit block scratch. Omit it for automatic allocation. Synchronize
        before reuse if its descriptor specifies ``auto_sync=False``.

    Returns
    -------
    tuple[ThreadData, ThreadData]
        Fresh selected keys and associated values, retaining each input's
        element type and extent. Only the first ``min(k, valid_items)`` blocked
        positions are defined, or ``min(k, N)`` when ``valid_items`` is omitted,
        where ``N = block_threads * items_per_thread``. Associations are
        preserved. Results are unsorted, with no stability guarantee for ties.
        Floating-point ordering, count validation, and undefined tails follow
        :func:`cuda.coop.cutlass.topk_min_keys`.

    See Also
    --------
    cuda.coop.cutlass.topk_min_pairs
    cuda.coop.cutlass.topk_max_keys
    """
    if values is None:
        raise TypeError("TopK values must be a numeric ThreadData payload")
    return _topk(
        group,
        keys,
        values,
        selection="max",
        k=k,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


__all__ = ["topk_min_keys", "topk_min_pairs", "topk_max_keys", "topk_max_pairs"]
