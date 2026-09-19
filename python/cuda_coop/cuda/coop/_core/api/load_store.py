# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common cooperative load and store entry points.

These frontends validate the shared algorithm subset and delegate one call to
the active compiler backend. ThreadData allocation and backend-specific CUB
selection remain outside this module.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _common_selector,
    _group_primitive_marker,
    _validate_common_operation_group,
)
from ._payload import (
    ThreadDataLike,
    _common_thread_data_extent,
    _ReadableThreadDataLike,
    _validate_common_integer_value,
    _validate_common_numeric_scalar,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)

_I32_MAX = (1 << 31) - 1
_I64_MAX = (1 << 63) - 1
_COMMON_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
)
_WARP_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
    }
)


def _validate_common_load_store_options(
    operation: str,
    group: ThreadGroup,
    *,
    algorithm: Any,
    payload: Any,
    valid_items: Any,
    oob_default: Any = None,
    offset: Any,
    temp_storage: Any,
) -> None:
    """Enforce the group-dependent common overload matrix."""

    if _backend_module_name() is None:
        return
    _validate_common_operation_group(operation, group)
    if operation == "load" and oob_default is not None and valid_items is None:
        raise ValueError("cuda.coop.load oob_default requires valid_items")
    if valid_items is not None:
        static_valid_items = _validate_common_integer_value(
            operation,
            "valid_items",
            valid_items,
        )
        if static_valid_items is not None:
            if not 0 <= static_valid_items <= _I32_MAX:
                raise ValueError(
                    f"cuda.coop.{operation} valid_items must be between 0 "
                    "and 2147483647"
                )
            if group.static_size is not None:
                items_per_thread = (
                    _common_thread_data_extent(
                        operation,
                        "output" if operation == "load" else "value",
                        payload,
                    )
                    if isinstance(payload, _ReadableThreadDataLike)
                    else 1
                )
                tile_items = group.static_size * items_per_thread
                if static_valid_items > tile_items:
                    raise ValueError(
                        f"cuda.coop.{operation} valid_items "
                        f"{static_valid_items} exceeds group tile size "
                        f"{tile_items}"
                    )
    if oob_default is not None:
        _validate_common_numeric_scalar(operation, "oob_default", oob_default)
    if offset is not None:
        static_offset = _validate_common_integer_value(
            operation,
            "offset",
            offset,
        )
        if static_offset is not None and not 0 <= static_offset <= _I64_MAX:
            raise ValueError(
                f"cuda.coop.{operation} offset must be between 0 and "
                "9223372036854775807"
            )
    if group.kind in {"warp", "threads_within_warp"}:
        if algorithm not in _WARP_LOAD_STORE_ALGORITHMS:
            raise ValueError(
                f"cuda.coop.{operation} algorithm {algorithm!r} is supported "
                "only for block groups"
            )
        if temp_storage is not None:
            raise ValueError(
                f"cuda.coop.{operation} temp_storage is not supported for "
                "Warp groups; omit it so the implementation can provide "
                "per-group storage"
            )
    elif temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)


@_common_group_operation(
    "load",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadDataLike[Any],
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Load a group tile from memory into per-thread values.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating threads; see :ref:`thread groups <coop-thread-groups>`.
        Supports blocks and physical or logical warps. Warp loads require
        an enclosing block size divisible by 32.
    source : array
        One-dimensional contiguous source array in device-accessible memory.
        Its element dtype must match ``output``; an untyped ``ThreadData``
        infers its dtype from this array. The array must contain all elements
        selected by ``offset`` and ``valid_items``.
    output : cuda.coop.ThreadDataLike
        Writable :ref:`per-thread payload <coop-thread-data>`.
        Load populates this payload in place. The group's tile contains
        ``group_size * items_per_thread`` elements.
    algorithm : str, optional
        Compile-time load algorithm, default ``"direct"``. ``"direct"`` gives
        each thread consecutive elements (blocked order); ``"striped"`` gives
        neighboring threads neighboring elements at each item index.
        ``"vectorize"`` uses vector accesses when possible, and ``"transpose"``
        uses shared scratch to rearrange striped accesses into blocked order.
        Both return blocked order. Blocks also support ``"warp_transpose"``
        and ``"warp_transpose_timesliced"``, which return blocked order and
        require a block size divisible by 32.
    valid_items : int or integer scalar, optional
        Number of valid elements in the group's tile, shared by all threads
        in that group. Supply a value between zero and the tile size,
        inclusive. ``None`` loads the full tile. Slots beyond this valid
        prefix retain their previous values unless ``oob_default`` is given.
    oob_default : numeric scalar, optional
        Value written to slots beyond ``valid_items``. Requires an explicit
        ``valid_items`` count. For example, use zero to pad a partial tile
        before summing it. A runtime value must have the payload dtype and
        be uniform across the group. ``None`` leaves those slots unchanged;
        initialize them before reading them.
    offset : int or integer scalar, optional
        Nonnegative offset in elements from the start of ``source``, uniform
        across the group. ``None`` means zero. For block tiles, supply the
        block's starting offset explicitly. For Warp tiles, the backend adds
        ``(linear_thread_rank // group_size) * tile_size`` automatically;
        do not include that within-block group offset a second time.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for block
        transpose-family algorithms. ``None`` uses automatic scratch.
        Direct, striped, and vectorized loads need no shared scratch.
        Warp loads require ``None``.

    Returns
    -------
    None
        The call populates ``output`` in place.

    See Also
    --------
    :cpp:class:`cub::BlockLoad`, :cpp:class:`cub::WarpLoad`
        C++ block and warp Load primitives.

    Examples
    --------
    Copy an array with Numba-CUDA-MLIR, using 128 threads and two values per
    thread. Each block loads up to 256 elements. The last block pads its
    missing values with zero and stores only the valid prefix.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_load_example.py
        :language: python
        :start-after: # example-begin
        :end-before: # example-end
        :dedent: 4

    The qualified import activates the Numba-CUDA-MLIR backend even if another
    module imported ``cuda.coop`` first. Use the qualified
    ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _common_selector(
        "load", "algorithm", algorithm, _COMMON_LOAD_STORE_ALGORITHMS
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "load",
            "output",
            output,
            allow_untyped_thread_data=True,
            require_thread_data=True,
        )
    _validate_common_load_store_options(
        "load",
        group,
        algorithm=algorithm,
        payload=output,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )

    _group_primitive_marker(
        "load",
        group,
        source,
        output,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )


@_common_group_operation(
    "store",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def store(
    group: ThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Store a group tile from per-thread values into memory.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating threads; see :ref:`thread groups <coop-thread-groups>`.
        Supports blocks and physical or logical warps. Warp stores require
        an enclosing block size divisible by 32.
    destination : array
        Writable one-dimensional contiguous array in device-accessible
        memory, with the same element dtype as ``value``. It must contain
        every element selected by ``offset`` and ``valid_items``.
    value : numeric scalar or cuda.coop.ThreadDataLike
        This thread's value or readable :ref:`payload <coop-thread-data>`.
        Initialize every item that will be stored. The tile contains
        ``group_size * items_per_thread`` elements, with one item per thread
        for a scalar. Store preserves the input, including when its algorithm
        rearranges values internally.
    algorithm : str, optional
        Compile-time store algorithm, default ``"direct"``. ``"direct"``
        expects blocked values; ``"striped"`` expects striped values.
        ``"vectorize"`` and ``"transpose"`` also expect blocked values and
        use vector accesses or shared-memory rearrangement, respectively.
        Blocks additionally support ``"warp_transpose"`` and
        ``"warp_transpose_timesliced"``, both requiring a block size divisible
        by 32. See :ref:`data layouts <coop-data-layouts>` before pairing
        different Load and Store algorithms.
    valid_items : int or integer scalar, optional
        Number of valid elements in the group's tile, uniform across the
        group and between zero and the tile size, inclusive. ``None`` stores
        the entire tile. Elements outside the valid prefix are not written.
    offset : int or integer scalar, optional
        Nonnegative offset in elements from the start of ``destination``,
        uniform across the group. ``None`` means zero. Supply each block's
        origin explicitly. Warp stores also add the within-block group
        origin automatically, using the same addressing rule as
        :func:`cuda.coop.load`.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-temp-storage>` for block
        transpose-family algorithms. ``None`` uses automatic scratch.
        Direct, striped, and vectorized stores need no shared scratch.
        Warp stores require ``None``.

    Returns
    -------
    None
        The call writes to ``destination`` and leaves ``value`` unchanged.

    See Also
    --------
    :cpp:class:`cub::BlockStore`, :cpp:class:`cub::WarpStore`
        C++ block and warp Store primitives.

    Examples
    --------
    Store a partial tile at an element offset. The untouched prefix and
    suffix keep their sentinel values. A second output checks that transpose
    Store preserved every thread's input payload.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_store_example.py
        :language: python
        :start-after: # example-begin
        :end-before: # example-end
        :dedent: 4

    See :ref:`participation and synchronization <coop-participation>` for
    control-flow requirements at primitive calls.
    """

    algorithm = _common_selector(
        "store", "algorithm", algorithm, _COMMON_LOAD_STORE_ALGORITHMS
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "store",
            "value",
            value,
            allow_readonly_thread_data=True,
        )
    _validate_common_load_store_options(
        "store",
        group,
        algorithm=algorithm,
        payload=value,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )

    _group_primitive_marker(
        "store",
        group,
        destination,
        value,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )


__all__ = ["load", "store"]
