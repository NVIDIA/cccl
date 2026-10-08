# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adapt qualified radix inputs and preserve public result shapes.

Sort accepts scalar or readable/register payloads and returns fresh keys and
optional associated values. Rank returns Int32 positions and can also write a
separate bin-prefix payload. All forms require a complete block.
"""

from __future__ import annotations

from cuda.coop._core.thread_group import ThreadGroup
from cuda.coop._typing import CompilerIntegerLike, CompilerScalarLike

from ._temp_storage import TempStorage
from ._thread_data import ThreadData, _snapshot_readable_payload

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
from typing import TypeVar

from cuda.coop._typing import IntegerValue

from .._core.api.thread_group import BlockGroup
from .._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    TempStorageLike,
    ThreadDataLike,
)
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample

_KeyT = TypeVar(
    "_KeyT",
    bound=(
        "int | np.int32"
        " | np.uint32 | np.int64"
        " | np.uint64 | CompilerIntegerLike"
        " | float | np.float32"
        " | np.float64 | CompilerScalarLike"
    ),
)

_ValueT = TypeVar("_ValueT", bound=CommonNumericScalar)

_RankKeyT = TypeVar(
    "_RankKeyT",
    bound=(
        "int | np.int32"
        " | np.uint32 | np.int64"
        " | np.uint64 | CompilerIntegerLike"
    ),
)


def _validate_group(group):
    """Require the physical block used by CUB radix collectives."""

    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass radix group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            "cuda.coop.cutlass radix requires a complete physical block"
        )


def _input(value, name, primitive):
    """Copy readable payloads and adapt register containers.

    Direct ``cuda.coop.cutlass`` calls may also pass a scalar key or value.
    Calls through the shared ``cuda.coop`` API require a readable payload.
    """

    return _snapshot_readable_payload(
        value, name=name, primitive=primitive, allow_scalar=True
    )


def _sort(
    group,
    keys,
    values,
    *,
    begin_bit,
    end_bit,
    descending,
    blocked_to_striped,
    temp_storage,
):
    """Validate sort controls and matching key/value shapes.

    Adapt operands before comparing scalar versus payload form and item count.
    The provider then resolves independent dtypes, bit bounds, exact scratch
    layout, and the blocked or striped output method.
    """

    _validate_group(group)
    primitive = "radix_sort_keys" if values is None else "radix_sort_pairs"
    for name, value in (
        ("descending", descending),
        ("blocked_to_striped", blocked_to_striped),
    ):
        if not isinstance(value, bool):
            raise TypeError(
                f"cuda.coop.cutlass.{primitive} {name} must be "
                "a compile-time bool"
            )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError(
            "cuda.coop.cutlass radix temp_storage must be TempStorage"
        )
    keys = _input(keys, "keys", primitive)
    if values is not None:
        values = _input(values, "values", primitive)
        if isinstance(keys, ThreadData) != isinstance(values, ThreadData):
            raise TypeError(
                "radix_sort_pairs keys and values must have matching "
                "scalar or array shapes"
            )
        if (
            isinstance(keys, ThreadData)
            and keys.items_per_thread != values.items_per_thread
        ):
            raise ValueError(
                "radix_sort_pairs keys and values must have matching "
                "items_per_thread"
            )
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._radix_sort import provider_radix_sort

    return provider_radix_sort(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        values=values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_sort_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _KeyT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> ThreadData | _KeyT:
    """Return stable radix-sorted keys without changing the input.

    This qualified form of :func:`cuda.coop.radix_sort_keys` adds scalar and
    CuTe register inputs, floating-point keys, and striped output. Every block
    thread participates with the same controls and per-thread extent.

    Parameters
    ----------
    group : ThreadGroup
        Complete physical block from ``this_block()``. Multidimensional blocks
        are supported. The block tile must contain at most 65,535 items.
    keys : scalar, ThreadData, or CuTe register payload
        Keys in blocked order, with element type ``Int32``, ``Uint32``,
        ``Int64``, ``Uint64``, ``Float32``, or ``Float64``. A scalar contributes
        one key per thread. Register-memory tensors and ``TensorSSA`` values are
        converted through :meth:`ThreadData.from_payload
        <cuda.coop.cutlass.ThreadData.from_payload>`; read-only payloads are
        also accepted.
    begin_bit, end_bit : integer, optional
        Block-uniform half-open interval in CUB's transformed key bits, with
        ``0 <= begin_bit < end_bit <= key_width``. Begin defaults to zero;
        omitted end selects the key width. Either bound may be a runtime CuTe
        integer, signed up to 64 bits or unsigned up to 32 bits. Invalid
        runtime bounds trap before narrowing to CUB's integer arguments.
    descending : bool, optional
        Compile-time selector for descending order. The default is ascending.
    temp_storage : TempStorage, optional
        Explicit block scratch, otherwise allocated automatically. Requested
        alignment is a minimum. With ``auto_sync=False``, synchronize the
        block before reusing the storage.
    blocked_to_striped : bool, optional
        Compile-time output-layout selector. The default is blocked output,
        where thread ``t`` owns sorted indices ``t * items_per_thread + i``.
        ``True`` returns striped output, with indices ``i * block_threads + t``.
        Use a matching striped Store to write those registers in sorted order.

    Returns
    -------
    CuTe scalar or ThreadData
        Sorted keys with the input element type. Scalar inputs return a CuTe
        scalar; payload inputs return fresh ``ThreadData`` of the same extent,
        including when the input uses a CuTe register representation. Equal
        selected digits retain blocked input order in either direction.

    Notes
    -----
    CUB transforms keys before selecting the requested bits. Signed integers
    invert their sign bit. Floating-point keys invert all bits when negative
    and only the sign bit otherwise. Returned keys keep their original bits.
    Positive and negative zero compare equally; NaNs follow the transformed
    bit order, with no guarantee of numeric ordering.

    See Also
    --------
    cuda.coop.cutlass.radix_sort_pairs
    cuda.coop.cutlass.radix_rank_keys

    Examples
    --------
    Sort all key bits in ascending order, then sort key/index pairs by
    their low four bits in descending order. The latter produces striped
    registers, so its stores use ``algorithm="striped"``.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_radix_sort_examples.py
        :language: python
        :start-after: # qualified-radix-sort-example-begin
        :end-before: # qualified-radix-sort-example-end
        :dedent: 4
    """
    return _sort(
        group,
        keys,
        None,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_sort_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_KeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _KeyT,
    values: CommonThreadDataLike[_ValueT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ValueT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadData, ThreadData] | tuple[_KeyT, _ValueT]:
    """Return stable radix-sorted keys and their associated values.

    This qualified form of :func:`cuda.coop.radix_sort_pairs` adds scalar and
    CuTe register inputs, floating-point keys, and striped output. Neither
    input is modified. Key transformations and stable tie handling follow
    :func:`cuda.coop.cutlass.radix_sort_keys`.

    Parameters
    ----------
    group : ThreadGroup
        Complete physical block. Every thread participates with the same
        controls; the block tile contains at most 65,535 items.
    keys, values : scalar, ThreadData, or CuTe register payload
        Matching scalar or fixed-size payload shapes. Payloads have equal
        per-thread extents and use blocked input order. Key types are ``Int32``,
        ``Uint32``, ``Int64``, ``Uint64``, ``Float32``, and ``Float64``. Values
        may independently use signed and unsigned 8-, 16-, 32-, and 64-bit
        integers, ``Float32``, or ``Float64``. Read-only and CuTe register
        payloads are accepted; one operand may be ``ThreadData`` while the other
        is a register payload of the same extent.
    begin_bit, end_bit : integer, optional
        Block-uniform half-open interval in transformed key bits. Begin
        defaults to zero and end to the key width. Runtime integer bounds
        and validation follow ``radix_sort_keys``.
    descending : bool, optional
        Compile-time order selector. The default is ascending key order.
    temp_storage : TempStorage, optional
        Explicit block scratch. Omit for automatic allocation. Synchronize
        before reuse if its descriptor specifies ``auto_sync=False``.
    blocked_to_striped : bool, optional
        Compile-time selector for striped output, applied to both keys and
        values. The default is blocked output. Store both results with the
        corresponding layout.

    Returns
    -------
    tuple
        Sorted keys and associated values, each retaining its input element
        type. Scalar operands return CuTe scalars; payload operands return
        fresh ``ThreadData`` with the original per-thread extent. Associations
        are preserved, and equal selected digits retain blocked input order.

    See Also
    --------
    cuda.coop.cutlass.radix_sort_keys

    Examples
    --------
    Stably order key/index pairs by their low four bits in descending
    order. Match striped output registers with striped stores.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_radix_sort_examples.py
        :language: python
        :start-after: # qualified-radix-sort-example-begin
        :end-before: # qualified-radix-sort-example-end
        :dedent: 4
    """
    if values is None:
        raise TypeError(
            "radix_sort_pairs values must be a numeric scalar or payload"
        )
    return _sort(
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_rank_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_RankKeyT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _RankKeyT,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadDataLike[np.int32]
    | ThreadDataLike[CompilerIntegerLike]
    | None = None,
) -> ThreadData | CompilerIntegerLike:
    """Return stable digit ranks and optional exclusive bin prefixes.

    This qualified form of :func:`cuda.coop.radix_rank_keys` adds scalar
    and CuTe register inputs and a writable prefix output. Keys remain
    unchanged.

    Parameters
    ----------
    group : ThreadGroup
        Complete physical block. Every thread participates with identical
        compile-time controls and payload extents. Multidimensional blocks
        are supported; the block tile contains at most 65,535 items.
    keys : scalar, ThreadData, or CuTe register payload
        ``Int32``, ``Uint32``, ``Int64``, or ``Uint64`` keys in blocked order.
        A scalar contributes one key per thread. CuTe register-memory tensors,
        ``TensorSSA`` values, and read-only payloads are accepted. Signed keys
        invert their sign bit before digit extraction.
    begin_bit, end_bit : int, optional
        Compile-time half-open interval within the key width, selecting one
        through eight bits. Begin defaults to zero. Omitted end is begin
        plus ``radix_bits``, or begin plus four if both are omitted.
    radix_bits : int, optional
        Compile-time digit width. If ``end_bit`` is explicit, this must equal
        ``end_bit - begin_bit``.
    descending : bool, optional
        Compile-time selector for descending digit order. The default is
        ascending order.
    exclusive_digit_prefix : ThreadData, optional
        Writable ``Int32`` side output, distinct from ``keys``. An omitted
        dtype is inferred as ``Int32``. Each thread provides
        ``P = max(1, ceil(2**radix_bits / block_threads))`` items. Thread ``t``
        owns ascending bin indices ``t * P + i`` in either ordering direction.
        A bin's prefix counts keys with smaller digits for ascending order or
        greater digits for descending order. Slots past the last bin are
        undefined and must not be read or stored. Register tensors cannot be
        passed directly as this output.

    Returns
    -------
    Int32 or ThreadData
        Zero-based ranks with fixed ``Int32`` element type. Scalar input
        returns a CuTe scalar; payload input returns fresh ``ThreadData`` with
        the input extent. Equal digits retain blocked input order. Element
        ``i`` at thread ``t`` receives the sorted position of the original
        key at blocked index ``t * items_per_thread + i``.

    Notes
    -----
    Scratch allocation and synchronization before reuse are automatic.

    See Also
    --------
    cuda.coop.cutlass.radix_sort_keys

    Examples
    --------
    Rank the low four bits and retrieve the starting offset of each of
    the sixteen digit bins. Only the sixteen defined prefix slots are stored.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_radix_rank_examples.py
        :language: python
        :start-after: # qualified-radix-rank-example-begin
        :end-before: # qualified-radix-rank-example-end
        :dedent: 4
    """
    _validate_group(group)
    if not isinstance(descending, bool):
        raise TypeError(
            "cuda.coop.cutlass.radix_rank_keys descending must be "
            "a compile-time bool"
        )
    if exclusive_digit_prefix is keys:
        raise ValueError(
            "radix_rank_keys exclusive_digit_prefix must be distinct from keys"
        )
    keys = _input(keys, "keys", "radix_rank_keys")
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._radix_sort import provider_radix_rank

    return provider_radix_rank(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )


__all__ = ["radix_rank_keys", "radix_sort_keys", "radix_sort_pairs"]
