# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Literal, Protocol

from .. import Payload as _Payload
from .. import TempStorage as TempStorage
from .. import ThreadData as _ThreadData
from .._dsl._launch import CutlassLaunchMetadata as _CutlassLaunchMetadata
from .._dsl.block import BlockAdjacentDifferenceType as BlockAdjacentDifferenceType
from .._dsl.block import BlockDiscontinuityType as BlockDiscontinuityType
from .._dsl.block import BlockExchangeType as BlockExchangeType
from .._dsl.block import BlockRunLengthDecode as BlockRunLengthDecode
from .._dsl.block import BlockShuffleType as BlockShuffleType
from .._dsl.block import adjacent_difference as adjacent_difference
from .._dsl.block import (
    adjacent_difference_subtract_left as adjacent_difference_subtract_left,
)
from .._dsl.block import (
    adjacent_difference_subtract_right as adjacent_difference_subtract_right,
)
from .._dsl.block import discontinuity as discontinuity
from .._dsl.block import discontinuity_flag_heads as discontinuity_flag_heads
from .._dsl.block import (
    discontinuity_flag_heads_and_tails as discontinuity_flag_heads_and_tails,
)
from .._dsl.block import discontinuity_flag_tails as discontinuity_flag_tails
from .._dsl.block import exchange as exchange
from .._dsl.block import exchange_blocked_to_striped as exchange_blocked_to_striped
from .._dsl.block import (
    exchange_blocked_to_warp_striped as exchange_blocked_to_warp_striped,
)
from .._dsl.block import exchange_scatter_to_blocked as exchange_scatter_to_blocked
from .._dsl.block import exchange_scatter_to_striped as exchange_scatter_to_striped
from .._dsl.block import (
    exchange_scatter_to_striped_flagged as exchange_scatter_to_striped_flagged,
)
from .._dsl.block import (
    exchange_scatter_to_striped_guarded as exchange_scatter_to_striped_guarded,
)
from .._dsl.block import exchange_striped_to_blocked as exchange_striped_to_blocked
from .._dsl.block import (
    exchange_warp_striped_to_blocked as exchange_warp_striped_to_blocked,
)
from .._dsl.block import exclusive_scan as exclusive_scan
from .._dsl.block import exclusive_sum as exclusive_sum
from .._dsl.block import histogram as histogram
from .._dsl.block import inclusive_scan as inclusive_scan
from .._dsl.block import inclusive_sum as inclusive_sum
from .._dsl.block import radix_sort_keys_descending as radix_sort_keys_descending
from .._dsl.block import radix_sort_pairs_descending as radix_sort_pairs_descending
from .._dsl.block import reduce as reduce
from .._dsl.block import row_sum as row_sum
from .._dsl.block import run_length as run_length
from .._dsl.block import run_length_decode as run_length_decode
from .._dsl.block import scan as scan
from .._dsl.block import shuffle as shuffle
from .._dsl.block import shuffle_down as shuffle_down
from .._dsl.block import shuffle_offset as shuffle_offset
from .._dsl.block import shuffle_rotate as shuffle_rotate
from .._dsl.block import shuffle_up as shuffle_up
from .._dsl.block import sum as sum

class _DeferredValue(Protocol):
    """Deferred scoped callable that returns one transformed value."""

    def __call__(self, value: object, /, *args: object, **kwargs: object) -> object:
        """Invoke the captured cooperative specialization."""

class _DeferredPair(Protocol):
    """Deferred scoped callable that transforms key-value payloads."""

    def __call__(
        self,
        keys: object,
        values: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> tuple[object, object]:
        """Invoke the captured key-value specialization."""

class _DeferredLoad(Protocol):
    """Deferred scoped block-load callable."""

    def __call__(
        self,
        source: object,
        output: _ThreadData[Any] | None = None,
        /,
        *,
        items_per_thread: int | None = None,
        valid_items: object = None,
        num_valid_items: object = None,
        oob_default: object = None,
        offset: object = None,
        algorithm: object = "direct",
        dtype: object = None,
        threads_per_block: object = None,
        dim: object = None,
        temp_storage: object = None,
        payload: _Payload | Literal["prims"] | None = None,
        alignment: object = None,
        bounds_check: bool = False,
        is_volatile: bool = False,
        is_nontemporal: bool = False,
        is_invariant: bool = False,
        is_invariant_group: bool = False,
        ordering: object = "not_atomic",
        syncscope: object = None,
        loc: object = None,
        ip: object = None,
        launch_metadata: _CutlassLaunchMetadata | None = None,
        launch_meta: _CutlassLaunchMetadata | None = None,
        launch: _CutlassLaunchMetadata | None = None,
        launch_config: _CutlassLaunchMetadata | None = None,
    ) -> object:
        """Load from ``source`` using captured defaults."""

class _DeferredStore(Protocol):
    """Deferred scoped block-store callable."""

    def __call__(
        self,
        destination: object,
        value: object,
        /,
        *,
        items_per_thread: int | None = None,
        valid_items: object = None,
        num_valid_items: object = None,
        algorithm: object = "direct",
        offset: object = None,
        dtype: object = None,
        threads_per_block: object = None,
        dim: object = None,
        temp_storage: object = None,
        payload: _Payload | Literal["prims"] | None = None,
        alignment: object = None,
        bounds_check: bool = False,
        is_volatile: bool = False,
        is_nontemporal: bool = False,
        ordering: object = "not_atomic",
        syncscope: object = None,
        loc: object = None,
        ip: object = None,
        launch_metadata: _CutlassLaunchMetadata | None = None,
        launch_meta: _CutlassLaunchMetadata | None = None,
        launch: _CutlassLaunchMetadata | None = None,
        launch_config: _CutlassLaunchMetadata | None = None,
    ) -> None:
        """Store ``value`` using captured defaults."""

def load(
    source: Any,
    output: _ThreadData[Any] | None = None,
    /,
    *,
    items_per_thread: int | None = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    algorithm: Any = "direct",
    dtype: Any = None,
    threads_per_block: Any = None,
    dim: Any = None,
    temp_storage: Any = None,
    payload: _Payload | Literal["prims"] | None = None,
    alignment: Any = None,
    bounds_check: bool = False,
    is_volatile: bool = False,
    is_nontemporal: bool = False,
    is_invariant: bool = False,
    is_invariant_group: bool = False,
    ordering: Any = "not_atomic",
    syncscope: Any = None,
    loc: Any = None,
    ip: Any = None,
    launch_metadata: _CutlassLaunchMetadata | None = None,
    launch_meta: _CutlassLaunchMetadata | None = None,
    launch: _CutlassLaunchMetadata | None = None,
    launch_config: _CutlassLaunchMetadata | None = None,
) -> Any:
    """Load a block tile into per-thread data.

    CUTLASS tensor sources use the default tensor path. Public
    ``cutlass.Array`` sources dispatch to the Prims array path. Pass
    ``payload=Payload.PRIMS`` to materialize per-thread register data from an
    otherwise memory-backed source. Prims-specific memory controls such as
    ``bounds_check=`` also select that path. ``dtype=`` alone remains on the
    tensor route, and ``offset=`` remains available to canonical CUB.
    """

def store(
    destination: Any,
    value: Any,
    /,
    *,
    items_per_thread: int | None = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    algorithm: Any = "direct",
    offset: Any = None,
    dtype: Any = None,
    threads_per_block: Any = None,
    dim: Any = None,
    temp_storage: Any = None,
    payload: _Payload | Literal["prims"] | None = None,
    alignment: Any = None,
    bounds_check: bool = False,
    is_volatile: bool = False,
    is_nontemporal: bool = False,
    ordering: Any = "not_atomic",
    syncscope: Any = None,
    loc: Any = None,
    ip: Any = None,
    launch_metadata: _CutlassLaunchMetadata | None = None,
    launch_meta: _CutlassLaunchMetadata | None = None,
    launch: _CutlassLaunchMetadata | None = None,
    launch_config: _CutlassLaunchMetadata | None = None,
) -> None:
    """Store per-thread data into a block tile.

    CUTLASS tensor destinations use the default tensor path. Public
    ``cutlass.Array`` destinations dispatch to the Prims array path. Pass
    ``payload=Payload.PRIMS`` to store through that path. Prims-specific
    bounds/memory controls such as ``bounds_check=`` also select it. ``dtype=``
    and ``ThreadData.dtype`` alone remain on the tensor route.
    """

def make_load(
    dtype: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    algorithm: Any = "direct",
    *,
    dim: Any = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    oob_default: Any = None,
    payload: _Payload | Literal["prims"] | None = None,
    offset: Any = None,
    alignment: Any = None,
    bounds_check: bool = False,
    is_volatile: bool = False,
    is_nontemporal: bool = False,
    is_invariant: bool = False,
    is_invariant_group: bool = False,
    ordering: Any = "not_atomic",
    syncscope: Any = None,
    loc: Any = None,
    ip: Any = None,
    launch_metadata: _CutlassLaunchMetadata | None = None,
    launch_meta: _CutlassLaunchMetadata | None = None,
    launch: _CutlassLaunchMetadata | None = None,
    launch_config: _CutlassLaunchMetadata | None = None,
) -> _DeferredLoad:
    """Return a deferred block load callable for tensors or ``cutlass.Array``.

    A factory-bound ``payload=`` selector becomes the deferred-call default and
    can be overridden by a later ``payload=`` argument. Factory-bound
    Prims-specific bounds/memory controls select the Prims array path and become
    deferred-call defaults. ``dtype=`` and ``offset=`` can stay on canonical CUB.
    """

def make_store(
    dtype: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    algorithm: Any = "direct",
    *,
    dim: Any = None,
    valid_items: Any = None,
    num_valid_items: Any = None,
    payload: _Payload | Literal["prims"] | None = None,
    offset: Any = None,
    alignment: Any = None,
    bounds_check: bool = False,
    is_volatile: bool = False,
    is_nontemporal: bool = False,
    ordering: Any = "not_atomic",
    syncscope: Any = None,
    loc: Any = None,
    ip: Any = None,
    launch_metadata: _CutlassLaunchMetadata | None = None,
    launch_meta: _CutlassLaunchMetadata | None = None,
    launch: _CutlassLaunchMetadata | None = None,
    launch_config: _CutlassLaunchMetadata | None = None,
) -> _DeferredStore:
    """Return a deferred block store callable for tensors or ``cutlass.Array``.

    A factory-bound ``payload=`` selector becomes the deferred-call default and
    can be overridden by a later ``payload=`` argument. Factory-bound
    Prims-specific bounds/memory controls select the Prims array path and become
    deferred-call defaults. ``dtype=`` and ``offset=`` can stay on canonical CUB.
    """

def make_adjacent_difference(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    difference_op: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block adjacent-difference specialization."""

def make_discontinuity(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    flag_op: object = None,
    *,
    dim: object = None,
    flag_dtype: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block discontinuity specialization."""

def make_exchange(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    block_exchange_type: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block exchange specialization."""

def make_scan(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    initial_value: object = None,
    mode: str = "exclusive",
    scan_op: object = "+",
    prefix_op: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block scan specialization."""

def make_exclusive_scan(
    dtype: object,
    threads_per_block: object = None,
    scan_op: object = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    prefix_op: object = None,
    algorithm: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred exclusive block scan specialization."""

def make_inclusive_scan(
    dtype: object,
    threads_per_block: object = None,
    scan_op: object = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    prefix_op: object = None,
    algorithm: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred inclusive block scan specialization."""

def make_exclusive_sum(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    prefix_op: object = None,
    algorithm: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred exclusive block sum specialization."""

def make_inclusive_sum(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    prefix_op: object = None,
    algorithm: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred inclusive block sum specialization."""

def make_reduce(
    dtype: object,
    threads_per_block: object = None,
    binary_op: object = None,
    items_per_thread: int = 1,
    algorithm: object = None,
    *,
    dim: object = None,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block reduction specialization."""

def make_sum(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    algorithm: object = None,
    *,
    dim: object = None,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block sum specialization."""

def make_shuffle(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int | None = None,
    block_shuffle_type: object = None,
    distance: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block shuffle specialization."""

def make_radix_sort_keys_descending(
    dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    begin_bit: object = 0,
    end_bit: object = None,
    *,
    dim: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred descending block radix-key-sort specialization."""

def make_radix_sort_pairs_descending(
    keys: object = None,
    values: object = None,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    begin_bit: object = 0,
    end_bit: object = None,
    *,
    dim: object = None,
    key_dtype: object = None,
    value_dtype: object = None,
    **kwargs: object,
) -> _DeferredPair:
    """Build a deferred descending block radix-pair-sort specialization."""

def make_histogram(
    item_dtype: object,
    counter_dtype: object,
    threads_per_block: object = None,
    items_per_thread: int = 1,
    *,
    dim: object = None,
    bins: object = None,
    bins_per_thread: int | None = None,
    algorithm: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block histogram specialization."""

def make_run_length(
    item_dtype: object,
    threads_per_block: object = None,
    runs_per_thread: int | None = None,
    decoded_items_per_thread: int = 1,
    *,
    dim: object = None,
    total_decoded_size: object = None,
    decoded_offset_dtype: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred block run-length specialization."""

__all__ = [
    "BlockAdjacentDifferenceType",
    "BlockDiscontinuityType",
    "BlockExchangeType",
    "BlockRunLengthDecode",
    "BlockShuffleType",
    "TempStorage",
    "adjacent_difference",
    "adjacent_difference_subtract_left",
    "adjacent_difference_subtract_right",
    "discontinuity",
    "discontinuity_flag_heads",
    "discontinuity_flag_heads_and_tails",
    "discontinuity_flag_tails",
    "exchange",
    "exchange_blocked_to_striped",
    "exchange_blocked_to_warp_striped",
    "exchange_scatter_to_blocked",
    "exchange_scatter_to_striped",
    "exchange_scatter_to_striped_flagged",
    "exchange_scatter_to_striped_guarded",
    "exchange_striped_to_blocked",
    "exchange_warp_striped_to_blocked",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "make_adjacent_difference",
    "make_discontinuity",
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_histogram",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_radix_sort_keys_descending",
    "make_radix_sort_pairs_descending",
    "make_reduce",
    "make_run_length",
    "make_scan",
    "make_shuffle",
    "make_store",
    "make_sum",
    "radix_sort_keys_descending",
    "radix_sort_pairs_descending",
    "reduce",
    "row_sum",
    "run_length",
    "run_length_decode",
    "scan",
    "shuffle",
    "shuffle_down",
    "shuffle_offset",
    "shuffle_rotate",
    "shuffle_up",
    "store",
    "sum",
]
