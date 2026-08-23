# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Literal, Protocol

from .. import Payload as _Payload
from .. import TempStorage as TempStorage
from .. import ThreadData as _ThreadData
from .._dsl._launch import CutlassLaunchMetadata as _CutlassLaunchMetadata
from .._dsl.block import radix_sort_keys_descending as radix_sort_keys_descending
from .._dsl.block import radix_sort_pairs_descending as radix_sort_pairs_descending
from .._dsl.block import reduce as reduce
from .._dsl.block import row_sum as row_sum
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

__all__ = [
    "TempStorage",
    "load",
    "make_load",
    "make_radix_sort_keys_descending",
    "make_radix_sort_pairs_descending",
    "make_reduce",
    "make_store",
    "make_sum",
    "radix_sort_keys_descending",
    "radix_sort_pairs_descending",
    "reduce",
    "row_sum",
    "store",
    "sum",
]
