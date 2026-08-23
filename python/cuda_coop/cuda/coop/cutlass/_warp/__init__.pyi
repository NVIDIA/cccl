# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Literal, Protocol

from .. import Payload as _Payload
from .. import TempStorage as TempStorage
from .. import ThreadData as _ThreadData
from .._dsl.warp import WarpExchangeType as WarpExchangeType
from .._dsl.warp import exclusive_scan as exclusive_scan
from .._dsl.warp import exclusive_sum as exclusive_sum
from .._dsl.warp import inclusive_scan as inclusive_scan
from .._dsl.warp import inclusive_sum as inclusive_sum
from .._dsl.warp import max as max
from .._dsl.warp import min as min
from .._dsl.warp import reduce as reduce
from .._dsl.warp import scan as scan
from .._dsl.warp import sum as sum

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
    """Deferred scoped physical-warp load callable."""

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
        threads_in_warp: int = 32,
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
    ) -> object:
        """Load from ``source`` using captured defaults."""

class _DeferredStore(Protocol):
    """Deferred scoped physical-warp store callable."""

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
        threads_in_warp: int = 32,
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
    threads_in_warp: int = 32,
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
) -> Any:
    """Load a warp tile into per-thread data.

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
    threads_in_warp: int = 32,
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
) -> None:
    """Store per-thread data into a warp tile.

    CUTLASS tensor destinations use the default tensor path. Public
    ``cutlass.Array`` destinations dispatch to the Prims array path. Pass
    ``payload=Payload.PRIMS`` to store through that path. Prims-specific
    bounds/memory controls such as ``bounds_check=`` also select it. ``dtype=``
    and ``ThreadData.dtype`` alone remain on the tensor route.
    """

def make_load(
    dtype: Any = None,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Any = "direct",
    num_valid_items: Any = None,
    oob_default: Any = None,
    *,
    payload: _Payload | Literal["prims"] | None = None,
    valid_items: Any = None,
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
) -> _DeferredLoad:
    """Return a deferred warp load callable for tensors or ``cutlass.Array``.

    A factory-bound ``payload=`` selector becomes the deferred-call default and
    can be overridden by a later ``payload=`` argument. Factory-bound
    Prims-specific bounds/memory controls select the Prims array path and become
    deferred-call defaults. ``dtype=`` and ``offset=`` can stay on canonical CUB.
    """

def make_store(
    dtype: Any = None,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Any = "direct",
    num_valid_items: Any = None,
    *,
    payload: _Payload | Literal["prims"] | None = None,
    valid_items: Any = None,
    offset: Any = None,
    alignment: Any = None,
    bounds_check: bool = False,
    is_volatile: bool = False,
    is_nontemporal: bool = False,
    ordering: Any = "not_atomic",
    syncscope: Any = None,
    loc: Any = None,
    ip: Any = None,
) -> _DeferredStore:
    """Return a deferred warp store callable for tensors or ``cutlass.Array``.

    A factory-bound ``payload=`` selector becomes the deferred-call default and
    can be overridden by a later ``payload=`` argument. Factory-bound
    Prims-specific bounds/memory controls select the Prims array path and become
    deferred-call defaults. ``dtype=`` and ``offset=`` can stay on canonical CUB.
    """

def make_exclusive_scan(
    dtype: object,
    scan_op: object = None,
    initial_value: object = None,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred exclusive physical-warp scan specialization."""

def make_inclusive_scan(
    dtype: object,
    scan_op: object = None,
    initial_value: object = None,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred inclusive physical-warp scan specialization."""

def make_exclusive_sum(
    dtype: object,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred exclusive physical-warp sum specialization."""

def make_inclusive_sum(
    dtype: object,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred inclusive physical-warp sum specialization."""

def make_reduce(
    dtype: object,
    binary_op: object = None,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred physical-warp reduction specialization."""

def make_sum(
    dtype: object,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred physical-warp sum specialization."""

def make_max(
    dtype: object,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred physical-warp maximum specialization."""

def make_min(
    dtype: object,
    threads_in_warp: int = 32,
    valid_items: object = None,
    **kwargs: object,
) -> _DeferredValue:
    """Build a deferred physical-warp minimum specialization."""

__all__ = [
    "TempStorage",
    "WarpExchangeType",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_max",
    "make_min",
    "make_reduce",
    "make_store",
    "make_sum",
    "max",
    "min",
    "reduce",
    "scan",
    "store",
    "sum",
]
