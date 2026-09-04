# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS register payload shared by cooperative primitives."""

from __future__ import annotations

from typing import Any, Iterator

_UNINITIALIZED = object()


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

    def __init__(self, items_per_thread: int, dtype: Any = None) -> None:
        if (
            not isinstance(items_per_thread, int)
            or isinstance(items_per_thread, bool)
            or items_per_thread < 1
        ):
            raise ValueError("items_per_thread must be a positive integer")
        self._items_per_thread = items_per_thread
        self.dtype = dtype
        self._items: list[Any] = [_UNINITIALIZED] * items_per_thread

    @property
    def items_per_thread(self) -> int:
        """Number of values in this fixed-size payload."""

        return self._items_per_thread

    @property
    def dtype(self) -> Any:
        return self._dtype

    @dtype.setter
    def dtype(self, value: Any) -> None:
        if value is None:
            self._dtype = None
            return
        from ._provider import _canonical_type

        try:
            self._dtype = _canonical_type(value, feature="ThreadData")
        except NotImplementedError as error:
            raise TypeError(str(error)) from error

    def _validate_index(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("ThreadData index must be an integer")
        if not 0 <= index < self.items_per_thread:
            raise IndexError("ThreadData index is out of range")
        return index

    def __len__(self) -> int:
        return self.items_per_thread

    def __getitem__(self, index: int) -> Any:
        index = self._validate_index(index)
        item = self._items[index]
        if item is _UNINITIALIZED:
            raise RuntimeError("ThreadData item was read before it was initialized")
        return item

    def __setitem__(self, index: int, value: Any) -> None:
        index = self._validate_index(index)
        self._items[index] = value

    def __iter__(self) -> Iterator[Any]:
        for index in range(self.items_per_thread):
            yield self[index]

    def __repr__(self) -> str:
        return (
            f"ThreadData(items_per_thread={self.items_per_thread}, "
            f"dtype={self.dtype!r})"
        )


__all__ = ["ThreadData"]
