# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Temporary-storage identity and slice planning for CUTLASS collectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_DEFAULT_SCOPE = "cuda.coop.cutlass"
_TEMP_STORAGE_SHARING_VALUES = frozenset(("shared", "exclusive"))


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    remainder = value % alignment
    return value if remainder == 0 else (value + alignment - remainder)


def _topk_cub_temp_storage_requirement(
    *,
    block_threads: int,
    items_per_thread: int,
    key_bytes: int,
    value_bytes: int = 0,
) -> tuple[int, int]:
    tile_items = max(1, block_threads) * max(1, items_per_thread)
    item_payload_bytes = max(1, key_bytes)
    if value_bytes > 0:
        item_payload_bytes += max(1, value_bytes)
    exchange_bytes = 8 + tile_items * item_payload_bytes
    staged_output_bytes = tile_items * max(1, key_bytes)
    if value_bytes > 0:
        staged_output_bytes += tile_items * max(1, value_bytes)
    # CUB TopK uses a 256-bin histogram plus BlockScan scratch before the
    # exchange phase. This is intentionally conservative because CuTe must
    # allocate before NVRTC can tell us the exact C++ TempStorage size.
    pass_bytes = 4096 + max(1, block_threads) * 16
    return _align_up(max(exchange_bytes, pass_bytes) + staged_output_bytes, 16), 16


@dataclass(frozen=True)
class TempStorageSlice:
    byte_offset_in_bytes: int
    size_in_bytes: int
    alignment: int


def _merge_slice_requirements(
    existing: TempStorageSlice,
    *,
    required_size_in_bytes: int,
    required_alignment: int,
) -> TempStorageSlice:
    next_size = max(existing.size_in_bytes, required_size_in_bytes)
    next_alignment = max(existing.alignment, required_alignment)
    return TempStorageSlice(
        byte_offset_in_bytes=existing.byte_offset_in_bytes,
        size_in_bytes=next_size,
        alignment=next_alignment,
    )


def _plan_next_slice(
    planned_size_in_bytes: int,
    *,
    required_size_in_bytes: int,
    required_alignment: int,
) -> TempStorageSlice:
    offset = _align_up(planned_size_in_bytes, required_alignment)
    return TempStorageSlice(
        byte_offset_in_bytes=offset,
        size_in_bytes=required_size_in_bytes,
        alignment=required_alignment,
    )


def _plan_primitive_slice(
    primitive_slices: dict[str, TempStorageSlice],
    primitive_name: str,
    *,
    sharing: str,
    planned_size_in_bytes: int,
    required_size_in_bytes: int,
    required_alignment: int,
) -> tuple[TempStorageSlice, int]:
    if sharing == "exclusive":
        planned = _plan_next_slice(
            planned_size_in_bytes,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
        )
        next_size = (
            planned_size_in_bytes
            if required_size_in_bytes == 0
            else planned.byte_offset_in_bytes + planned.size_in_bytes
        )
        return planned, next_size

    current = primitive_slices.get(primitive_name)
    if current is not None:
        merged = _merge_slice_requirements(
            current,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
        )
        return merged, max(
            planned_size_in_bytes,
            merged.size_in_bytes,
        )

    planned = TempStorageSlice(
        byte_offset_in_bytes=0,
        size_in_bytes=required_size_in_bytes,
        alignment=required_alignment,
    )
    return planned, max(planned_size_in_bytes, planned.size_in_bytes)


@dataclass(frozen=True)
class TempStorageUse:
    primitive_name: str
    required_size_in_bytes: int
    required_alignment: int
    byte_offset_in_bytes: int


class TempStorageBase:
    """Explicit shared-memory scratch planner for CuTe collectives."""

    scope = _DEFAULT_SCOPE

    def __init__(
        self,
        size_in_bytes: int | None = None,
        alignment: int | None = None,
        auto_sync: bool | None = None,
        sharing: Literal["shared", "exclusive"] = "shared",
    ):
        deferred = size_in_bytes is None
        if size_in_bytes is not None:
            if not isinstance(size_in_bytes, int) or isinstance(size_in_bytes, bool):
                raise TypeError("TempStorage size_in_bytes must be an integer or None.")
            if size_in_bytes <= 0:
                raise ValueError(
                    "TempStorage size_in_bytes must be a positive integer."
                )

        if alignment is not None:
            if not isinstance(alignment, int) or isinstance(alignment, bool):
                raise TypeError("TempStorage alignment must be an integer or None.")
            if alignment <= 0:
                raise ValueError("TempStorage alignment must be a positive integer.")
            if alignment & (alignment - 1):
                raise ValueError("TempStorage alignment must be a power of 2.")

        if not isinstance(sharing, str):
            raise TypeError(
                "TempStorage sharing must be a string: 'shared' or 'exclusive'."
            )
        sharing_value = sharing.strip().lower()
        if sharing_value not in _TEMP_STORAGE_SHARING_VALUES:
            raise ValueError("TempStorage sharing must be 'shared' or 'exclusive'.")

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise TypeError("TempStorage auto_sync must be None/True/False.")
        if sharing_value == "exclusive" and auto_sync is True:
            raise ValueError(
                "TempStorage with sharing='exclusive' does not support auto_sync=True."
            )
        self.size_in_bytes = size_in_bytes
        self.alignment = alignment
        self.sharing = sharing_value
        self.auto_sync = (
            False
            if sharing_value == "exclusive"
            else (True if auto_sync is None else auto_sync)
        )
        self._deferred = deferred
        self._required_size_in_bytes = 0
        self._required_alignment = 1
        self._uses: list[TempStorageUse] = []
        self._primitive_slices: dict[str, TempStorageSlice] = {}
        self._planned_size_in_bytes = 0

    def record_use(
        self,
        primitive_name: str,
        *,
        required_size_in_bytes: int = 0,
        required_alignment: int = 1,
    ) -> TempStorageUse:
        if not isinstance(required_size_in_bytes, int) or isinstance(
            required_size_in_bytes, bool
        ):
            raise TypeError("required_size_in_bytes must be an integer")
        if required_size_in_bytes < 0:
            raise ValueError("required_size_in_bytes must be >= 0")
        if not isinstance(required_alignment, int) or isinstance(
            required_alignment, bool
        ):
            raise TypeError("required_alignment must be an integer")
        if required_alignment <= 0:
            raise ValueError("required_alignment must be > 0")
        if required_alignment & (required_alignment - 1):
            raise ValueError("required_alignment must be a power of two")

        if (
            self.size_in_bytes is not None
            and self.size_in_bytes < required_size_in_bytes
        ):
            raise ValueError(
                "TempStorage size_in_bytes is smaller than required by primitive "
                f"use ({self.size_in_bytes} < {required_size_in_bytes})"
            )
        if self.alignment is not None and self.alignment < required_alignment:
            raise ValueError(
                "TempStorage alignment is smaller than required by primitive use "
                f"({self.alignment} < {required_alignment})"
            )

        slice_plan, next_planned_size = _plan_primitive_slice(
            self._primitive_slices,
            primitive_name,
            sharing=self.sharing,
            planned_size_in_bytes=self._planned_size_in_bytes,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
        )
        if self.size_in_bytes is not None and self.size_in_bytes < next_planned_size:
            raise ValueError(
                "TempStorage size_in_bytes is smaller than the cumulative planned "
                "primitive uses "
                f"({self.size_in_bytes} < {next_planned_size})"
            )
        self._primitive_slices[primitive_name] = slice_plan
        self._planned_size_in_bytes = next_planned_size

        self._required_size_in_bytes = max(
            self._required_size_in_bytes, required_size_in_bytes
        )
        self._required_alignment = max(self._required_alignment, required_alignment)
        use = TempStorageUse(
            primitive_name=primitive_name,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
            byte_offset_in_bytes=slice_plan.byte_offset_in_bytes,
        )
        self._uses.append(use)
        return use

    @property
    def uses(self) -> tuple[TempStorageUse, ...]:
        return tuple(self._uses)

    @property
    def is_deferred(self) -> bool:
        """Whether this storage is finalized after whole-kernel tracing."""

        return self._deferred

    @property
    def required_size_in_bytes(self) -> int:
        return self._planned_size_in_bytes

    @property
    def capacity_size_in_bytes(self) -> int | None:
        return self.size_in_bytes

    @property
    def required_alignment(self) -> int:
        return (
            self.alignment if self.alignment is not None else self._required_alignment
        )

    def reset_uses(self) -> None:
        self._required_size_in_bytes = 0
        self._required_alignment = 1
        self._uses.clear()
        self._primitive_slices.clear()
        self._planned_size_in_bytes = 0

    def _snapshot_uses(self):
        return (
            self._required_size_in_bytes,
            self._required_alignment,
            list(self._uses),
            dict(self._primitive_slices),
            self._planned_size_in_bytes,
        )

    def _restore_uses(self, snapshot) -> None:
        (
            self._required_size_in_bytes,
            self._required_alignment,
            uses,
            primitive_slices,
            self._planned_size_in_bytes,
        ) = snapshot
        self._uses = list(uses)
        self._primitive_slices = dict(primitive_slices)

    def slice_for_primitive(self, primitive_name: str) -> TempStorageSlice | None:
        return self._primitive_slices.get(primitive_name)

    def slice_for_latest_use(self, primitive_name: str) -> TempStorageSlice | None:
        if not self._uses or self._uses[-1].primitive_name != primitive_name:
            return None
        use = self._uses[-1]
        return TempStorageSlice(
            byte_offset_in_bytes=use.byte_offset_in_bytes,
            size_in_bytes=use.required_size_in_bytes,
            alignment=use.required_alignment,
        )


class TempStorage(TempStorageBase):
    """Identity-scoped scratch planner for CUTLASS block collectives."""

    scope = "cuda.coop.cutlass"

    def sync(self) -> None:
        """Synchronize the threads that may reuse this block scratch."""

        from ._thread_group import this_block

        this_block().sync()


__all__ = ["TempStorage"]
