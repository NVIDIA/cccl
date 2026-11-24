# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError

from ._base import BlockPrimitive

_SUPPORTED_DTYPES = frozenset(
    {
        datatype.bfloat16,
        datatype.float16,
        datatype.float32,
        datatype.float64,
        datatype.int8,
        datatype.uint8,
        datatype.int16,
        datatype.uint16,
        datatype.int32,
        datatype.uint32,
        datatype.int64,
        datatype.uint64,
    }
)


def _normalize_dtype(dtype):
    if isinstance(dtype, datatype.DType):
        return dtype
    if hasattr(dtype, "dtype"):
        return dtype.dtype
    raise TileTypeError(f"Unsupported dtype {dtype}")


def _merge_methods(
    *,
    primitive: str,
    methods: dict | tuple | None,
    allowed_keys: tuple[str, ...],
    explicit_values: dict[str, object | None],
) -> dict[str, object]:
    """Merge constant method options without changing the caller's mapping.

    Accept a mapping or alternating key/value tuple, reject unknown keys,
    and overlay non-None explicit values. Repeated tuple keys keep the last
    value. The lowering separately rejects runtime arguments that would
    conflict with a method entry; runtime IR values do not belong here.
    """

    if methods is None:
        methods_dict: dict[str, object] = {}
    elif isinstance(methods, tuple):
        if len(methods) % 2 != 0:
            raise TileTypeError(
                f"coop.{primitive} methods tuple must have even length "
                "(key/value pairs)"
            )
        methods_dict = {}
        for i in range(0, len(methods), 2):
            key = methods[i]
            value = methods[i + 1]
            if not isinstance(key, str):
                raise TileTypeError(
                    f"coop.{primitive} methods tuple keys must be strings"
                )
            methods_dict[key] = value
    else:
        if not isinstance(methods, Mapping):
            raise TileTypeError(
                f"coop.{primitive} methods must be a dict or tuple "
                f"of key/value pairs"
            )
        methods_dict = dict(methods)

    unknown = sorted(k for k in methods_dict if k not in allowed_keys)
    if unknown:
        joined = ", ".join(unknown)
        raise TileTypeError(
            f"coop.{primitive} methods contains unsupported keys: {joined}"
        )

    merged = dict(methods_dict)
    for key, value in explicit_values.items():
        if value is not None:
            merged[key] = value
    return merged


class BlockMergeSortKeys(BlockPrimitive):
    """Validate known key-sort controls against a concrete Tile extent.

    A guarded call needs both valid count and default key. Check known counts
    against ``block_threads * items_per_thread`` and defaults against the
    key's numeric kind. Dynamic controls use lowering-owned placeholders.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        descending: bool = False,
        valid_items: int | None = None,
        oob_default: float | None = None,
    ) -> None:
        super().__init__(
            name="block_merge_sort_keys", block_threads=block_threads
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.merge_sort_keys descending must be a bool"
            )
        if valid_items is not None:
            if isinstance(valid_items, bool) or not isinstance(
                valid_items, int
            ):
                raise TileTypeError(
                    "coop.merge_sort_keys valid_items must be an int"
                )
            if valid_items < 0:
                raise TileTypeError(
                    "coop.merge_sort_keys valid_items must be >= 0"
                )
            if oob_default is None:
                raise TileTypeError(
                    "coop.merge_sort_keys oob_default must be provided when "
                    "valid_items is set"
                )
        if oob_default is not None and valid_items is None:
            raise TileTypeError(
                "coop.merge_sort_keys valid_items must be "
                "provided when oob_default is set"
            )
        if oob_default is not None and (
            isinstance(oob_default, bool)
            or not isinstance(oob_default, numbers.Real)
        ):
            raise TileTypeError(
                "coop.merge_sort_keys oob_default must be a numeric constant"
            )
        self.items_per_thread = items_per_thread
        self.dtype = _normalize_dtype(dtype)
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.merge_sort_keys does not support dtype {self.dtype}"
            )
        self.descending = descending
        self.valid_items = valid_items
        self.oob_default = oob_default
        if (
            self.oob_default is not None
            and self.dtype
            in (
                datatype.int8,
                datatype.uint8,
                datatype.int16,
                datatype.uint16,
                datatype.int32,
                datatype.uint32,
                datatype.int64,
                datatype.uint64,
            )
            and not isinstance(self.oob_default, int)
        ):
            raise TileTypeError(
                "coop.merge_sort_keys oob_default must be an "
                "integer for integer dtypes"
            )
        if self.valid_items is not None:
            total_items = self.items_per_thread * self.block_threads
            if self.valid_items > total_items:
                raise TileTypeError(
                    "coop.merge_sort_keys valid_items exceeds tile size"
                )


@dataclass(frozen=True)
class BlockMergeSortKeysOptions:
    descending: bool = False
    valid_items: int | None = None
    oob_default: float | int | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        descending: bool | None,
        valid_items: int | None,
        oob_default: float | None,
        methods: dict | tuple | None,
    ) -> BlockMergeSortKeysOptions:
        merged = _merge_methods(
            primitive="merge_sort_keys",
            methods=methods,
            allowed_keys=("descending", "valid_items", "oob_default"),
            explicit_values={
                "descending": descending,
                "valid_items": valid_items,
                "oob_default": oob_default,
            },
        )
        descending = merged.get("descending")
        valid_items = merged.get("valid_items")
        oob_default = merged.get("oob_default")
        if descending is None:
            descending = False
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.merge_sort_keys descending must be a bool"
            )
        if valid_items is not None:
            if isinstance(valid_items, bool) or not isinstance(
                valid_items, int
            ):
                raise TileTypeError(
                    "coop.merge_sort_keys valid_items must be an int"
                )
            if valid_items < 0:
                raise TileTypeError(
                    "coop.merge_sort_keys valid_items must be >= 0"
                )
            if oob_default is None:
                raise TileTypeError(
                    "coop.merge_sort_keys oob_default must be provided when "
                    "valid_items is set"
                )
        if oob_default is not None and valid_items is None:
            raise TileTypeError(
                "coop.merge_sort_keys valid_items must be "
                "provided when oob_default is set"
            )
        if oob_default is not None and (
            isinstance(oob_default, bool)
            or not isinstance(oob_default, numbers.Real)
        ):
            raise TileTypeError(
                "coop.merge_sort_keys oob_default must be a numeric constant"
            )
        return cls(
            descending=descending,
            valid_items=valid_items,
            oob_default=oob_default,
            methods=merged or None,
        )


class BlockMergeSortPairs(BlockPrimitive):
    """Validate paired sorting controls and both element types.

    The valid count and default key follow the key-only guarded contract.
    The lowering checks equal Tile shapes before constructing this adapter;
    this object checks known controls against the candidate item count.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        key_dtype,
        value_dtype,
        block_threads: int = 128,
        descending: bool = False,
        valid_items: int | None = None,
        oob_default: float | None = None,
    ) -> None:
        super().__init__(
            name="block_merge_sort_pairs", block_threads=block_threads
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.merge_sort_pairs descending must be a bool"
            )
        if valid_items is not None:
            if isinstance(valid_items, bool) or not isinstance(
                valid_items, int
            ):
                raise TileTypeError(
                    "coop.merge_sort_pairs valid_items must be an int"
                )
            if valid_items < 0:
                raise TileTypeError(
                    "coop.merge_sort_pairs valid_items must be >= 0"
                )
            if oob_default is None:
                raise TileTypeError(
                    "coop.merge_sort_pairs oob_default must be provided when "
                    "valid_items is set"
                )
        if oob_default is not None and valid_items is None:
            raise TileTypeError(
                "coop.merge_sort_pairs valid_items must be "
                "provided when oob_default is set"
            )
        if oob_default is not None and (
            isinstance(oob_default, bool)
            or not isinstance(oob_default, numbers.Real)
        ):
            raise TileTypeError(
                "coop.merge_sort_pairs oob_default must be a numeric constant"
            )
        self.items_per_thread = items_per_thread
        self.key_dtype = _normalize_dtype(key_dtype)
        self.value_dtype = _normalize_dtype(value_dtype)
        if self.key_dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.merge_sort_pairs does not support key dtype "
                f"{self.key_dtype}"
            )
        if self.value_dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.merge_sort_pairs does not support value "
                f"dtype {self.value_dtype}"
            )
        self.descending = descending
        self.valid_items = valid_items
        self.oob_default = oob_default
        if (
            self.oob_default is not None
            and self.key_dtype
            in (
                datatype.int8,
                datatype.uint8,
                datatype.int16,
                datatype.uint16,
                datatype.int32,
                datatype.uint32,
                datatype.int64,
                datatype.uint64,
            )
            and not isinstance(self.oob_default, int)
        ):
            raise TileTypeError(
                "coop.merge_sort_pairs oob_default must be an "
                "integer for integer key dtypes"
            )
        if self.valid_items is not None:
            total_items = self.items_per_thread * self.block_threads
            if self.valid_items > total_items:
                raise TileTypeError(
                    "coop.merge_sort_pairs valid_items exceeds tile size"
                )


@dataclass(frozen=True)
class BlockMergeSortPairsOptions:
    descending: bool = False
    valid_items: int | None = None
    oob_default: float | int | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        descending: bool | None,
        valid_items: int | None,
        oob_default: float | None,
        methods: dict | tuple | None,
    ) -> BlockMergeSortPairsOptions:
        merged = _merge_methods(
            primitive="merge_sort_pairs",
            methods=methods,
            allowed_keys=("descending", "valid_items", "oob_default"),
            explicit_values={
                "descending": descending,
                "valid_items": valid_items,
                "oob_default": oob_default,
            },
        )
        descending = merged.get("descending")
        valid_items = merged.get("valid_items")
        oob_default = merged.get("oob_default")
        if descending is None:
            descending = False
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.merge_sort_pairs descending must be a bool"
            )
        if valid_items is not None:
            if isinstance(valid_items, bool) or not isinstance(
                valid_items, int
            ):
                raise TileTypeError(
                    "coop.merge_sort_pairs valid_items must be an int"
                )
            if valid_items < 0:
                raise TileTypeError(
                    "coop.merge_sort_pairs valid_items must be >= 0"
                )
            if oob_default is None:
                raise TileTypeError(
                    "coop.merge_sort_pairs oob_default must be provided when "
                    "valid_items is set"
                )
        if oob_default is not None and valid_items is None:
            raise TileTypeError(
                "coop.merge_sort_pairs valid_items must be "
                "provided when oob_default is set"
            )
        if oob_default is not None and (
            isinstance(oob_default, bool)
            or not isinstance(oob_default, numbers.Real)
        ):
            raise TileTypeError(
                "coop.merge_sort_pairs oob_default must be a numeric constant"
            )
        return cls(
            descending=descending,
            valid_items=valid_items,
            oob_default=oob_default,
            methods=merged or None,
        )
