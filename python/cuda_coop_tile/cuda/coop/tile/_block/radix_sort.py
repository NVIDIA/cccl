# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

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

_INNER_SCAN_TO_TOKEN = {
    None: "warp_scans",
    "raking": "raking",
    "raking_memoize": "raking_memoize",
    "warp_scans": "warp_scans",
}

_SMEM_CONFIG_TO_TOKEN = {
    None: "four_byte",
    "default": "four_byte",
    "four_byte": "four_byte",
    "4": "four_byte",
    4: "four_byte",
    "eight_byte": "eight_byte",
    "8": "eight_byte",
    8: "eight_byte",
}

_MAX_RADIX_SORT_BITS = 8
_RADIX_SORT_BITS_LIMIT_ERROR = (
    "radix_bits must be in the range [1, 8]; cuda.coop.tile's native provider "
    "currently supports at most 8 radix bits per pass, so use <= 8 or leave "
    "radix_bits unset"
)


def _normalize_inner_scan(inner_scan_algorithm: str | None) -> str | None:
    if inner_scan_algorithm not in _INNER_SCAN_TO_TOKEN:
        raise TileTypeError(
            "coop.radix_sort inner_scan_algorithm must be one of: "
            "raking, raking_memoize, warp_scans"
        )
    return inner_scan_algorithm


def _normalize_smem_config(smem_config: str | int | None) -> str | int | None:
    if smem_config not in _SMEM_CONFIG_TO_TOKEN:
        raise TileTypeError(
            "coop.radix_sort smem_config must be one of: default, "
            "four_byte, eight_byte, 4, 8"
        )
    return smem_config


def _validate_radix_bits(radix_bits: int, *, primitive: str) -> None:
    if radix_bits <= 0 or radix_bits > _MAX_RADIX_SORT_BITS:
        raise TileTypeError(f"coop.{primitive} {_RADIX_SORT_BITS_LIMIT_ERROR}")


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


class BlockRadixSortKeys(BlockPrimitive):
    """Validate and normalize a concrete key-sort specialization.

    Resolve omitted bit bounds from the key width and convert tuning aliases
    to the tokens consumed by lowering. Runtime bounds use validation-only
    stand-ins here and are emitted as scalar operands later.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        descending: bool = False,
        begin_bit: int | None = None,
        end_bit: int | None = None,
        radix_bits: int | None = None,
        memoize_outer_scan: bool | None = None,
        inner_scan_algorithm: str | None = None,
        smem_config: str | int | None = None,
    ) -> None:
        super().__init__(
            name="block_radix_sort_keys", block_threads=block_threads
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if items_per_thread * block_threads > 65535:
            raise TileTypeError(
                "coop.radix_sort_keys tile size must not exceed 65535"
            )
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.radix_sort_keys descending must be a bool"
            )
        if begin_bit is not None and (
            isinstance(begin_bit, bool) or not isinstance(begin_bit, int)
        ):
            raise TileTypeError("coop.radix_sort_keys begin_bit must be an int")
        if end_bit is not None and (
            isinstance(end_bit, bool) or not isinstance(end_bit, int)
        ):
            raise TileTypeError("coop.radix_sort_keys end_bit must be an int")
        if radix_bits is not None and (
            isinstance(radix_bits, bool) or not isinstance(radix_bits, int)
        ):
            raise TileTypeError(
                "coop.radix_sort_keys radix_bits must be an int"
            )
        if memoize_outer_scan is not None and not isinstance(
            memoize_outer_scan, bool
        ):
            raise TileTypeError(
                "coop.radix_sort_keys memoize_outer_scan must be a bool"
            )
        self.items_per_thread = items_per_thread
        self.dtype = _normalize_dtype(dtype)
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.radix_sort_keys does not support dtype {self.dtype}"
            )
        self.descending = descending
        self.key_bits = self.dtype.bitwidth
        self.begin_bit = 0 if begin_bit is None else begin_bit
        self.end_bit = self.key_bits if end_bit is None else end_bit
        self.radix_bits = 4 if radix_bits is None else radix_bits
        self.memoize_outer_scan = (
            True if memoize_outer_scan is None else memoize_outer_scan
        )
        self.inner_scan_algorithm = _normalize_inner_scan(inner_scan_algorithm)
        self.inner_scan_token = _INNER_SCAN_TO_TOKEN[self.inner_scan_algorithm]
        self.smem_config = _normalize_smem_config(smem_config)
        self.smem_config_token = _SMEM_CONFIG_TO_TOKEN[self.smem_config]
        if self.begin_bit < 0:
            raise TileTypeError("coop.radix_sort_keys begin_bit must be >= 0")
        if self.end_bit < self.begin_bit:
            raise TileTypeError(
                "coop.radix_sort_keys end_bit must be >= begin_bit"
            )
        if self.end_bit > self.key_bits:
            raise TileTypeError(
                "coop.radix_sort_keys end_bit exceeds key bit width"
            )
        _validate_radix_bits(self.radix_bits, primitive="radix_sort_keys")


class BlockRadixSortPairs(BlockPrimitive):
    """Validate paired Radix Sort while retaining separate element types.

    Keys determine the bit width and ordering controls. Values retain their
    own supported dtype and follow the key permutation. Both payload extents
    are checked by the lowering before this adapter is constructed.
    """

    def __init__(
        self,
        *,
        items_per_thread: int,
        key_dtype,
        value_dtype,
        block_threads: int = 128,
        descending: bool = False,
        begin_bit: int | None = None,
        end_bit: int | None = None,
        radix_bits: int | None = None,
        memoize_outer_scan: bool | None = None,
        inner_scan_algorithm: str | None = None,
        smem_config: str | int | None = None,
    ) -> None:
        super().__init__(
            name="block_radix_sort_pairs", block_threads=block_threads
        )
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if items_per_thread * block_threads > 65535:
            raise TileTypeError(
                "coop.radix_sort_pairs tile size must not exceed 65535"
            )
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.radix_sort_pairs descending must be a bool"
            )
        if begin_bit is not None and (
            isinstance(begin_bit, bool) or not isinstance(begin_bit, int)
        ):
            raise TileTypeError(
                "coop.radix_sort_pairs begin_bit must be an int"
            )
        if end_bit is not None and (
            isinstance(end_bit, bool) or not isinstance(end_bit, int)
        ):
            raise TileTypeError("coop.radix_sort_pairs end_bit must be an int")
        if radix_bits is not None and (
            isinstance(radix_bits, bool) or not isinstance(radix_bits, int)
        ):
            raise TileTypeError(
                "coop.radix_sort_pairs radix_bits must be an int"
            )
        if memoize_outer_scan is not None and not isinstance(
            memoize_outer_scan, bool
        ):
            raise TileTypeError(
                "coop.radix_sort_pairs memoize_outer_scan must be a bool"
            )
        self.items_per_thread = items_per_thread
        self.key_dtype = _normalize_dtype(key_dtype)
        self.value_dtype = _normalize_dtype(value_dtype)
        if self.key_dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.radix_sort_pairs does not support key dtype "
                f"{self.key_dtype}"
            )
        if self.value_dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.radix_sort_pairs does not support value "
                f"dtype {self.value_dtype}"
            )
        self.descending = descending
        self.key_bits = self.key_dtype.bitwidth
        self.begin_bit = 0 if begin_bit is None else begin_bit
        self.end_bit = self.key_bits if end_bit is None else end_bit
        self.radix_bits = 4 if radix_bits is None else radix_bits
        self.memoize_outer_scan = (
            True if memoize_outer_scan is None else memoize_outer_scan
        )
        self.inner_scan_algorithm = _normalize_inner_scan(inner_scan_algorithm)
        self.inner_scan_token = _INNER_SCAN_TO_TOKEN[self.inner_scan_algorithm]
        self.smem_config = _normalize_smem_config(smem_config)
        self.smem_config_token = _SMEM_CONFIG_TO_TOKEN[self.smem_config]
        if self.begin_bit < 0:
            raise TileTypeError("coop.radix_sort_pairs begin_bit must be >= 0")
        if self.end_bit < self.begin_bit:
            raise TileTypeError(
                "coop.radix_sort_pairs end_bit must be >= begin_bit"
            )
        if self.end_bit > self.key_bits:
            raise TileTypeError(
                "coop.radix_sort_pairs end_bit exceeds key bit width"
            )
        _validate_radix_bits(self.radix_bits, primitive="radix_sort_pairs")


@dataclass(frozen=True)
class BlockRadixSortKeysOptions:
    descending: bool = False
    begin_bit: int | None = None
    end_bit: int | None = None
    radix_bits: int | None = None
    memoize_outer_scan: bool | None = None
    inner_scan_algorithm: str | None = None
    smem_config: str | int | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        descending: bool | None,
        begin_bit: int | None,
        end_bit: int | None,
        radix_bits: int | None,
        memoize_outer_scan: bool | None,
        inner_scan_algorithm: str | None,
        smem_config: str | int | None,
        methods: dict | tuple | None,
    ) -> BlockRadixSortKeysOptions:
        merged = _merge_methods(
            primitive="radix_sort_keys",
            methods=methods,
            allowed_keys=(
                "descending",
                "begin_bit",
                "end_bit",
                "radix_bits",
                "memoize_outer_scan",
                "inner_scan_algorithm",
                "smem_config",
            ),
            explicit_values={
                "descending": descending,
                "begin_bit": begin_bit,
                "end_bit": end_bit,
                "radix_bits": radix_bits,
                "memoize_outer_scan": memoize_outer_scan,
                "inner_scan_algorithm": inner_scan_algorithm,
                "smem_config": smem_config,
            },
        )
        descending = merged.get("descending")
        begin_bit = merged.get("begin_bit")
        end_bit = merged.get("end_bit")
        radix_bits = merged.get("radix_bits")
        memoize_outer_scan = merged.get("memoize_outer_scan")
        inner_scan_algorithm = merged.get("inner_scan_algorithm")
        smem_config = merged.get("smem_config")
        if descending is None:
            descending = False
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.radix_sort_keys descending must be a bool"
            )
        if begin_bit is not None and (
            isinstance(begin_bit, bool) or not isinstance(begin_bit, int)
        ):
            raise TileTypeError("coop.radix_sort_keys begin_bit must be an int")
        if end_bit is not None and (
            isinstance(end_bit, bool) or not isinstance(end_bit, int)
        ):
            raise TileTypeError("coop.radix_sort_keys end_bit must be an int")
        if radix_bits is not None and (
            isinstance(radix_bits, bool) or not isinstance(radix_bits, int)
        ):
            raise TileTypeError(
                "coop.radix_sort_keys radix_bits must be an int"
            )
        if radix_bits is not None:
            _validate_radix_bits(radix_bits, primitive="radix_sort_keys")
        if memoize_outer_scan is not None and not isinstance(
            memoize_outer_scan, bool
        ):
            raise TileTypeError(
                "coop.radix_sort_keys memoize_outer_scan must be a bool"
            )
        if (
            inner_scan_algorithm is not None
            and inner_scan_algorithm not in _INNER_SCAN_TO_TOKEN
        ):
            raise TileTypeError(
                "coop.radix_sort_keys inner_scan_algorithm must be one of: "
                "raking, raking_memoize, warp_scans"
            )
        if smem_config is not None and smem_config not in _SMEM_CONFIG_TO_TOKEN:
            raise TileTypeError(
                "coop.radix_sort_keys smem_config must be one of: "
                "default, four_byte, eight_byte, 4, 8"
            )
        return cls(
            descending=descending,
            begin_bit=begin_bit,
            end_bit=end_bit,
            radix_bits=radix_bits,
            memoize_outer_scan=memoize_outer_scan,
            inner_scan_algorithm=inner_scan_algorithm,
            smem_config=smem_config,
            methods=merged or None,
        )


@dataclass(frozen=True)
class BlockRadixSortPairsOptions:
    descending: bool = False
    begin_bit: int | None = None
    end_bit: int | None = None
    radix_bits: int | None = None
    memoize_outer_scan: bool | None = None
    inner_scan_algorithm: str | None = None
    smem_config: str | int | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        descending: bool | None,
        begin_bit: int | None,
        end_bit: int | None,
        radix_bits: int | None,
        memoize_outer_scan: bool | None,
        inner_scan_algorithm: str | None,
        smem_config: str | int | None,
        methods: dict | tuple | None,
    ) -> BlockRadixSortPairsOptions:
        merged = _merge_methods(
            primitive="radix_sort_pairs",
            methods=methods,
            allowed_keys=(
                "descending",
                "begin_bit",
                "end_bit",
                "radix_bits",
                "memoize_outer_scan",
                "inner_scan_algorithm",
                "smem_config",
            ),
            explicit_values={
                "descending": descending,
                "begin_bit": begin_bit,
                "end_bit": end_bit,
                "radix_bits": radix_bits,
                "memoize_outer_scan": memoize_outer_scan,
                "inner_scan_algorithm": inner_scan_algorithm,
                "smem_config": smem_config,
            },
        )
        descending = merged.get("descending")
        begin_bit = merged.get("begin_bit")
        end_bit = merged.get("end_bit")
        radix_bits = merged.get("radix_bits")
        memoize_outer_scan = merged.get("memoize_outer_scan")
        inner_scan_algorithm = merged.get("inner_scan_algorithm")
        smem_config = merged.get("smem_config")
        if descending is None:
            descending = False
        if not isinstance(descending, bool):
            raise TileTypeError(
                "coop.radix_sort_pairs descending must be a bool"
            )
        if begin_bit is not None and (
            isinstance(begin_bit, bool) or not isinstance(begin_bit, int)
        ):
            raise TileTypeError(
                "coop.radix_sort_pairs begin_bit must be an int"
            )
        if end_bit is not None and (
            isinstance(end_bit, bool) or not isinstance(end_bit, int)
        ):
            raise TileTypeError("coop.radix_sort_pairs end_bit must be an int")
        if radix_bits is not None and (
            isinstance(radix_bits, bool) or not isinstance(radix_bits, int)
        ):
            raise TileTypeError(
                "coop.radix_sort_pairs radix_bits must be an int"
            )
        if radix_bits is not None:
            _validate_radix_bits(radix_bits, primitive="radix_sort_pairs")
        if memoize_outer_scan is not None and not isinstance(
            memoize_outer_scan, bool
        ):
            raise TileTypeError(
                "coop.radix_sort_pairs memoize_outer_scan must be a bool"
            )
        if (
            inner_scan_algorithm is not None
            and inner_scan_algorithm not in _INNER_SCAN_TO_TOKEN
        ):
            raise TileTypeError(
                "coop.radix_sort_pairs inner_scan_algorithm must be one of: "
                "raking, raking_memoize, warp_scans"
            )
        if smem_config is not None and smem_config not in _SMEM_CONFIG_TO_TOKEN:
            raise TileTypeError(
                "coop.radix_sort_pairs smem_config must be one of: "
                "default, four_byte, eight_byte, 4, 8"
            )
        return cls(
            descending=descending,
            begin_bit=begin_bit,
            end_bit=end_bit,
            radix_bits=radix_bits,
            memoize_outer_scan=memoize_outer_scan,
            inner_scan_algorithm=inner_scan_algorithm,
            smem_config=smem_config,
            methods=merged or None,
        )
