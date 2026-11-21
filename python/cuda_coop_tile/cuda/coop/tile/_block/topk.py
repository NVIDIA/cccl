# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass

from cuda.tile._exception import TileTypeError

from ._base import BlockPrimitive
from .radix_sort import _SUPPORTED_DTYPES, _merge_methods, _normalize_dtype


def _validate_int_range(
    value: int | None,
    *,
    name: str,
    primitive: str,
    lower: int | None = None,
    upper: int | None = None,
) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TileTypeError(f"coop.{primitive} {name} must be an int")
    if lower is not None and value < lower:
        raise TileTypeError(f"coop.{primitive} {name} must be >= {lower}")
    if upper is not None and value > upper:
        raise TileTypeError(f"coop.{primitive} {name} must be <= {upper}")


class _BlockTopKBase(BlockPrimitive):
    def _init_common(
        self,
        *,
        primitive: str,
        items_per_thread: int,
        block_threads: int,
        descending: bool,
        k: int | None,
        num_valid: int | None,
        key_dtype,
        begin_bit: int | None,
        end_bit: int | None,
    ) -> None:
        """Record and check the portions of TopK controls known at lowering.

        Validate the key dtype, nonnegative k, valid prefix, and bit interval
        against the candidate tile. ``None`` for k or num_valid can stand for
        a dynamic operand, so this check cannot establish its runtime range.
        Bounds omitted from this record default to the full key width.
        """

        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        if not isinstance(descending, bool):
            raise TileTypeError(f"coop.{primitive} descending must be a bool")
        if k is not None and (not isinstance(k, int) or isinstance(k, bool)):
            raise TileTypeError(f"coop.{primitive} k must be an int")

        self.items_per_thread = items_per_thread
        self.descending = descending
        self.k = k
        self.num_valid = num_valid
        self.key_dtype = _normalize_dtype(key_dtype)
        if self.key_dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.{primitive} does not support key dtype {self.key_dtype}"
            )
        self.key_bits = self.key_dtype.bitwidth
        _validate_int_range(
            k,
            name="k",
            primitive=primitive,
            lower=0,
            upper=block_threads * items_per_thread,
        )
        _validate_int_range(
            num_valid,
            name="valid_items",
            primitive=primitive,
            lower=0,
            upper=block_threads * items_per_thread,
        )
        _validate_int_range(
            begin_bit, name="begin_bit", primitive=primitive, lower=0
        )
        _validate_int_range(
            end_bit, name="end_bit", primitive=primitive, lower=0
        )
        self.begin_bit = 0 if begin_bit is None else begin_bit
        self.end_bit = self.key_bits if end_bit is None else end_bit
        if self.begin_bit >= self.key_bits:
            raise TileTypeError(
                f"coop.{primitive} begin_bit must be < key bit width"
            )
        if self.end_bit <= self.begin_bit:
            raise TileTypeError(f"coop.{primitive} end_bit must be > begin_bit")
        if self.end_bit > self.key_bits:
            raise TileTypeError(
                f"coop.{primitive} end_bit exceeds key bit width"
            )


class BlockTopKKeys(_BlockTopKBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        descending: bool = False,
        k: int | None,
        num_valid: int | None = None,
        begin_bit: int | None = None,
        end_bit: int | None = None,
    ) -> None:
        super().__init__(name="block_topk_keys", block_threads=block_threads)
        self._init_common(
            primitive="topk_keys",
            items_per_thread=items_per_thread,
            block_threads=block_threads,
            descending=descending,
            k=k,
            num_valid=num_valid,
            key_dtype=dtype,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        self.dtype = self.key_dtype


class BlockTopKPairs(_BlockTopKBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        key_dtype,
        value_dtype,
        block_threads: int = 128,
        descending: bool = False,
        k: int | None,
        num_valid: int | None = None,
        begin_bit: int | None = None,
        end_bit: int | None = None,
    ) -> None:
        super().__init__(name="block_topk_pairs", block_threads=block_threads)
        self._init_common(
            primitive="topk_pairs",
            items_per_thread=items_per_thread,
            block_threads=block_threads,
            descending=descending,
            k=k,
            num_valid=num_valid,
            key_dtype=key_dtype,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        self.value_dtype = _normalize_dtype(value_dtype)
        if self.value_dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.topk_pairs does not support value dtype "
                f"{self.value_dtype}"
            )


@dataclass(frozen=True)
class BlockTopKKeysOptions:
    """Hold known TopK controls before dtype and extent validation.

    Only bit bounds may come from ``methods``. k and num_valid are separate
    arguments, and the selected public function fixes descending. The
    lowering keeps runtime operands outside this record.
    """

    descending: bool = False
    k: int | None = None
    num_valid: int | None = None
    begin_bit: int | None = None
    end_bit: int | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        descending: bool,
        k: int | None,
        num_valid: int | None,
        begin_bit: int | None,
        end_bit: int | None,
        methods: dict | tuple | None,
    ) -> BlockTopKKeysOptions:
        merged = _merge_methods(
            primitive="topk_keys",
            methods=methods,
            allowed_keys=("begin_bit", "end_bit"),
            explicit_values={
                "begin_bit": begin_bit,
                "end_bit": end_bit,
            },
        )
        begin_bit = merged.get("begin_bit")
        end_bit = merged.get("end_bit")
        _validate_int_range(k, name="k", primitive="topk_keys", lower=0)
        _validate_int_range(
            num_valid, name="valid_items", primitive="topk_keys", lower=0
        )
        _validate_int_range(
            begin_bit, name="begin_bit", primitive="topk_keys", lower=0
        )
        _validate_int_range(
            end_bit, name="end_bit", primitive="topk_keys", lower=0
        )
        return cls(
            descending=descending,
            k=k,
            num_valid=num_valid,
            begin_bit=begin_bit,
            end_bit=end_bit,
            methods=merged or None,
        )


@dataclass(frozen=True)
class BlockTopKPairsOptions:
    """Normalize paired TopK controls using the key-only option contract.

    The later block adapter checks the value dtype as well as the key type.
    Shape agreement is checked by the lowering, before provider emission.
    """

    descending: bool = False
    k: int | None = None
    num_valid: int | None = None
    begin_bit: int | None = None
    end_bit: int | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        descending: bool,
        k: int | None,
        num_valid: int | None,
        begin_bit: int | None,
        end_bit: int | None,
        methods: dict | tuple | None,
    ) -> BlockTopKPairsOptions:
        merged = _merge_methods(
            primitive="topk_pairs",
            methods=methods,
            allowed_keys=("begin_bit", "end_bit"),
            explicit_values={
                "begin_bit": begin_bit,
                "end_bit": end_bit,
            },
        )
        begin_bit = merged.get("begin_bit")
        end_bit = merged.get("end_bit")
        _validate_int_range(k, name="k", primitive="topk_pairs", lower=0)
        _validate_int_range(
            num_valid, name="valid_items", primitive="topk_pairs", lower=0
        )
        _validate_int_range(
            begin_bit, name="begin_bit", primitive="topk_pairs", lower=0
        )
        _validate_int_range(
            end_bit, name="end_bit", primitive="topk_pairs", lower=0
        )
        return cls(
            descending=descending,
            k=k,
            num_valid=num_valid,
            begin_bit=begin_bit,
            end_bit=end_bit,
            methods=merged or None,
        )
