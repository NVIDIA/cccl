# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockRadixRank semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._bindings import ArgumentBinding
from .._symbols import semantic_token
from .._types import (
    INT32,
    Array,
    CxxFunction,
    Dependency,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ._common import normalize_block_dim, normalize_positive_int
from .radix import (
    RadixBitRange,
    RadixOrder,
    make_radix_bit_range,
    normalize_radix_order,
)

_KEY_T = Dependency("KeyT")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_TEMPLATE_PARAMETERS = (
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("RADIX_BITS"),
    TemplateParameter("IS_DESCENDING"),
    TemplateParameter("MEMOIZE_OUTER_SCAN"),
    TemplateParameter("INNER_SCAN_ALGORITHM"),
    TemplateParameter("SMEM_CONFIG"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


def block_radix_rank_bins_per_thread(radix_bits: int, block_threads: int) -> int:
    """Return CUB's per-thread exclusive-prefix array extent."""

    radix_bits = normalize_positive_int("radix_bits", radix_bits)
    block_threads = normalize_positive_int("block_threads", block_threads)
    return max(1, ((1 << radix_bits) + block_threads - 1) // block_threads)


@dataclass(frozen=True)
class BlockRadixRankSemantics:
    """Dimension-independent radix-rank call contract.

    Static bit intervals describe CUB's ``BFEDigitExtractor`` argument exactly.
    Runtime-classified intervals describe the equivalent operands used by a
    provider-owned runtime-width shim; such a record cannot be specialized as
    ``cub::BlockRadixRank`` until the interval becomes static.
    """

    key_dtype: Any
    items_per_thread: int
    bit_range: RadixBitRange
    order: RadixOrder
    block_threads: int | None
    exclusive_digit_prefix_items_per_thread: int | None
    parameters: tuple[Any, ...]

    @property
    def descending(self) -> bool:
        return self.order.descending

    @property
    def radix_bits(self) -> int | None:
        return self.bit_range.radix_bits

    @property
    def has_exclusive_digit_prefix(self) -> bool:
        return self.exclusive_digit_prefix_items_per_thread is not None

    @property
    def expected_exclusive_digit_prefix_items_per_thread(self) -> int | None:
        if self.radix_bits is None or self.block_threads is None:
            return None
        return block_radix_rank_bins_per_thread(
            self.radix_bits,
            self.block_threads,
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_radix_rank",
            semantic_token(self.key_dtype),
            self.items_per_thread,
            self.bit_range.semantic_key,
            self.order.value,
            self.block_threads,
            self.exclusive_digit_prefix_items_per_thread,
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockRadixRankSpec:
    """Fully specialized CUB BlockRadixRank call semantics."""

    specialization: AlgorithmSpec
    call: BlockRadixRankSemantics
    block_dim: tuple[int, int, int]

    @property
    def key_dtype(self) -> Any:
        return self.call.key_dtype

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def bit_range(self) -> RadixBitRange:
        return self.call.bit_range

    @property
    def radix_bits(self) -> int:
        assert self.call.radix_bits is not None
        return self.call.radix_bits

    @property
    def descending(self) -> bool:
        return self.call.descending

    @property
    def bins_per_thread(self) -> int:
        assert self.call.expected_exclusive_digit_prefix_items_per_thread is not None
        return self.call.expected_exclusive_digit_prefix_items_per_thread

    @property
    def has_exclusive_digit_prefix(self) -> bool:
        return self.call.has_exclusive_digit_prefix

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_radix_rank_semantics(
    *,
    key_dtype: Any,
    items_per_thread: int,
    begin_bit: int | ArgumentBinding,
    end_bit: int | ArgumentBinding,
    key_bit_width: int | None = None,
    descending: bool | RadixOrder = False,
    block_threads: int | None = None,
    exclusive_digit_prefix_items_per_thread: int | None = None,
) -> BlockRadixRankSemantics:
    """Build normalized static or runtime-width BlockRadixRank semantics."""

    if key_dtype is None:
        raise ValueError("key dtype must be provided")
    items_per_thread = normalize_positive_int("items_per_thread", items_per_thread)
    if block_threads is not None:
        block_threads = normalize_positive_int("block_threads", block_threads)
    if exclusive_digit_prefix_items_per_thread is not None:
        exclusive_digit_prefix_items_per_thread = normalize_positive_int(
            "exclusive_digit_prefix_items_per_thread",
            exclusive_digit_prefix_items_per_thread,
        )
    bit_range = make_radix_bit_range(
        begin_bit=begin_bit,
        end_bit=end_bit,
        bit_width=key_bit_width,
    )
    order = normalize_radix_order(descending)

    expected_prefix_items = None
    if bit_range.radix_bits is not None and block_threads is not None:
        expected_prefix_items = block_radix_rank_bins_per_thread(
            bit_range.radix_bits,
            block_threads,
        )
    if (
        exclusive_digit_prefix_items_per_thread is not None
        and expected_prefix_items is not None
        and exclusive_digit_prefix_items_per_thread != expected_prefix_items
    ):
        raise ValueError(
            "exclusive_digit_prefix must contain "
            f"{expected_prefix_items} items per thread"
        )

    parameters: list[Any] = [
        TempStorageParameter(),
        Array(_KEY_T, _ITEMS_PER_THREAD, name="keys"),
        Array(
            INT32,
            _ITEMS_PER_THREAD,
            name="ranks",
            is_output=True,
            is_return=False,
        ),
    ]
    if bit_range.is_static:
        assert bit_range.static_begin_bit is not None
        assert bit_range.radix_bits is not None
        parameters.append(
            CxxFunction(
                "::cub::BFEDigitExtractor<KeyT>"
                f"({bit_range.static_begin_bit}, {bit_range.radix_bits})",
                _KEY_T,
                name="digit_extractor",
            )
        )
    else:
        # Provider shims may keep the interval runtime-valued even though the
        # native CUB class specialization requires a static RADIX_BITS value.
        parameters.extend(
            (
                Value(INT32, name="begin_bit"),
                Value(INT32, name="end_bit"),
            )
        )
    if exclusive_digit_prefix_items_per_thread is not None:
        parameters.append(
            Array(
                INT32,
                exclusive_digit_prefix_items_per_thread,
                name="exclusive_digit_prefix",
                is_output=True,
                is_return=False,
            )
        )

    return BlockRadixRankSemantics(
        key_dtype=key_dtype,
        items_per_thread=items_per_thread,
        bit_range=bit_range,
        order=order,
        block_threads=block_threads,
        exclusive_digit_prefix_items_per_thread=(
            exclusive_digit_prefix_items_per_thread
        ),
        parameters=tuple(parameters),
    )


def make_block_radix_rank_spec(
    *,
    key_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    begin_bit: int,
    end_bit: int,
    key_bit_width: int | None = None,
    descending: bool | RadixOrder = False,
    with_exclusive_digit_prefix: bool = False,
) -> BlockRadixRankSpec:
    """Build a fully specialized CUB BlockRadixRank description."""

    if not isinstance(with_exclusive_digit_prefix, bool):
        raise ValueError("with_exclusive_digit_prefix must be a boolean")
    block_dim = normalize_block_dim(block_dim)
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    bit_range = make_radix_bit_range(
        begin_bit=begin_bit,
        end_bit=end_bit,
        bit_width=key_bit_width,
    )
    if not bit_range.is_static or bit_range.radix_bits is None:
        raise ValueError("cub::BlockRadixRank requires a static radix bit range")
    bins_per_thread = block_radix_rank_bins_per_thread(
        bit_range.radix_bits,
        block_threads,
    )
    call = make_block_radix_rank_semantics(
        key_dtype=key_dtype,
        items_per_thread=items_per_thread,
        begin_bit=bit_range.begin_bit,
        end_bit=bit_range.end_bit,
        key_bit_width=key_bit_width,
        descending=descending,
        block_threads=block_threads,
        exclusive_digit_prefix_items_per_thread=(
            bins_per_thread if with_exclusive_digit_prefix else None
        ),
    )
    specialization = Algorithm(
        struct_name="BlockRadixRank",
        method_name="RankKeys",
        c_name="block_radix_rank",
        includes=("cub/block/block_radix_rank.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=(call.parameters,),
    ).specialize(
        {
            "BLOCK_DIM_X": block_dim[0],
            "RADIX_BITS": call.radix_bits,
            "IS_DESCENDING": call.order.cpp_bool,
            "MEMOIZE_OUTER_SCAN": "true",
            "INNER_SCAN_ALGORITHM": (
                "::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS"
            ),
            "SMEM_CONFIG": "cudaSharedMemBankSizeFourByte",
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
            "KeyT": key_dtype,
            "ITEMS_PER_THREAD": call.items_per_thread,
        },
        metadata={
            "scope": "block",
            "primitive": "radix_rank",
            "order": call.order,
            "begin_bit": call.bit_range.static_begin_bit,
            "end_bit": call.bit_range.static_end_bit,
            "exclusive_digit_prefix": call.has_exclusive_digit_prefix,
        },
    )
    return BlockRadixRankSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockRadixRankSemantics",
    "BlockRadixRankSpec",
    "block_radix_rank_bins_per_thread",
    "make_block_radix_rank_semantics",
    "make_block_radix_rank_spec",
]
