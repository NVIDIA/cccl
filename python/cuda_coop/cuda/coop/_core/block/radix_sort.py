# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockRadixSort semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._bindings import ArgumentBinding
from .._symbols import semantic_token
from .._types import (
    INT32,
    Array,
    Dependency,
    RuntimeValue,
    TemplateParameter,
    TempStorageParameter,
    Value,
)
from ._common import (
    normalize_block_dim,
    normalize_boolean_option,
    normalize_positive_int,
)
from .radix import (
    RadixBitRange,
    RadixOrder,
    make_radix_bit_range,
    normalize_radix_order,
)


class BlockRadixSortPayload(str, Enum):
    """Items transported by a block radix sort."""

    KEYS = "keys"
    PAIRS = "pairs"


class BlockRadixSortOutput(str, Enum):
    """Per-thread arrangement produced by the final radix pass."""

    BLOCKED = "blocked"
    STRIPED = "striped"


class BlockRadixSortBitPolicy(str, Enum):
    """Runtime bit-range overloads represented by a generated wrapper."""

    DEFAULT = "default"
    EXPLICIT = "explicit"
    BOTH = "both"

    @property
    def includes_default(self) -> bool:
        return self in {BlockRadixSortBitPolicy.DEFAULT, BlockRadixSortBitPolicy.BOTH}

    @property
    def includes_explicit(self) -> bool:
        return self in {
            BlockRadixSortBitPolicy.EXPLICIT,
            BlockRadixSortBitPolicy.BOTH,
        }


_KEY_T = Dependency("KeyT")
_VALUE_T = Dependency("ValueT")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_TEMPLATE_PARAMETERS = (
    TemplateParameter("KeyT"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("ValueT"),
    TemplateParameter("RADIX_BITS"),
    TemplateParameter("MEMOIZE_OUTER_SCAN"),
    TemplateParameter("INNER_SCAN_ALGORITHM"),
    TemplateParameter("SMEM_CONFIG"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


def _runtime_bit_range(
    *,
    begin_bit: Any,
    end_bit: Any,
    key_bit_width: int | None,
) -> RadixBitRange:
    """Validate known bounds, then retain only their runtime ABI identity."""

    make_radix_bit_range(
        begin_bit=begin_bit,
        end_bit=end_bit,
        bit_width=key_bit_width,
    )
    return make_radix_bit_range(
        begin_bit=RuntimeValue("begin_bit"),
        end_bit=RuntimeValue("end_bit"),
        bit_width=key_bit_width,
    )


def _method_parameters(
    *,
    payload: BlockRadixSortPayload,
    with_bits: bool,
) -> tuple[Any, ...]:
    parameters: list[Any] = [
        TempStorageParameter(),
        Array(
            _KEY_T,
            _ITEMS_PER_THREAD,
            name="keys",
            is_inout=True,
            is_return=False,
        ),
    ]
    if payload is BlockRadixSortPayload.PAIRS:
        parameters.append(
            Array(
                _VALUE_T,
                _ITEMS_PER_THREAD,
                name="values",
                is_inout=True,
                is_return=False,
            )
        )
    if with_bits:
        parameters.extend(
            (
                Value(INT32, name="begin_bit"),
                Value(INT32, name="end_bit"),
            )
        )
    return tuple(parameters)


@dataclass(frozen=True)
class BlockRadixSortSemantics:
    """Dimension-independent block-radix-sort call contract."""

    key_dtype: Any
    value_dtype: Any | None
    items_per_thread: int
    payload: BlockRadixSortPayload
    order: RadixOrder
    output: BlockRadixSortOutput
    bit_policy: BlockRadixSortBitPolicy
    bit_range: RadixBitRange | None
    parameters: tuple[tuple[Any, ...], ...]

    @property
    def descending(self) -> bool:
        return self.order.descending

    @property
    def has_values(self) -> bool:
        return self.payload is BlockRadixSortPayload.PAIRS

    @property
    def blocked_to_striped(self) -> bool:
        return self.output is BlockRadixSortOutput.STRIPED

    @property
    def method_name(self) -> str:
        if self.blocked_to_striped:
            return (
                "SortDescendingBlockedToStriped"
                if self.descending
                else "SortBlockedToStriped"
            )
        return "SortDescending" if self.descending else "Sort"

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            "block_radix_sort",
            semantic_token(self.key_dtype),
            semantic_token(self.value_dtype),
            self.items_per_thread,
            self.payload.value,
            self.order.value,
            self.output.value,
            self.bit_policy.value,
            semantic_token(self.bit_range),
            semantic_token(self.parameters),
        )


@dataclass(frozen=True)
class BlockRadixSortSpec:
    """Fully specialized CUB BlockRadixSort call semantics."""

    specialization: AlgorithmSpec
    call: BlockRadixSortSemantics
    block_dim: tuple[int, int, int]

    @property
    def key_dtype(self) -> Any:
        return self.call.key_dtype

    @property
    def value_dtype(self) -> Any | None:
        return self.call.value_dtype

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def payload(self) -> BlockRadixSortPayload:
        return self.call.payload

    @property
    def order(self) -> RadixOrder:
        return self.call.order

    @property
    def output(self) -> BlockRadixSortOutput:
        return self.call.output

    @property
    def bit_policy(self) -> BlockRadixSortBitPolicy:
        return self.call.bit_policy

    @property
    def descending(self) -> bool:
        return self.call.descending

    @property
    def blocked_to_striped(self) -> bool:
        return self.call.blocked_to_striped

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_radix_sort_semantics(
    *,
    key_dtype: Any,
    items_per_thread: int,
    descending: bool | RadixOrder = False,
    value_dtype: Any | None = None,
    blocked_to_striped: bool | BlockRadixSortOutput = False,
    begin_bit: int | ArgumentBinding | None = None,
    end_bit: int | ArgumentBinding | None = None,
    key_bit_width: int | None = None,
    bit_policy: str | BlockRadixSortBitPolicy | None = None,
) -> BlockRadixSortSemantics:
    """Build normalized default- or runtime-bit BlockRadixSort semantics."""

    if key_dtype is None:
        raise ValueError("key dtype must be provided")
    items_per_thread = normalize_positive_int("items_per_thread", items_per_thread)
    order = normalize_radix_order(descending)
    if isinstance(blocked_to_striped, BlockRadixSortOutput):
        output = blocked_to_striped
    else:
        output = (
            BlockRadixSortOutput.STRIPED
            if normalize_boolean_option("blocked_to_striped", blocked_to_striped)
            else BlockRadixSortOutput.BLOCKED
        )
    payload = (
        BlockRadixSortPayload.PAIRS
        if value_dtype is not None
        else BlockRadixSortPayload.KEYS
    )

    has_begin = begin_bit is not None
    has_end = end_bit is not None
    if has_begin != has_end:
        raise ValueError("begin_bit and end_bit must be provided together")
    if bit_policy is None:
        bit_policy = (
            BlockRadixSortBitPolicy.EXPLICIT
            if has_begin
            else BlockRadixSortBitPolicy.DEFAULT
        )
    else:
        bit_policy = BlockRadixSortBitPolicy(bit_policy)
    if bit_policy is BlockRadixSortBitPolicy.DEFAULT and has_begin:
        raise ValueError("default bit policy cannot include explicit bit bounds")
    if bit_policy is BlockRadixSortBitPolicy.EXPLICIT and not has_begin:
        raise ValueError("explicit bit policy requires begin_bit and end_bit")

    bit_range = None
    if has_begin:
        bit_range = _runtime_bit_range(
            begin_bit=begin_bit,
            end_bit=end_bit,
            key_bit_width=key_bit_width,
        )
    elif bit_policy.includes_explicit:
        bit_range = make_radix_bit_range(
            begin_bit=RuntimeValue("begin_bit"),
            end_bit=RuntimeValue("end_bit"),
            bit_width=key_bit_width,
        )

    parameters: list[tuple[Any, ...]] = []
    if bit_policy.includes_default:
        parameters.append(_method_parameters(payload=payload, with_bits=False))
    if bit_policy.includes_explicit:
        parameters.append(_method_parameters(payload=payload, with_bits=True))

    return BlockRadixSortSemantics(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        payload=payload,
        order=order,
        output=output,
        bit_policy=bit_policy,
        bit_range=bit_range,
        parameters=tuple(parameters),
    )


def make_block_radix_sort_spec(
    *,
    key_dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    descending: bool | RadixOrder = False,
    value_dtype: Any | None = None,
    blocked_to_striped: bool | BlockRadixSortOutput = False,
    begin_bit: int | ArgumentBinding | None = None,
    end_bit: int | ArgumentBinding | None = None,
    key_bit_width: int | None = None,
    bit_policy: str | BlockRadixSortBitPolicy | None = None,
) -> BlockRadixSortSpec:
    """Build a fully specialized CUB BlockRadixSort description."""

    block_dim = normalize_block_dim(block_dim)
    call = make_block_radix_sort_semantics(
        key_dtype=key_dtype,
        items_per_thread=items_per_thread,
        descending=descending,
        value_dtype=value_dtype,
        blocked_to_striped=blocked_to_striped,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=key_bit_width,
        bit_policy=bit_policy,
    )
    specialization = Algorithm(
        struct_name="BlockRadixSort",
        method_name=call.method_name,
        c_name="block_radix_sort",
        includes=("cub/block/block_radix_sort.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=call.parameters,
    ).specialize(
        {
            "KeyT": key_dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": call.items_per_thread,
            "ValueT": value_dtype if value_dtype is not None else "::cub::NullType",
            "RADIX_BITS": 4,
            "MEMOIZE_OUTER_SCAN": "true",
            "INNER_SCAN_ALGORITHM": (
                "::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS"
            ),
            "SMEM_CONFIG": "cudaSharedMemBankSizeFourByte",
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
        },
        metadata={
            "scope": "block",
            "primitive": "radix_sort",
            "payload": call.payload,
            "order": call.order,
            "output": call.output,
            "bit_policy": call.bit_policy,
        },
    )
    return BlockRadixSortSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


__all__ = [
    "BlockRadixSortBitPolicy",
    "BlockRadixSortOutput",
    "BlockRadixSortPayload",
    "BlockRadixSortSemantics",
    "BlockRadixSortSpec",
    "make_block_radix_sort_semantics",
    "make_block_radix_sort_spec",
]
