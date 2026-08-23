# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import INT32, Array, Dependency, TempStorageParameter, Value
from cuda.coop._core.block import (
    BlockRadixSortBitPolicy,
    BlockRadixSortOutput,
    BlockRadixSortPayload,
    RadixOrder,
    make_block_radix_sort_semantics,
    make_block_radix_sort_spec,
)


def test_default_key_sort_spec_owns_cub_specialization_and_abi():
    spec = make_block_radix_sort_spec(
        key_dtype="i32",
        key_bit_width=32,
        block_dim=(16, 2, 1),
        items_per_thread=4,
    )

    assert spec.payload is BlockRadixSortPayload.KEYS
    assert spec.order is RadixOrder.ASCENDING
    assert spec.output is BlockRadixSortOutput.BLOCKED
    assert spec.bit_policy is BlockRadixSortBitPolicy.DEFAULT
    assert spec.method_name == "Sort"
    assert spec.specialization.template_arguments == {
        "KeyT": "i32",
        "BLOCK_DIM_X": 16,
        "ITEMS_PER_THREAD": 4,
        "ValueT": "::cub::NullType",
        "RADIX_BITS": 4,
        "MEMOIZE_OUTER_SCAN": "true",
        "INNER_SCAN_ALGORITHM": ("::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS"),
        "SMEM_CONFIG": "cudaSharedMemBankSizeFourByte",
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("KeyT"),
                Dependency("ITEMS_PER_THREAD"),
                name="keys",
                is_inout=True,
                is_return=False,
            ),
        ),
    )


def test_explicit_descending_striped_pair_sort_owns_runtime_bit_abi():
    spec = make_block_radix_sort_spec(
        key_dtype="u64",
        value_dtype="i32",
        key_bit_width=64,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        descending=True,
        blocked_to_striped=True,
        begin_bit=4,
        end_bit=28,
    )

    assert spec.payload is BlockRadixSortPayload.PAIRS
    assert spec.order is RadixOrder.DESCENDING
    assert spec.output is BlockRadixSortOutput.STRIPED
    assert spec.bit_policy is BlockRadixSortBitPolicy.EXPLICIT
    assert spec.method_name == "SortDescendingBlockedToStriped"
    assert spec.call.bit_range is not None
    assert not spec.call.bit_range.is_static
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("KeyT"),
                Dependency("ITEMS_PER_THREAD"),
                name="keys",
                is_inout=True,
                is_return=False,
            ),
            Array(
                Dependency("ValueT"),
                Dependency("ITEMS_PER_THREAD"),
                name="values",
                is_inout=True,
                is_return=False,
            ),
            Value(INT32, name="begin_bit"),
            Value(INT32, name="end_bit"),
        ),
    )


def test_both_policy_preserves_default_and_explicit_overload_order():
    spec = make_block_radix_sort_spec(
        key_dtype="u32",
        key_bit_width=32,
        block_dim=(64, 1, 1),
        items_per_thread=3,
        bit_policy=BlockRadixSortBitPolicy.BOTH,
    )

    assert len(spec.specialization.parameters) == 2
    assert [parameter.name for parameter in spec.specialization.parameters[0]] == [
        "temp_storage",
        "keys",
    ]
    assert [parameter.name for parameter in spec.specialization.parameters[1]] == [
        "temp_storage",
        "keys",
        "begin_bit",
        "end_bit",
    ]

    with_bounds = make_block_radix_sort_semantics(
        key_dtype="u32",
        key_bit_width=32,
        items_per_thread=3,
        bit_policy=BlockRadixSortBitPolicy.BOTH,
        begin_bit=4,
        end_bit=12,
    )
    assert with_bounds.bit_range is not None
    assert not with_bounds.bit_range.is_static
    with pytest.raises(ValueError, match="dtype bit width"):
        make_block_radix_sort_semantics(
            key_dtype="u32",
            key_bit_width=32,
            items_per_thread=3,
            bit_policy=BlockRadixSortBitPolicy.BOTH,
            begin_bit=0,
            end_bit=33,
        )


def test_runtime_bit_values_do_not_fragment_semantic_identity():
    def make(begin_bit, end_bit):
        return make_block_radix_sort_semantics(
            key_dtype="u32",
            key_bit_width=32,
            items_per_thread=2,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    assert make(0, 4).semantic_key == make(8, 16).semantic_key
    assert (
        make(0, 4).semantic_key
        != make_block_radix_sort_semantics(
            key_dtype="u32",
            key_bit_width=32,
            items_per_thread=2,
        ).semantic_key
    )


def test_sort_boolean_options_accept_numpy_scalars_but_reject_truthy_integers():
    numpy_bool_base = type(
        "bool_",
        (),
        {"__module__": "numpy", "__bool__": lambda self: True},
    )

    spec = make_block_radix_sort_spec(
        key_dtype="u32",
        key_bit_width=32,
        block_dim=(32, 1, 1),
        items_per_thread=1,
        descending=np.bool_(True),
        blocked_to_striped=np.bool_(True),
    )
    assert spec.descending
    assert spec.blocked_to_striped
    assert make_block_radix_sort_spec(
        key_dtype="u32",
        block_dim=(32, 1, 1),
        items_per_thread=1,
        descending=type("DerivedBool", (numpy_bool_base,), {})(),
    ).descending

    moduleless_type = type("Moduleless", (), {})
    moduleless_type.__module__ = None
    with pytest.raises(ValueError, match="descending must be a boolean"):
        make_block_radix_sort_spec(
            key_dtype="u32",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            descending=moduleless_type(),
        )

    with pytest.raises(ValueError, match="blocked_to_striped must be a boolean"):
        make_block_radix_sort_spec(
            key_dtype="u32",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            blocked_to_striped=1,
        )


@pytest.mark.parametrize("items_per_thread", [0, -1, True, 1.5, "two"])
def test_sort_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        make_block_radix_sort_semantics(
            key_dtype="u32",
            items_per_thread=items_per_thread,
        )


def test_sort_rejects_invalid_bit_overload_shapes_and_bounds():
    with pytest.raises(ValueError, match="provided together"):
        make_block_radix_sort_semantics(
            key_dtype="u32",
            items_per_thread=1,
            begin_bit=0,
        )
    with pytest.raises(ValueError, match="greater than begin_bit"):
        make_block_radix_sort_semantics(
            key_dtype="u32",
            key_bit_width=32,
            items_per_thread=1,
            begin_bit=8,
            end_bit=8,
        )
    with pytest.raises(ValueError, match="dtype bit width"):
        make_block_radix_sort_semantics(
            key_dtype="u32",
            key_bit_width=32,
            items_per_thread=1,
            begin_bit=0,
            end_bit=33,
        )
    with pytest.raises(ValueError, match="explicit bit policy requires"):
        make_block_radix_sort_semantics(
            key_dtype="u32",
            items_per_thread=1,
            bit_policy="explicit",
        )
