# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import (
    INT32,
    ArgumentBinding,
    ArgumentKind,
    Array,
    CxxFunction,
    Dependency,
    ParameterRole,
    RuntimeValue,
    TempStorageParameter,
    Value,
)
from cuda.coop._core.block import (
    RadixOrder,
    block_radix_rank_bins_per_thread,
    make_block_radix_rank_semantics,
    make_block_radix_rank_spec,
    make_radix_bit_range,
    resolve_static_radix_end_bit,
)


def test_radix_rank_spec_owns_cub_specialization_and_full_abi():
    spec = make_block_radix_rank_spec(
        key_dtype="u32",
        key_bit_width=32,
        block_dim=(16, 2, 1),
        items_per_thread=4,
        begin_bit=3,
        end_bit=9,
        descending=True,
        with_exclusive_digit_prefix=True,
    )

    assert spec.block_dim == (16, 2, 1)
    assert spec.radix_bits == 6
    assert spec.descending
    assert spec.bins_per_thread == 2
    assert spec.has_exclusive_digit_prefix
    assert spec.specialization.template_arguments == {
        "BLOCK_DIM_X": 16,
        "RADIX_BITS": 6,
        "IS_DESCENDING": "true",
        "MEMOIZE_OUTER_SCAN": "true",
        "INNER_SCAN_ALGORITHM": ("::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS"),
        "SMEM_CONFIG": "cudaSharedMemBankSizeFourByte",
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
        "KeyT": "u32",
        "ITEMS_PER_THREAD": 4,
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD"), name="keys"),
            Array(
                INT32,
                Dependency("ITEMS_PER_THREAD"),
                name="ranks",
                is_output=True,
                is_return=False,
            ),
            CxxFunction(
                "::cub::BFEDigitExtractor<KeyT>(3, 6)",
                Dependency("KeyT"),
                name="digit_extractor",
            ),
            Array(
                INT32,
                2,
                name="exclusive_digit_prefix",
                is_output=True,
                is_return=False,
            ),
        ),
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.STATIC, ParameterRole.CONSTANT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]


def test_runtime_width_semantics_drop_provider_payloads():
    first = make_block_radix_rank_semantics(
        key_dtype="u64",
        key_bit_width=64,
        items_per_thread=3,
        begin_bit=RuntimeValue("first_begin"),
        end_bit=RuntimeValue("first_end"),
        descending=False,
        exclusive_digit_prefix_items_per_thread=2,
    )
    second = make_block_radix_rank_semantics(
        key_dtype="u64",
        key_bit_width=64,
        items_per_thread=3,
        begin_bit=RuntimeValue("other_begin"),
        end_bit=RuntimeValue("other_end"),
        descending=RadixOrder.ASCENDING,
        exclusive_digit_prefix_items_per_thread=2,
    )

    assert not first.bit_range.is_static
    assert first.radix_bits is None
    assert first.semantic_key == second.semantic_key
    assert first.parameters[3:5] == (
        Value(INT32, name="begin_bit"),
        Value(INT32, name="end_bit"),
    )
    assert first.parameters[-1] == Array(
        INT32,
        2,
        name="exclusive_digit_prefix",
        is_output=True,
        is_return=False,
    )


def test_radix_range_supports_explicit_runtime_bindings():
    interval = make_radix_bit_range(
        begin_bit=ArgumentBinding.runtime(),
        end_bit=ArgumentBinding.static(16),
        bit_width=32,
    )

    assert not interval.is_static
    assert interval.static_begin_bit is None
    assert interval.static_end_bit == 16
    assert interval.radix_bits is None

    with pytest.raises(ValueError, match="end_bit must be positive"):
        make_radix_bit_range(
            begin_bit=ArgumentBinding.runtime(),
            end_bit=ArgumentBinding.static(-1),
            bit_width=32,
        )


def test_static_default_resolution_preserves_frontend_policy():
    assert (
        resolve_static_radix_end_bit(
            begin_bit=30,
            end_bit=None,
            bit_width=32,
            default_radix_bits=4,
            clamp_default=True,
        )
        == 32
    )
    assert (
        resolve_static_radix_end_bit(
            begin_bit=0,
            end_bit=None,
            bit_width=64,
            default_to_bit_width=True,
        )
        == 64
    )
    with pytest.raises(ValueError, match="bit_width is unavailable"):
        resolve_static_radix_end_bit(
            begin_bit=0,
            end_bit=None,
            bit_width=None,
            default_radix_bits=4,
            clamp_default=True,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "two"])
def test_radix_rank_rejects_invalid_item_count(value):
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        make_block_radix_rank_semantics(
            key_dtype="u32",
            key_bit_width=32,
            items_per_thread=value,
            begin_bit=0,
            end_bit=4,
        )


@pytest.mark.parametrize(
    ("begin_bit", "end_bit", "message"),
    [
        (True, 4, "begin_bit must be an integer"),
        (1.5, 4, "begin_bit must be an integer"),
        (-1, 4, "begin_bit must be non-negative"),
        (32, 32, "begin_bit must be less than the dtype bit width"),
        (5, 5, "end_bit must be greater than begin_bit"),
        (0, 33, "end_bit must not exceed the dtype bit width"),
    ],
)
def test_radix_rank_rejects_invalid_static_bit_ranges(begin_bit, end_bit, message):
    with pytest.raises(ValueError, match=message):
        make_radix_bit_range(
            begin_bit=begin_bit,
            end_bit=end_bit,
            bit_width=32,
        )


def test_radix_rank_rejects_invalid_order_shape_and_prefix_extent():
    legacy_numpy_bool = type(
        "bool_",
        (),
        {
            "__module__": "numpy",
            "__bool__": lambda self: True,
        },
    )()
    assert make_block_radix_rank_spec(
        key_dtype="u32",
        key_bit_width=32,
        block_dim=(32, 1, 1),
        items_per_thread=1,
        begin_bit=0,
        end_bit=4,
        descending=np.bool_(True),
    ).descending
    assert make_block_radix_rank_spec(
        key_dtype="u32",
        key_bit_width=32,
        block_dim=(32, 1, 1),
        items_per_thread=1,
        begin_bit=0,
        end_bit=4,
        descending=legacy_numpy_bool,
    ).descending
    with pytest.raises(ValueError, match="descending must be a boolean"):
        make_block_radix_rank_spec(
            key_dtype="u32",
            key_bit_width=32,
            block_dim=(32, 1, 1),
            items_per_thread=1,
            begin_bit=0,
            end_bit=4,
            descending=1,
        )
    with pytest.raises(ValueError, match="block_dim"):
        make_block_radix_rank_spec(
            key_dtype="u32",
            key_bit_width=32,
            block_dim=(32, 0, 1),
            items_per_thread=1,
            begin_bit=0,
            end_bit=4,
        )
    with pytest.raises(ValueError, match="must contain 4 items per thread"):
        make_block_radix_rank_semantics(
            key_dtype="u32",
            key_bit_width=32,
            items_per_thread=1,
            begin_bit=0,
            end_bit=6,
            block_threads=16,
            exclusive_digit_prefix_items_per_thread=2,
        )


def test_bins_per_thread_matches_cub_extent_formula():
    assert block_radix_rank_bins_per_thread(4, 32) == 1
    assert block_radix_rank_bins_per_thread(6, 16) == 4
