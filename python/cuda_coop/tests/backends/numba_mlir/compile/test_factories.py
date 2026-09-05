# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from inspect import signature
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._types import Invocable

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

LTOIR_MAGIC = bytes.fromhex("ed434e7f")


def _binary_op(lhs, rhs):
    return lhs + rhs


def _prefix_op(value):
    return value


def _compare_op(lhs, rhs):
    return lhs < rhs


def _difference_op(lhs, rhs):
    return lhs - rhs


def _flag_op(lhs, rhs):
    return lhs != rhs


def _assert_ltoir_files(instance):
    assert instance.files
    for filename in instance.files:
        path = Path(filename)
        assert path.suffix == ".ltoir"
        assert path.read_bytes().startswith(LTOIR_MAGIC)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: coop._block.make_load(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_store(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_exchange(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_merge_sort_keys(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            compare_op=_compare_op,
        ),
        lambda: coop._block.make_merge_sort_pairs(
            types.int32,
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            compare_op=_compare_op,
        ),
        lambda: coop._block.make_radix_sort_keys(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_sort_keys_descending(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_sort_pairs(
            types.int32, types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_sort_pairs_descending(
            types.int32, types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_topk_max_keys(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_topk_min_keys(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            num_valid=np.int32(17),
        ),
        lambda: coop._block.make_topk_max_pairs(
            types.int32, types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_topk_min_pairs(
            types.float32, types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_rank(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=0,
            end_bit=8,
        ),
        lambda: coop._block.make_reduce(
            types.int32,
            threads_per_block=64,
            binary_op=_binary_op,
            items_per_thread=2,
        ),
        lambda: coop._block.make_reduce(
            types.int32,
            threads_per_block=64,
            binary_op=_binary_op,
            num_valid=32,
        ),
        lambda: coop._block.make_sum(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_sum(types.int32, threads_per_block=64, num_valid=32),
        lambda: coop._block.make_scan(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_scan(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            block_aggregate=True,
        ),
        lambda: coop._block.make_exclusive_sum(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_inclusive_sum(
            types.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop._block.make_exclusive_scan(
            types.int32,
            threads_per_block=64,
            scan_op="+",
            items_per_thread=2,
        ),
        lambda: coop._block.make_inclusive_scan(
            types.int32,
            threads_per_block=64,
            scan_op="+",
            items_per_thread=2,
        ),
        lambda: coop._block.make_adjacent_difference(
            types.int32,
            threads_per_block=64,
            items_per_thread=1,
            difference_op=_difference_op,
        ),
        lambda: coop._block.make_discontinuity(
            types.int32,
            threads_per_block=64,
            items_per_thread=1,
            flag_op=_flag_op,
        ),
        lambda: coop._block.make_shuffle(
            types.int32, threads_per_block=64, items_per_thread=1
        ),
    ],
)
def test_make_block_factories_return_invocable(factory):
    instance = factory()
    assert isinstance(instance, Invocable)
    _assert_ltoir_files(instance)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: coop._warp.make_load(
            types.int32, items_per_thread=2, threads_in_warp=32
        ),
        lambda: coop._warp.make_store(
            types.int32, items_per_thread=2, threads_in_warp=32
        ),
        lambda: coop._warp.make_exchange(
            types.int32, items_per_thread=2, threads_in_warp=32
        ),
        lambda: coop._warp.make_reduce(types.int32, _binary_op, threads_in_warp=32),
        lambda: coop._warp.make_sum(types.int32, threads_in_warp=32),
        lambda: coop._warp.make_max(types.int32, threads_in_warp=32),
        lambda: coop._warp.make_min(types.int32, threads_in_warp=32),
        lambda: coop._warp.make_exclusive_sum(types.int32, threads_in_warp=32),
        lambda: coop._warp.make_inclusive_sum(types.int32, threads_in_warp=32),
        lambda: coop._warp.make_exclusive_scan(
            types.int32, _binary_op, threads_in_warp=32
        ),
        lambda: coop._warp.make_inclusive_scan(
            types.int32, _binary_op, threads_in_warp=32
        ),
        lambda: coop._warp.make_merge_sort_keys(
            types.int32,
            items_per_thread=2,
            compare_op=_compare_op,
            threads_in_warp=32,
        ),
        lambda: coop._warp.make_merge_sort_pairs(
            types.int32,
            types.int32,
            items_per_thread=2,
            compare_op=_compare_op,
            threads_in_warp=32,
        ),
    ],
)
def test_make_warp_factories_return_invocable(factory):
    instance = factory()
    assert isinstance(instance, Invocable)
    _assert_ltoir_files(instance)


def test_make_warp_exchange_rejects_output_form_for_non_scatter_mode():
    with pytest.raises(ValueError, match="only supported for ScatterToStriped"):
        coop._warp.make_exchange(
            types.int32,
            items_per_thread=2,
            threads_in_warp=32,
            warp_exchange_type=coop._warp.WarpExchangeType.StripedToBlocked,
            use_output_items=False,
        )


def test_make_histogram_returns_stateful_instance():
    histo = coop._block.make_histogram(
        types.uint8,
        types.uint32,
        threads_per_block=64,
        items_per_thread=1,
    )

    assert hasattr(histo, "init")
    assert hasattr(histo, "composite")


def test_make_scan_rejects_block_aggregate_with_prefix_op():
    with pytest.raises(ValueError, match="block_aggregate"):
        coop._block.make_scan(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            prefix_op=_prefix_op,
            block_aggregate=True,
        )


def test_make_histogram_accepts_numba_dtype_keyword_aliases():
    histo = coop._block.make_histogram(
        item_dtype=types.uint8,
        counter_dtype=types.uint32,
        threads_per_block=64,
        items_per_thread=1,
    )

    assert hasattr(histo, "init")
    assert hasattr(histo, "composite")


def test_make_histogram_rejects_explicit_temp_storage():
    with pytest.raises(
        NotImplementedError,
        match="Explicit temp_storage is not supported for histogram.",
    ):
        coop._block.make_histogram(
            types.uint8,
            types.uint32,
            threads_per_block=64,
            items_per_thread=1,
            temp_storage=object(),
        )


def test_make_histogram_rejects_invalid_static_shape():
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        coop._block.make_histogram(
            types.uint8,
            types.uint32,
            threads_per_block=64,
            items_per_thread=True,
        )
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        coop._block.make_histogram(
            types.uint8,
            types.uint32,
            threads_per_block=64,
            items_per_thread=1,
            bins=0,
        )


def test_make_run_length_returns_stateful_instance():
    rle = coop._block.make_run_length(
        types.int32,
        threads_per_block=64,
        runs_per_thread=1,
        decoded_items_per_thread=1,
    )

    assert hasattr(rle, "decode")


def test_make_run_length_accepts_item_dtype_keyword_alias():
    rle = coop._block.make_run_length(
        item_dtype=types.int32,
        threads_per_block=64,
        runs_per_thread=1,
        decoded_items_per_thread=1,
    )

    assert hasattr(rle, "decode")


@pytest.mark.parametrize("value", [True, 1.5])
def test_make_run_length_rejects_legacy_integer_coercions(value):
    with pytest.raises(ValueError, match="runs_per_thread must be a positive integer"):
        coop._block.make_run_length(
            types.int32,
            threads_per_block=64,
            runs_per_thread=value,
            decoded_items_per_thread=1,
        )


def test_make_radix_sort_pairs_accepts_numba_dtype_keyword_aliases():
    instance = coop._block.make_radix_sort_pairs(
        keys=types.int32,
        values=types.int32,
        threads_per_block=64,
        items_per_thread=2,
    )

    assert isinstance(instance, Invocable)


def test_make_radix_sort_rejects_incomplete_bit_range():
    with pytest.raises(ValueError, match="begin_bit and end_bit"):
        coop._block.make_radix_sort_keys(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=0,
        )


def test_make_radix_sort_rejects_invalid_core_options():
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        coop._block.make_radix_sort_keys(
            types.int32,
            threads_per_block=64,
            items_per_thread=1.5,
        )
    with pytest.raises(ValueError, match="blocked_to_striped must be a boolean"):
        coop._block.make_radix_sort_keys(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            blocked_to_striped=1,
        )
    with pytest.raises(ValueError, match="dtype bit width"):
        coop._block.make_radix_sort_keys(
            types.uint32,
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=0,
            end_bit=33,
        )


def test_make_radix_sort_does_not_expose_decomposer():
    for factory in (
        coop._block.make_radix_sort_keys,
        coop._block.make_radix_sort_keys_descending,
        coop._block.make_radix_sort_pairs,
        coop._block.make_radix_sort_pairs_descending,
    ):
        assert "decomposer" not in signature(factory).parameters

    with pytest.raises(TypeError, match="unexpected keyword argument 'decomposer'"):
        coop._block.make_radix_sort_keys(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            decomposer=object(),
        )


def test_make_topk_pairs_accepts_numba_dtype_keyword_aliases():
    instance = coop._block.make_topk_max_pairs(
        key_dtype=types.int32,
        value_dtype=types.int32,
        threads_per_block=64,
        items_per_thread=2,
    )

    assert isinstance(instance, Invocable)


def test_make_topk_rejects_incomplete_bit_range():
    with pytest.raises(ValueError, match="begin_bit and end_bit"):
        coop._block.make_topk_max_keys(
            types.uint32,
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=0,
        )


def test_make_store_rejects_oob_default():
    with pytest.raises(ValueError, match="oob_default is only valid for BlockLoad"):
        coop._block.make_store(
            types.int32,
            threads_per_block=64,
            items_per_thread=2,
            oob_default=types.int32(0),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: coop._block.make_load(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_store(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_histogram(
            types.uint8,
            types.uint32,
            dim=64,
            items_per_thread=1,
        ),
        lambda: coop._block.make_run_length(
            types.int32,
            dim=64,
            runs_per_thread=1,
            decoded_items_per_thread=1,
        ),
        lambda: coop._block.make_exchange(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_merge_sort_keys(
            types.int32,
            dim=64,
            items_per_thread=2,
            compare_op=_compare_op,
        ),
        lambda: coop._block.make_merge_sort_pairs(
            types.int32,
            types.int32,
            dim=64,
            items_per_thread=2,
            compare_op=_compare_op,
        ),
        lambda: coop._block.make_radix_sort_keys(
            types.int32, dim=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_sort_keys_descending(
            types.int32, dim=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_sort_pairs(
            types.int32, types.int32, dim=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_sort_pairs_descending(
            types.int32, types.int32, dim=64, items_per_thread=2
        ),
        lambda: coop._block.make_topk_max_keys(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_topk_min_keys(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_topk_max_pairs(
            types.int32, types.int32, dim=64, items_per_thread=2
        ),
        lambda: coop._block.make_topk_min_pairs(
            types.int32, types.int32, dim=64, items_per_thread=2
        ),
        lambda: coop._block.make_radix_rank(
            types.int32,
            dim=64,
            items_per_thread=2,
            begin_bit=0,
            end_bit=8,
        ),
        lambda: coop._block.make_reduce(
            types.int32,
            dim=64,
            binary_op=_binary_op,
            items_per_thread=2,
        ),
        lambda: coop._block.make_sum(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_scan(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_exclusive_sum(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_inclusive_sum(types.int32, dim=64, items_per_thread=2),
        lambda: coop._block.make_exclusive_scan(
            types.int32,
            dim=64,
            scan_op="+",
            items_per_thread=2,
        ),
        lambda: coop._block.make_inclusive_scan(
            types.int32,
            dim=64,
            scan_op="+",
            items_per_thread=2,
        ),
        lambda: coop._block.make_adjacent_difference(
            types.int32,
            dim=64,
            items_per_thread=1,
            difference_op=_difference_op,
        ),
        lambda: coop._block.make_discontinuity(
            types.int32,
            dim=64,
            items_per_thread=1,
            flag_op=_flag_op,
        ),
        lambda: coop._block.make_shuffle(types.int32, dim=64, items_per_thread=1),
    ],
)
def test_make_block_factories_accept_dim_alias(factory):
    instance = factory()
    if isinstance(instance, Invocable):
        _assert_ltoir_files(instance)
    elif hasattr(instance, "init"):
        assert hasattr(instance, "composite")
    else:
        assert hasattr(instance, "decode")


def test_make_radix_rank_rejects_invalid_core_options():
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        coop._block.make_radix_rank(
            types.uint32,
            threads_per_block=32,
            items_per_thread=1.5,
            begin_bit=0,
            end_bit=4,
        )
    with pytest.raises(ValueError, match="descending must be a boolean"):
        coop._block.make_radix_rank(
            types.uint32,
            threads_per_block=32,
            items_per_thread=1,
            begin_bit=0,
            end_bit=4,
            descending=1,
        )
