# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import sys
import types

import pytest

import cuda.coop.cutlass as coop

CUDA_COOP_CUTLASS_DSL_SCOPE = "cuda.coop.cutlass._dsl"
CUDA_COOP_CUTLASS_DSL_BLOCK_SCOPE = f"{CUDA_COOP_CUTLASS_DSL_SCOPE}.block"
CUDA_COOP_CUTLASS_DSL_WARP_SCOPE = f"{CUDA_COOP_CUTLASS_DSL_SCOPE}.warp"
BLOCK_FACTORY_ADAPTERS = [
    "make_adjacent_difference",
    "make_discontinuity",
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_histogram",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_merge_sort_keys",
    "make_merge_sort_pairs",
    "make_radix_rank",
    "make_radix_sort_keys",
    "make_radix_sort_keys_descending",
    "make_radix_sort_pairs",
    "make_radix_sort_pairs_descending",
    "make_reduce",
    "make_run_length",
    "make_scan",
    "make_shuffle",
    "make_store",
    "make_sum",
    "make_topk_max_keys",
    "make_topk_max_pairs",
    "make_topk_min_keys",
    "make_topk_min_pairs",
]
WARP_FACTORY_ADAPTERS = [
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_max",
    "make_merge_sort_keys",
    "make_merge_sort_pairs",
    "make_min",
    "make_reduce",
    "make_store",
    "make_sum",
]


def _capture(**payload):
    return payload


def _native_capture(**payload):
    return payload


def _native_launch_capture(**payload):
    return payload


_native_capture._supports_native_thread_data = True
_native_launch_capture._supports_native_thread_data = True
_native_launch_capture._preserves_launch_metadata = True


class _TestInt32:
    name = "int32"
    kind = "i"
    itemsize = 4
    width = 32


@pytest.fixture(autouse=True)
def _restore_provider_registries():
    block_dispatch = importlib.import_module(
        f"{CUDA_COOP_CUTLASS_DSL_BLOCK_SCOPE}._dispatch"
    )
    warp_dispatch = importlib.import_module(
        f"{CUDA_COOP_CUTLASS_DSL_WARP_SCOPE}._dispatch"
    )

    block_impls = dict(block_dispatch._IMPLS)
    warp_impls = dict(warp_dispatch._IMPLS)
    try:
        yield
    finally:
        block_dispatch._IMPLS.clear()
        block_dispatch._IMPLS.update(block_impls)
        warp_dispatch._IMPLS.clear()
        warp_dispatch._IMPLS.update(warp_impls)


def test_cute_factories_are_private_scoped_adapters():
    for name in BLOCK_FACTORY_ADAPTERS:
        assert hasattr(coop._block, name), name
        assert name in coop._block.__all__
        assert not hasattr(coop, name)

    for name in WARP_FACTORY_ADAPTERS:
        assert hasattr(coop._warp, name), name
        assert name in coop._warp.__all__
        assert not hasattr(coop, name)


def test_block_factories_forward_to_scoped_primitives():
    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "reduce", _capture
    )
    reduce_op = coop._block.make_reduce(
        object,
        dim=64,
        binary_op="max",
    )
    assert reduce_op(7, marker="reduce") == {
        "value": 7,
        "args": (),
        "binary_op": "max",
        "launch_metadata": {"threads_per_block": 64},
        "marker": "reduce",
    }

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "sum", _capture
    )
    tuple_dim_sum = coop._block.make_sum(object, threads_per_block=(4, 8, 1))
    assert tuple_dim_sum(5) == {
        "value": 5,
        "args": (),
        "launch_metadata": {"block": (4, 8, 1)},
    }
    for algorithm in (
        "raking_commutative_only",
        "raking",
        "warp_reductions",
    ):
        assert (
            coop._block.make_sum(object, algorithm=algorithm)(5)["algorithm"]
            == algorithm
        )
        assert (
            coop._block.make_reduce(object, algorithm=algorithm)(5)["algorithm"]
            == algorithm
        )
    assert "algorithm" not in coop._block.make_sum(object)(5)
    assert "algorithm" not in coop._block.make_reduce(object)(5)
    valid_sum = coop._block.make_sum(object, num_valid=7)
    assert valid_sum(5)["num_valid"] == 7
    valid_items_sum = coop._block.make_sum(object, valid_items=6)
    assert valid_items_sum(5)["num_valid"] == 6
    with pytest.raises(TypeError, match="both num_valid and valid_items"):
        coop._block.make_sum(object, num_valid=7, valid_items=6)

    compatible_metadata_sum = coop._block.make_sum(
        object,
        dim=(4, 8, 1),
        launch_metadata={"threads_per_block": 32, "tag": "keep"},
    )
    assert compatible_metadata_sum(6)["launch_metadata"] == {
        "threads_per_block": 32,
        "tag": "keep",
    }
    with pytest.raises(TypeError, match="multiple thread-count keys"):
        coop._block.make_sum(
            object,
            dim=32,
            launch_metadata={
                "threads_per_block": 32,
                "block": (4, 8, 1),
            },
        )

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "exclusive_scan", _capture
    )
    exclusive = coop._block.make_exclusive_scan(
        object,
        threads_per_block=32,
        scan_op="max",
        initial_value=0,
    )
    assert exclusive(11) == {
        "value": 11,
        "args": (),
        "scan_op": "max",
        "initial_value": 0,
        "launch_metadata": {"threads_per_block": 32},
    }

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "exchange", _capture
    )
    values = object()
    output = object()
    exchange = coop._block.make_exchange(
        object,
        block_exchange_type=coop._block.BlockExchangeType.BlockedToStriped,
    )
    exchange_payload = exchange(values, output=output, marker="exchange")
    assert exchange_payload["value"] is values
    assert exchange_payload["output"] is output
    assert exchange_payload["mode"] == "blocked_to_striped"
    assert exchange_payload["marker"] == "exchange"

    selector_first_exchange = coop._block.make_exchange(
        coop._block.BlockExchangeType.BlockedToStriped
    )
    selector_first_payload = selector_first_exchange(values)
    assert selector_first_payload["mode"] == "blocked_to_striped"

    mode_call_exchange = coop._block.make_exchange(object)
    mode_call_payload = mode_call_exchange(values, mode="blocked_to_striped")
    assert mode_call_payload["mode"] == "blocked_to_striped"

    mode_bound_exchange = coop._block.make_exchange(object, mode="blocked_to_striped")
    mode_bound_payload = mode_bound_exchange(values)
    assert mode_bound_payload["mode"] == "blocked_to_striped"

    explicit_exchange = coop._block.make_exchange(
        object,
        block_exchange_type=coop._block.BlockExchangeType.BlockedToStriped,
    )
    with pytest.raises(TypeError, match="conflicting mode"):
        explicit_exchange(values, mode="striped_to_blocked")

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "topk_max_keys", _capture
    )
    keys = object()
    topk = coop._block.make_topk_max_keys(
        object,
        threads_per_block=128,
        num_valid=5,
        begin_bit=2,
        end_bit=9,
    )
    topk_payload = topk(keys, 3)
    assert topk_payload["keys"] is keys
    assert topk_payload["k"] == 3
    assert topk_payload["num_valid"] == 5
    assert topk_payload["begin_bit"] == 2
    assert topk_payload["end_bit"] == 9
    assert topk_payload["descending"] is True
    assert topk_payload["launch_metadata"] == {"threads_per_block": 128}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "radix_sort_pairs", _capture
    )
    values_out = object()
    pair_sort = coop._block.make_radix_sort_pairs(
        key_dtype=int,
        value_dtype=int,
        begin_bit=1,
        end_bit=8,
    )
    pair_sort_payload = pair_sort(keys, values_out)
    assert pair_sort_payload["keys"] is keys
    assert pair_sort_payload["values"] is values_out
    assert pair_sort_payload["begin_bit"] == 1
    assert pair_sort_payload["end_bit"] == 8


def test_block_factory_stateful_adapters():
    histogram = coop._block.make_histogram(
        object,
        int,
        bins=16,
        bins_per_thread=2,
        algorithm="sort",
    )
    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "histogram", _capture
    )
    samples = coop.ThreadData.from_values(1, 2, dtype=int)
    histogram_payload = histogram(samples)
    assert histogram_payload["samples"] is samples
    assert histogram_payload["bins"] == 16
    assert histogram_payload["bins_per_thread"] == 2
    assert histogram_payload["counter_dtype"] is int
    assert histogram_payload["algorithm"] == "sort"

    run_length = coop._block.make_run_length(
        object,
        runs_per_thread=2,
        decoded_items_per_thread=3,
    )
    lengths = coop.ThreadData.from_values(1, 2, dtype=int)
    parent = run_length(samples, lengths)
    assert isinstance(parent, coop._block.BlockRunLengthDecode)

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "run_length_decode", _capture
    )
    decode_payload = parent.decode(decoded_window_offset=4)
    assert decode_payload["run_values"] is samples
    assert decode_payload["run_lengths"] is lengths
    assert decode_payload["decoded_items_per_thread"] == 3
    assert decode_payload["decoded_window_offset"] == 4

    inferred_run_length = coop._block.make_run_length(
        object,
        decoded_items_per_thread=3,
    )
    inferred_parent = inferred_run_length(samples, lengths)
    assert isinstance(inferred_parent, coop._block.BlockRunLengthDecode)


def test_direct_block_primitives_accept_factory_launch_aliases():
    keys = object()
    values = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "topk_max_keys", _native_launch_capture
    )
    tuple_dim_payload = coop._block.topk_max_keys(
        keys,
        3,
        threads_per_block=(4, 8, 1),
    )
    assert tuple_dim_payload["keys"] is keys
    assert tuple_dim_payload["k"] == 3
    assert tuple_dim_payload["descending"] is True
    assert tuple_dim_payload["launch_metadata"] == {"block": (4, 8, 1)}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "topk_min_pairs", _native_launch_capture
    )
    merged_metadata_payload = coop._block.topk_min_pairs(
        keys,
        values,
        2,
        dim=32,
        launch_metadata={"tag": "keep"},
    )
    assert merged_metadata_payload["keys"] is keys
    assert merged_metadata_payload["values"] is values
    assert merged_metadata_payload["k"] == 2
    assert merged_metadata_payload["descending"] is False
    assert merged_metadata_payload["launch_metadata"] == {
        "tag": "keep",
        "threads_per_block": 32,
    }

    with pytest.raises(TypeError, match="conflicting threads_per_block and dim"):
        coop._block.topk_max_keys(keys, 3, threads_per_block=32, dim=64)

    with pytest.raises(
        TypeError,
        match="conflicting launch metadata and threads_per_block",
    ):
        coop._block.topk_max_keys(
            keys,
            3,
            threads_per_block=64,
            launch_metadata={"block": (4, 8, 1)},
        )


def _check_direct_block_exclusive_sum_accepts_launch_aliases():
    value = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "exclusive_sum", _native_launch_capture
    )
    exclusive_sum_payload = coop._block.exclusive_sum(value, dim=32)
    assert exclusive_sum_payload["value"] is value
    assert exclusive_sum_payload["launch_metadata"] == {"threads_per_block": 32}


def _check_direct_block_inclusive_scan_accepts_launch_aliases():
    value = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "inclusive_scan", _native_launch_capture
    )
    inclusive_scan_payload = coop._block.inclusive_scan(
        value,
        scan_op="max",
        threads_per_block=(4, 8, 1),
    )
    assert inclusive_scan_payload["value"] is value
    assert inclusive_scan_payload["scan_op"] == "max"
    assert inclusive_scan_payload["launch_metadata"] == {"block": (4, 8, 1)}


def _check_direct_block_reduce_accepts_launch_aliases():
    value = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "reduce", _native_launch_capture
    )
    reduce_payload = coop._block.reduce(
        value,
        binary_op="max",
        dim=32,
        launch_config={"tag": "keep"},
    )
    assert reduce_payload["value"] is value
    assert reduce_payload["binary_op"] == "max"
    assert reduce_payload["launch_config"] == {
        "tag": "keep",
        "threads_per_block": 32,
    }


def _check_direct_block_exchange_accepts_launch_aliases():
    value = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "exchange", _native_launch_capture
    )
    exchange_payload = coop._block.exchange_blocked_to_striped(
        value,
        threads_per_block=32,
    )
    assert exchange_payload["value"] is value
    assert exchange_payload["mode"] == "blocked_to_striped"
    assert exchange_payload["launch_metadata"] == {"threads_per_block": 32}


def _check_direct_block_shuffle_accepts_launch_aliases():
    value = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "shuffle", _native_launch_capture
    )
    shuffle_payload = coop._block.shuffle_rotate(
        value,
        distance=2,
        dim=(2, 16, 1),
    )
    assert shuffle_payload["value"] is value
    assert shuffle_payload["mode"] == "rotate"
    assert shuffle_payload["distance"] == 2
    assert shuffle_payload["launch_metadata"] == {"block": (2, 16, 1)}


def _check_direct_block_adjacent_difference_accepts_launch_aliases():
    value = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "adjacent_difference_subtract_right",
        _native_launch_capture,
    )
    difference_payload = coop._block.adjacent_difference(
        value,
        block_adjacent_difference_type=(
            coop._block.BlockAdjacentDifferenceType.SubtractRight
        ),
        dim=32,
    )
    assert difference_payload["value"] is value
    assert difference_payload["launch_metadata"] == {"threads_per_block": 32}
    assert "block_adjacent_difference_type" not in difference_payload
    assert "difference_op" not in difference_payload


def _check_direct_block_discontinuity_accepts_launch_aliases():
    value = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "discontinuity_flag_heads_and_tails",
        _native_launch_capture,
    )
    discontinuity_payload = coop._block.discontinuity(
        value,
        block_discontinuity_type=coop._block.BlockDiscontinuityType.HEADS_AND_TAILS,
        threads_per_block=32,
    )
    assert discontinuity_payload["value"] is value
    assert discontinuity_payload["launch_metadata"] == {"threads_per_block": 32}
    assert "block_discontinuity_type" not in discontinuity_payload
    assert "flag_op" not in discontinuity_payload


def _check_direct_block_histogram_accepts_launch_aliases():
    samples = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "histogram", _native_launch_capture
    )
    histogram_payload = coop._block.histogram(
        samples,
        bins=16,
        bins_per_thread=2,
        algorithm="sort",
        dim=32,
    )
    assert histogram_payload["samples"] is samples
    assert histogram_payload["bins"] == 16
    assert histogram_payload["bins_per_thread"] == 2
    assert histogram_payload["algorithm"] == "sort"
    assert histogram_payload["launch_metadata"] == {"threads_per_block": 32}


def _check_direct_block_run_length_accepts_launch_aliases():
    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "run_length_decode", _native_launch_capture
    )
    run_values = coop.ThreadData.from_values(1, 2, dtype=_TestInt32)
    run_lengths = coop.ThreadData.from_values(3, 4, dtype=_TestInt32)
    run_length = coop._block.run_length(
        run_values,
        run_lengths,
        decoded_items_per_thread=2,
        dim=32,
    )
    run_length_payload = run_length.decode(
        decoded_window_offset=1,
        launch_metadata={"tag": "keep"},
    )
    assert run_length_payload["run_values"] is run_values
    assert run_length_payload["run_lengths"] is run_lengths
    assert run_length_payload["decoded_items_per_thread"] == 2
    assert run_length_payload["decoded_window_offset"] == 1
    assert run_length_payload["launch_metadata"] == {
        "tag": "keep",
        "threads_per_block": 32,
    }


def _check_direct_block_temp_storage_accepts_launch_aliases():
    temp_storage = coop._block.TempStorage(4096)
    sum_values = coop.ThreadData.from_values(5, 6, dtype=_TestInt32)
    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "sum", _native_launch_capture
    )
    temp_storage_payload = coop._block.sum(
        sum_values,
        temp_storage=temp_storage,
        threads_per_block=32,
    )
    assert temp_storage_payload["value"] is sum_values
    assert temp_storage_payload["launch_metadata"] == {"threads_per_block": 32}
    assert "temp_storage" not in temp_storage_payload


def _check_direct_block_merge_sort_keys_accepts_launch_aliases():
    keys = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "merge_sort_keys", _native_launch_capture
    )
    merge_payload = coop._block.merge_sort_keys(keys, dim=32)
    assert merge_payload["keys"] is keys
    assert merge_payload["launch_metadata"] == {"threads_per_block": 32}


def _check_direct_block_radix_sort_pairs_accepts_launch_aliases():
    keys = object()
    values = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "radix_sort_pairs", _native_launch_capture
    )
    radix_payload = coop._block.radix_sort_pairs(
        keys,
        values,
        begin_bit=1,
        end_bit=8,
        threads_per_block=32,
    )
    assert radix_payload["keys"] is keys
    assert radix_payload["values"] is values
    assert radix_payload["begin_bit"] == 1
    assert radix_payload["end_bit"] == 8
    assert radix_payload["launch_metadata"] == {"threads_per_block": 32}


def _check_direct_block_radix_rank_accepts_launch_aliases():
    keys = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "radix_rank", _native_launch_capture
    )
    radix_rank_payload = coop._block.radix_rank(keys, dim=(4, 8, 1))
    assert radix_rank_payload["keys"] is keys
    assert radix_rank_payload["launch_metadata"] == {"block": (4, 8, 1)}


@pytest.mark.parametrize(
    "check",
    [
        _check_direct_block_exclusive_sum_accepts_launch_aliases,
        _check_direct_block_inclusive_scan_accepts_launch_aliases,
        _check_direct_block_reduce_accepts_launch_aliases,
        _check_direct_block_exchange_accepts_launch_aliases,
        _check_direct_block_shuffle_accepts_launch_aliases,
        _check_direct_block_adjacent_difference_accepts_launch_aliases,
        _check_direct_block_discontinuity_accepts_launch_aliases,
        _check_direct_block_histogram_accepts_launch_aliases,
        _check_direct_block_run_length_accepts_launch_aliases,
        _check_direct_block_temp_storage_accepts_launch_aliases,
        _check_direct_block_merge_sort_keys_accepts_launch_aliases,
        _check_direct_block_radix_sort_pairs_accepts_launch_aliases,
        _check_direct_block_radix_rank_accepts_launch_aliases,
    ],
    ids=lambda check: check.__name__.removeprefix("_check_"),
)
def test_direct_block_algorithm_families_accept_factory_launch_aliases(check):
    check()


def test_block_algorithm_family_factories_forward_to_scoped_primitives():
    value = object()
    keys = object()
    values = object()

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "exclusive_sum", _capture
    )
    exclusive_sum_payload = coop._block.make_exclusive_sum(
        object,
        threads_per_block=32,
    )(value)
    assert exclusive_sum_payload["value"] is value
    assert exclusive_sum_payload["launch_metadata"] == {"threads_per_block": 32}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "inclusive_sum", _capture
    )
    inclusive_sum_payload = coop._block.make_inclusive_sum(
        object,
        threads_per_block=32,
    )(value)
    assert inclusive_sum_payload["value"] is value
    assert inclusive_sum_payload["launch_metadata"] == {"threads_per_block": 32}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "exclusive_scan", _capture
    )
    scan_payload = coop._block.make_scan(
        object,
        threads_per_block=32,
        scan_op="max",
        initial_value=0,
    )(value)
    assert scan_payload["value"] is value
    assert scan_payload["scan_op"] == "max"
    assert scan_payload["initial_value"] == 0
    assert scan_payload["launch_metadata"] == {"threads_per_block": 32}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "inclusive_scan", _capture
    )
    inclusive_scan_payload = coop._block.make_inclusive_scan(
        object,
        threads_per_block=32,
        scan_op="min",
    )(value)
    assert inclusive_scan_payload["value"] is value
    assert inclusive_scan_payload["scan_op"] == "min"
    assert inclusive_scan_payload["launch_metadata"] == {"threads_per_block": 32}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "adjacent_difference_subtract_left", _capture
    )
    adjacent_payload = coop._block.make_adjacent_difference(
        object,
        threads_per_block=32,
    )(value)
    assert adjacent_payload["value"] is value
    assert adjacent_payload["launch_metadata"] == {"threads_per_block": 32}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "discontinuity_flag_heads", _capture
    )
    discontinuity_payload = coop._block.make_discontinuity(
        object,
        threads_per_block=32,
    )(value)
    assert discontinuity_payload["value"] is value
    assert discontinuity_payload["launch_metadata"] == {"threads_per_block": 32}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "merge_sort_keys", _capture
    )
    merge_payload = coop._block.make_merge_sort_keys(
        object,
        threads_per_block=32,
        descending=True,
    )(keys)
    assert merge_payload["keys"] is keys
    assert merge_payload["descending"] is True
    assert merge_payload["launch_metadata"] == {"threads_per_block": 32}

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "radix_sort_keys", _capture
    )
    radix_keys_payload = coop._block.make_radix_sort_keys(
        object,
        threads_per_block=32,
        begin_bit=1,
        end_bit=8,
    )(keys)
    assert radix_keys_payload["keys"] is keys
    assert radix_keys_payload["begin_bit"] == 1
    assert radix_keys_payload["end_bit"] == 8
    assert radix_keys_payload["descending"] is False

    radix_keys_desc_payload = coop._block.make_radix_sort_keys_descending(
        object,
        threads_per_block=32,
        begin_bit=2,
        end_bit=9,
    )(keys)
    assert radix_keys_desc_payload["keys"] is keys
    assert radix_keys_desc_payload["begin_bit"] == 2
    assert radix_keys_desc_payload["end_bit"] == 9
    assert radix_keys_desc_payload["descending"] is True

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "radix_sort_pairs", _capture
    )
    radix_pairs_payload = coop._block.make_radix_sort_pairs_descending(
        key_dtype=int,
        value_dtype=int,
        threads_per_block=32,
        begin_bit=3,
        end_bit=10,
    )(keys, values)
    assert radix_pairs_payload["keys"] is keys
    assert radix_pairs_payload["values"] is values
    assert radix_pairs_payload["begin_bit"] == 3
    assert radix_pairs_payload["end_bit"] == 10
    assert radix_pairs_payload["descending"] is True

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "radix_rank", _capture
    )
    radix_rank_payload = coop._block.make_radix_rank(
        object,
        threads_per_block=32,
        begin_bit=4,
        end_bit=11,
    )(keys)
    assert radix_rank_payload["keys"] is keys
    assert radix_rank_payload["begin_bit"] == 4
    assert radix_rank_payload["end_bit"] == 11

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "topk_min_keys", _capture
    )
    topk_min_keys_payload = coop._block.make_topk_min_keys(
        object,
        threads_per_block=32,
        num_valid=7,
    )(keys, 2)
    assert topk_min_keys_payload["keys"] is keys
    assert topk_min_keys_payload["k"] == 2
    assert topk_min_keys_payload["num_valid"] == 7
    assert topk_min_keys_payload["descending"] is False

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "topk_max_pairs", _capture
    )
    topk_max_pairs_payload = coop._block.make_topk_max_pairs(
        key_dtype=int,
        value_dtype=int,
        threads_per_block=32,
    )(keys, values, 3)
    assert topk_max_pairs_payload["keys"] is keys
    assert topk_max_pairs_payload["values"] is values
    assert topk_max_pairs_payload["k"] == 3
    assert topk_max_pairs_payload["descending"] is True

    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "topk_min_pairs", _capture
    )
    topk_min_pairs_payload = coop._block.make_topk_min_pairs(
        key_dtype=int,
        value_dtype=int,
        threads_per_block=32,
    )(keys, values, 4)
    assert topk_min_pairs_payload["keys"] is keys
    assert topk_min_pairs_payload["values"] is values
    assert topk_min_pairs_payload["k"] == 4
    assert topk_min_pairs_payload["descending"] is False


def test_block_load_store_factories_accept_threads_per_block(monkeypatch):
    cutlass_mod = types.ModuleType("cutlass")
    cute_mod = types.ModuleType("cutlass.cute")
    cute_mod.arch = types.SimpleNamespace(
        thread_idx=lambda: (0, 0, 0),
        block_dim=lambda: (32, 1, 1),
    )
    cutlass_mod.cute = cute_mod
    monkeypatch.setitem(sys.modules, "cutlass", cutlass_mod)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute_mod)

    load = coop._block.make_load(
        int,
        threads_per_block=(32, 1, 1),
        items_per_thread=2,
    )
    loaded = load([10, 20, 30, 40])
    assert list(loaded) == [10, 20]

    store = coop._block.make_store(
        int,
        dim=(32, 1, 1),
        items_per_thread=2,
    )
    destination = [0, 0, 0, 0]
    store(destination, loaded)
    assert destination == [10, 20, 0, 0]

    direct_loaded = coop._block.load(
        [30, 40, 50, 60],
        items_per_thread=2,
        threads_per_block=(32, 1, 1),
    )
    assert list(direct_loaded) == [30, 40]

    direct_destination = [0, 0, 0, 0]
    coop._block.store(
        direct_destination,
        direct_loaded,
        dim=(32, 1, 1),
    )
    assert direct_destination == [30, 40, 0, 0]

    with pytest.raises(TypeError, match="conflicting threads_per_block and dim"):
        coop._block.load(
            [1, 2],
            items_per_thread=1,
            threads_per_block=32,
            dim=64,
        )


def test_block_shuffle_factory_mode_does_not_conflict_with_default_selector():
    getattr(coop._block, "_backend", coop._block)._api.register_provider_impl(
        "shuffle", _capture
    )

    shuffle = coop._block.make_shuffle(object)
    assert shuffle(7)["mode"] == "up"
    assert shuffle(7, mode="down")["mode"] == "down"
    assert (
        shuffle(7, block_shuffle_type=coop._block.BlockShuffleType.Down)["mode"]
        == "down"
    )

    mode_bound_shuffle = coop._block.make_shuffle(object, mode="rotate", distance=3)
    mode_bound_payload = mode_bound_shuffle(7)
    assert mode_bound_payload["mode"] == "rotate"
    assert mode_bound_payload["distance"] == 3

    explicit_shuffle = coop._block.make_shuffle(
        object,
        block_shuffle_type=coop._block.BlockShuffleType.Up,
    )
    with pytest.raises(TypeError, match="conflicting mode"):
        explicit_shuffle(7, mode="down")


def test_warp_factories_forward_to_scoped_primitives():
    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "sum", _capture
    )
    sum_op = coop._warp.make_sum(object, threads_in_warp=16)
    assert sum_op(7) == {
        "value": 7,
        "args": (),
        "threads_in_warp": 16,
    }
    valid_sum_op = coop._warp.make_sum(
        object,
        valid_items=7,
        threads_in_warp=16,
    )
    assert valid_sum_op(7)["valid_items"] == 7

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "reduce", _capture
    )
    reduce_op = coop._warp.make_reduce(
        object,
        binary_op="min",
        threads_in_warp=8,
    )
    assert reduce_op(11) == {
        "value": 11,
        "args": (),
        "binary_op": "min",
        "threads_in_warp": 8,
    }

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "exchange", _native_capture
    )
    values = coop.ThreadData.from_values(1, 2, dtype=int)
    exchange = coop._warp.make_exchange(
        object,
        threads_in_warp=16,
        warp_exchange_type=coop._warp.WarpExchangeType.BlockedToStriped,
    )
    exchange_payload = exchange(values)
    assert exchange_payload["value"] is values
    assert exchange_payload["mode"] == "blocked_to_striped"
    assert exchange_payload["threads_in_warp"] == 16

    selector_first_exchange = coop._warp.make_exchange(
        coop._warp.WarpExchangeType.BlockedToStriped,
        threads_in_warp=16,
    )
    selector_first_payload = selector_first_exchange(values)
    assert selector_first_payload["mode"] == "blocked_to_striped"
    assert selector_first_payload["threads_in_warp"] == 16

    mode_call_exchange = coop._warp.make_exchange(object, threads_in_warp=16)
    mode_call_payload = mode_call_exchange(values, mode="blocked_to_striped")
    assert mode_call_payload["mode"] == "blocked_to_striped"
    assert mode_call_payload["threads_in_warp"] == 16

    mode_bound_exchange = coop._warp.make_exchange(
        object,
        threads_in_warp=16,
        mode="blocked_to_striped",
    )
    mode_bound_payload = mode_bound_exchange(values)
    assert mode_bound_payload["mode"] == "blocked_to_striped"
    assert mode_bound_payload["threads_in_warp"] == 16

    explicit_exchange = coop._warp.make_exchange(
        object,
        threads_in_warp=16,
        warp_exchange_type=coop._warp.WarpExchangeType.BlockedToStriped,
    )
    with pytest.raises(TypeError, match="conflicting mode"):
        explicit_exchange(values, mode="striped_to_blocked")

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "merge_sort_keys", _native_capture
    )
    sort = coop._warp.make_merge_sort_keys(
        object,
        items_per_thread=2,
        compare_op=">",
        threads_in_warp=16,
        valid_items=27,
        oob_default=-999,
    )
    sort_payload = sort(values)
    assert sort_payload["keys"] is values
    assert sort_payload["compare_op"] == ">"
    assert sort_payload["threads_in_warp"] == 16
    assert sort_payload["valid_items"] == 27
    assert sort_payload["oob_default"] == -999
    override_payload = sort(values, valid_items=29, oob_default=999)
    assert override_payload["valid_items"] == 29
    assert override_payload["oob_default"] == 999

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "merge_sort_pairs", _native_capture
    )
    pair_sort = coop._warp.make_merge_sort_pairs(
        key_dtype=int,
        value_dtype=int,
        items_per_thread=2,
        threads_in_warp=16,
        valid_items=27,
        oob_default=-999,
    )
    pair_values = coop.ThreadData.from_values(3, 4, dtype=int)
    pair_sort_payload = pair_sort(values, pair_values)
    assert pair_sort_payload["keys"] is values
    assert pair_sort_payload["values"] is pair_values
    assert pair_sort_payload["threads_in_warp"] == 16
    assert pair_sort_payload["valid_items"] == 27
    assert pair_sort_payload["oob_default"] == -999


def test_warp_scan_and_reduction_factories_forward_to_scoped_primitives():
    value = object()

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "max", _capture
    )
    max_payload = coop._warp.make_max(object, threads_in_warp=16)(value)
    assert max_payload["value"] is value
    assert max_payload["threads_in_warp"] == 16

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "min", _capture
    )
    min_payload = coop._warp.make_min(object, threads_in_warp=16)(value)
    assert min_payload["value"] is value
    assert min_payload["threads_in_warp"] == 16

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "exclusive_sum", _capture
    )
    exclusive_sum_payload = coop._warp.make_exclusive_sum(
        object,
        threads_in_warp=16,
    )(value)
    assert exclusive_sum_payload["value"] is value
    assert exclusive_sum_payload["threads_in_warp"] == 16

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "inclusive_sum", _capture
    )
    inclusive_sum_payload = coop._warp.make_inclusive_sum(
        object,
        threads_in_warp=16,
    )(value)
    assert inclusive_sum_payload["value"] is value
    assert inclusive_sum_payload["threads_in_warp"] == 16

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "exclusive_scan", _capture
    )
    exclusive_scan_payload = coop._warp.make_exclusive_scan(
        object,
        scan_op="max",
        initial_value=0,
        threads_in_warp=16,
    )(value)
    assert exclusive_scan_payload["value"] is value
    assert exclusive_scan_payload["scan_op"] == "max"
    assert exclusive_scan_payload["initial_value"] == 0
    assert exclusive_scan_payload["threads_in_warp"] == 16

    getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl(
        "inclusive_scan", _capture
    )
    inclusive_scan_payload = coop._warp.make_inclusive_scan(
        object,
        scan_op="min",
        threads_in_warp=16,
    )(value)
    assert inclusive_scan_payload["value"] is value
    assert inclusive_scan_payload["scan_op"] == "min"
    assert inclusive_scan_payload["threads_in_warp"] == 16


def test_factory_duplicate_bound_kwargs_raise():
    sum_op = coop._warp.make_sum(object, threads_in_warp=16)
    with pytest.raises(TypeError, match="duplicate keyword argument"):
        sum_op(7, threads_in_warp=32)

    block_sum = coop._block.make_sum(object, threads_per_block=32)
    with pytest.raises(TypeError, match="duplicate launch metadata aliases"):
        block_sum(7, launch_config={"block": (32, 1, 1)})


def test_unsupported_compatibility_factory_arguments_are_explicit():
    with pytest.raises(NotImplementedError, match="methods"):
        coop._block.make_load(object, methods=object())

    with pytest.raises(NotImplementedError, match="methods"):
        coop._block.make_store(object, methods=object())

    with pytest.raises(NotImplementedError, match="compare_op"):
        coop._block.make_merge_sort_keys(object, compare_op=lambda lhs, rhs: lhs < rhs)
