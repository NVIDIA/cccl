# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.oracles import (
    assert_pairs_still_match_input as _assert_pairs_still_match_input,
)
from ..support.oracles import (
    gather_cpu_tensor as _gather_cpu_tensor,
)
from ..support.runtime import (
    LAUNCH_CASES as _LAUNCH_CASES,
)
from ..support.runtime import (
    LAUNCH_DESCENDING_CASES as _LAUNCH_DESCENDING_CASES,
)
from ..support.runtime import (
    RADIX_TEMP_STORAGE as _RADIX_TEMP_STORAGE,
)
from ..support.runtime import (
    TOPK_SCORE_K as _TOPK_SCORE_K,
)
from ..support.runtime import (
    TOPK_TEMP_STORAGE as _TOPK_TEMP_STORAGE,
)
from ..support.runtime import (
    Int32,
    coop,
    cute,
    cutlass,
    from_dlpack,
    runtime_pytestmark,
    torch,
)

pytestmark = runtime_pytestmark


@cute.kernel
def _radix_sort_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    begin_bit = cutlass.const_expr(4)
    end_bit = cutlass.const_expr(12)
    descending = cutlass.const_expr(True)
    sorted_key, sorted_val = coop._block.radix_sort_pairs(
        key,
        val,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    sorted_key_only = coop._block.radix_sort_keys(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val
        keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _radix_sort_temp_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    begin_bit = cutlass.const_expr(4)
    end_bit = cutlass.const_expr(12)
    descending = cutlass.const_expr(True)
    sorted_key, sorted_val = coop._block.radix_sort_pairs(
        key,
        val,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=_RADIX_TEMP_STORAGE,
    )
    sorted_key_only = coop._block.radix_sort_keys(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=_RADIX_TEMP_STORAGE,
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val
        keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _radix_sort_dynamic_bits_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    begin_bit: cutlass.Int32,
    end_bit: cutlass.Int32,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    descending = cutlass.const_expr(True)
    sorted_key, sorted_val = coop._block.radix_sort_pairs(
        key,
        val,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    sorted_key_only = coop._block.radix_sort_keys(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val
        keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _radix_sort_64bit_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    begin_bit = cutlass.const_expr(33)
    end_bit = cutlass.const_expr(45)
    descending = cutlass.const_expr(False)
    sorted_key, sorted_val = coop._block.radix_sort_pairs(
        key,
        val,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    sorted_key_only = coop._block.radix_sort_keys(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val
        keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _radix_sort_64bit_temp_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    begin_bit = cutlass.const_expr(33)
    end_bit = cutlass.const_expr(45)
    descending = cutlass.const_expr(False)
    sorted_key, sorted_val = coop._block.radix_sort_pairs(
        key,
        val,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=_RADIX_TEMP_STORAGE,
    )
    sorted_key_only = coop._block.radix_sort_keys(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=_RADIX_TEMP_STORAGE,
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val
        keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _radix_rank_32bit_kernel(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    rank = coop._block.radix_rank(
        key,
        begin_bit=cutlass.const_expr(28),
        end_bit=cutlass.const_expr(32),
        descending=cutlass.const_expr(False),
    )
    if tidx < block_x:
        rank_out[tidx] = rank


@cute.kernel
def _radix_rank_64bit_kernel(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    rank = coop._block.radix_rank(
        key,
        begin_bit=cutlass.const_expr(60),
        end_bit=cutlass.const_expr(64),
        descending=cutlass.const_expr(True),
    )
    if tidx < block_x:
        rank_out[tidx] = rank


@cute.kernel
def _radix_rank_prefix_kernel(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(4)
    descending = cutlass.const_expr(False)
    exclusive_digit_prefix = coop.ThreadData(1, dtype=Int32)
    rank = coop._block.radix_rank(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )
    if tidx < block_x:
        rank_out[tidx] = rank
        prefix_out[tidx] = exclusive_digit_prefix[0]


@cute.kernel
def _radix_rank_prefix_temp_kernel(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(4)
    descending = cutlass.const_expr(True)
    exclusive_digit_prefix = coop.ThreadData(1, dtype=Int32)
    rank = coop._block.radix_rank(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
        temp_storage=_RADIX_TEMP_STORAGE,
    )
    if tidx < block_x:
        rank_out[tidx] = rank
        prefix_out[tidx] = exclusive_digit_prefix[0]


@cute.kernel
def _radix_rank_prefix_multi_track_kernel(
    keys_in: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(6)
    descending = cutlass.const_expr(False)
    bins_per_thread = cutlass.const_expr(4)
    exclusive_digit_prefix = coop.ThreadData(bins_per_thread, dtype=Int32)
    coop._block.radix_rank(
        key,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )
    if tidx < block_x:
        base = tidx * bins_per_thread
        prefix_out[base + 0] = exclusive_digit_prefix[0]
        prefix_out[base + 1] = exclusive_digit_prefix[1]
        prefix_out[base + 2] = exclusive_digit_prefix[2]
        prefix_out[base + 3] = exclusive_digit_prefix[3]


@cute.kernel
def _radix_rank_multi_item_payload_kernel(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(4)
    descending = cutlass.const_expr(False)
    exclusive_digit_prefix = coop.ThreadData(1, dtype=Int32)
    if use_register_payload:
        fragment = cute.make_rmem_tensor((1, 3), Int32)
        fragment[0] = items[0]
        fragment[1] = items[1]
        fragment[2] = items[2]
        ranks = coop._block.radix_rank(
            fragment,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            exclusive_digit_prefix=exclusive_digit_prefix,
        )
        coop._block.store(rank_out, ranks)
        if tidx < block_x:
            prefix_out[tidx] = exclusive_digit_prefix[0]
    else:
        ranks = coop._block.radix_rank(
            items,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            exclusive_digit_prefix=exclusive_digit_prefix,
        )
        coop._block.store(rank_out, ranks)
        if tidx < block_x:
            prefix_out[tidx] = exclusive_digit_prefix[0]


@cute.kernel
def _radix_rank_multi_item_payload_temp_kernel(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    items = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(4)
    descending = cutlass.const_expr(True)
    exclusive_digit_prefix = coop.ThreadData(1, dtype=Int32)
    if use_register_payload:
        fragment = cute.make_rmem_tensor((1, 3), Int32)
        fragment[0] = items[0]
        fragment[1] = items[1]
        fragment[2] = items[2]
        ranks = coop._block.radix_rank(
            fragment.load(),
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            exclusive_digit_prefix=exclusive_digit_prefix,
            temp_storage=_RADIX_TEMP_STORAGE,
        )
        coop._block.store(rank_out, ranks)
        if tidx < block_x:
            prefix_out[tidx] = exclusive_digit_prefix[0]
    else:
        ranks = coop._block.radix_rank(
            items,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            exclusive_digit_prefix=exclusive_digit_prefix,
            temp_storage=_RADIX_TEMP_STORAGE,
        )
        coop._block.store(rank_out, ranks)
        if tidx < block_x:
            prefix_out[tidx] = exclusive_digit_prefix[0]


@cute.kernel
def _radix_sort_multi_item_payload_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    use_register_payload: cutlass.Constexpr,
):
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(4)
    descending = cutlass.const_expr(False)
    if use_register_payload:
        key_fragment = cute.make_rmem_tensor((1, 3), Int32)
        value_fragment = cute.make_rmem_tensor((1, 3), Int32)
        for item_idx in cutlass.range_constexpr(3):
            key_fragment[item_idx] = keys[item_idx]
            value_fragment[item_idx] = vals[item_idx]
        sorted_keys, sorted_vals = coop._block.radix_sort_pairs(
            key_fragment,
            value_fragment.load(),
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
        )
        sorted_keys_only = coop._block.radix_sort_keys(
            key_fragment.load(),
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
        )
        coop._block.store(keys_out, sorted_keys)
        coop._block.store(vals_out, sorted_vals)
        coop._block.store(keys_only_out, sorted_keys_only)
    else:
        sorted_keys, sorted_vals = coop._block.radix_sort_pairs(
            keys,
            vals,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
        )
        sorted_keys_only = coop._block.radix_sort_keys(
            keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
        )
        coop._block.store(keys_out, sorted_keys)
        coop._block.store(vals_out, sorted_vals)
        coop._block.store(keys_only_out, sorted_keys_only)


@cute.kernel
def _radix_sort_multi_item_payload_temp_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    use_register_payload: cutlass.Constexpr,
):
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(4)
    descending = cutlass.const_expr(True)
    if use_register_payload:
        key_fragment = cute.make_rmem_tensor((1, 3), Int32)
        value_fragment = cute.make_rmem_tensor((1, 3), Int32)
        for item_idx in cutlass.range_constexpr(3):
            key_fragment[item_idx] = keys[item_idx]
            value_fragment[item_idx] = vals[item_idx]
        sorted_keys, sorted_vals = coop._block.radix_sort_pairs(
            key_fragment,
            value_fragment.load(),
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            temp_storage=_RADIX_TEMP_STORAGE,
        )
        sorted_keys_only = coop._block.radix_sort_keys(
            key_fragment.load(),
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            temp_storage=_RADIX_TEMP_STORAGE,
        )
        coop._block.store(keys_out, sorted_keys)
        coop._block.store(vals_out, sorted_vals)
        coop._block.store(keys_only_out, sorted_keys_only)
    else:
        sorted_keys, sorted_vals = coop._block.radix_sort_pairs(
            keys,
            vals,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            temp_storage=_RADIX_TEMP_STORAGE,
        )
        sorted_keys_only = coop._block.radix_sort_keys(
            keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            temp_storage=_RADIX_TEMP_STORAGE,
        )
        coop._block.store(keys_out, sorted_keys)
        coop._block.store(vals_out, sorted_vals)
        coop._block.store(keys_only_out, sorted_keys_only)


@cute.kernel
def _group_radix_sort_pairs_scalar_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    group = coop.this_block()
    sorted_key, sorted_val = coop.radix_sort_pairs(
        group,
        keys_in[tidx],
        vals_in[tidx],
        begin_bit=cutlass.const_expr(4),
        end_bit=cutlass.const_expr(12),
        descending=cutlass.const_expr(True),
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val


@cute.kernel
def _group_radix_sort_pairs_register_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
):
    group = coop.this_block()
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    key_fragment = cute.make_rmem_tensor((1, 3), Int32)
    value_fragment = cute.make_rmem_tensor((1, 3), Int32)
    for item_idx in cutlass.range_constexpr(3):
        key_fragment[item_idx] = keys[item_idx]
        value_fragment[item_idx] = vals[item_idx]

    sorted_keys, sorted_vals = coop.radix_sort_pairs(
        group,
        key_fragment,
        value_fragment.load(),
        begin_bit=cutlass.const_expr(0),
        end_bit=cutlass.const_expr(4),
    )
    coop.store(group, keys_out, sorted_keys)
    coop.store(group, vals_out, sorted_vals)


@cute.kernel
def _topk_score_window_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    max_keys_out: cute.Tensor,
    min_pair_keys_out: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    k = cutlass.const_expr(_TOPK_SCORE_K)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(8)
    top_key = coop._block.topk_max_keys(
        key,
        k,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=_TOPK_TEMP_STORAGE,
    )
    pair_key, pair_val = coop._block.topk_min_pairs(
        key,
        val,
        k,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=_TOPK_TEMP_STORAGE,
    )
    if tidx < block_x:
        max_keys_out[tidx] = top_key
        min_pair_keys_out[tidx] = pair_key
        min_pair_vals_out[tidx] = pair_val


@cute.kernel
def _topk_pair_values_only_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    k = cutlass.const_expr(_TOPK_SCORE_K)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(8)
    _, pair_val = coop._block.topk_min_pairs(
        key,
        val,
        k,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=_TOPK_TEMP_STORAGE,
    )
    if tidx < block_x:
        min_pair_vals_out[tidx] = pair_val


@cute.kernel
def _topk_float_partial_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    max_keys_out: cute.Tensor,
    min_pair_keys_out: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    valid_items: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    k = cutlass.const_expr(7)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(32)
    top_key = coop._block.topk_max_keys(
        key,
        k,
        num_valid=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=_TOPK_TEMP_STORAGE,
    )
    pair_key, pair_val = coop._block.topk_min_pairs(
        key,
        val,
        k,
        num_valid=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=_TOPK_TEMP_STORAGE,
    )
    max_keys_out[tidx] = top_key
    min_pair_keys_out[tidx] = pair_key
    min_pair_vals_out[tidx] = pair_val


@cute.kernel
def _topk_multi_item_payload_score_window_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    max_keys_out: cute.Tensor,
    min_pair_keys_out: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    use_register_payload: cutlass.Constexpr,
):
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    k = cutlass.const_expr(_TOPK_SCORE_K)
    begin_bit = cutlass.const_expr(0)
    end_bit = cutlass.const_expr(8)
    if use_register_payload:
        key_fragment = cute.make_rmem_tensor((1, 3), Int32)
        value_fragment = cute.make_rmem_tensor((1, 3), Int32)
        for item_idx in cutlass.range_constexpr(3):
            key_fragment[item_idx] = keys[item_idx]
            value_fragment[item_idx] = vals[item_idx]
        top_keys = coop._block.topk_max_keys(
            key_fragment,
            k,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        pair_keys, pair_vals = coop._block.topk_min_pairs(
            key_fragment.load(),
            value_fragment,
            k,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        coop._block.store(max_keys_out, top_keys)
        coop._block.store(min_pair_keys_out, pair_keys)
        coop._block.store(min_pair_vals_out, pair_vals)
    else:
        top_keys = coop._block.topk_max_keys(
            keys,
            k,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        pair_keys, pair_vals = coop._block.topk_min_pairs(
            keys,
            vals,
            k,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        coop._block.store(max_keys_out, top_keys)
        coop._block.store(min_pair_keys_out, pair_keys)
        coop._block.store(min_pair_vals_out, pair_vals)


@cute.jit
def _run_radix_sort(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_sort_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_sort_temp(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_sort_temp_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_sort_dynamic_bits(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    begin_bit: cutlass.Int32,
    end_bit: cutlass.Int32,
    block_x: cutlass.Constexpr,
):
    _radix_sort_dynamic_bits_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        begin_bit,
        end_bit,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_sort_64bit(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_sort_64bit_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_sort_64bit_temp(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_sort_64bit_temp_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_rank_32bit(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_rank_32bit_kernel(keys_in, rank_out, block_x).launch(
        grid=(1, 1, 1), block=(block_x, 1, 1)
    )


@cute.jit
def _run_radix_rank_64bit(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_rank_64bit_kernel(keys_in, rank_out, block_x).launch(
        grid=(1, 1, 1), block=(block_x, 1, 1)
    )


@cute.jit
def _run_radix_rank_prefix(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_rank_prefix_kernel(
        keys_in,
        rank_out,
        prefix_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_rank_prefix_temp(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_rank_prefix_temp_kernel(
        keys_in,
        rank_out,
        prefix_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_rank_prefix_multi_track(
    keys_in: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _radix_rank_prefix_multi_track_kernel(
        keys_in,
        prefix_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_rank_multi_item_payload(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    _radix_rank_multi_item_payload_kernel(
        keys_in,
        rank_out,
        prefix_out,
        block_x,
        use_register_payload,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_rank_multi_item_payload_temp(
    keys_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    _radix_rank_multi_item_payload_temp_kernel(
        keys_in,
        rank_out,
        prefix_out,
        block_x,
        use_register_payload,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_sort_multi_item_payload(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    _radix_sort_multi_item_payload_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        use_register_payload,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_radix_sort_multi_item_payload_temp(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    _radix_sort_multi_item_payload_temp_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        use_register_payload,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_group_radix_sort_pairs_scalar(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _group_radix_sort_pairs_scalar_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_group_radix_sort_pairs_register(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _group_radix_sort_pairs_register_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_topk_score_window(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    max_keys_out: cute.Tensor,
    min_pair_keys_out: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _topk_score_window_kernel(
        keys_in,
        vals_in,
        max_keys_out,
        min_pair_keys_out,
        min_pair_vals_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_topk_pair_values_only(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _topk_pair_values_only_kernel(
        keys_in,
        vals_in,
        min_pair_vals_out,
        block_x,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_topk_float_partial(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    max_keys_out: cute.Tensor,
    min_pair_keys_out: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    valid_items: cutlass.Constexpr,
):
    _topk_float_partial_kernel(
        keys_in,
        vals_in,
        max_keys_out,
        min_pair_keys_out,
        min_pair_vals_out,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_topk_multi_item_payload_score_window(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    max_keys_out: cute.Tensor,
    min_pair_keys_out: cute.Tensor,
    min_pair_vals_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    _topk_multi_item_payload_score_window_kernel(
        keys_in,
        vals_in,
        max_keys_out,
        min_pair_keys_out,
        min_pair_vals_out,
        use_register_payload,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


def _expected_radix_order(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    width_bits = 64 if keys.dtype in {torch.int64, torch.uint64} else 32
    full_mask = (1 << width_bits) - 1
    raw_unsigned = [(int(key) & full_mask) for key in keys.tolist()]
    sign_flip = (
        0 if keys.dtype in {torch.uint32, torch.uint64} else 1 << (width_bits - 1)
    )
    ordered = [value ^ sign_flip for value in raw_unsigned]
    mask = (1 << (end_bit - begin_bit)) - 1
    key_sig = [int((value >> begin_bit) & mask) for value in ordered]
    idx = list(range(len(key_sig)))
    if descending:
        idx = sorted(idx, key=lambda i: (-int(key_sig[i]), i))
    else:
        idx = sorted(idx, key=lambda i: (int(key_sig[i]), i))
    return _gather_cpu_tensor(keys, idx), _gather_cpu_tensor(values, idx)


def _assert_topk_keys_unordered(
    actual_keys: torch.Tensor,
    expected_keys: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        torch.sort(actual_keys.cpu()).values,
        torch.sort(expected_keys.cpu()).values,
        atol=0,
        rtol=0,
    )


def _assert_topk_pairs_unordered(
    actual_keys: torch.Tensor,
    actual_vals: torch.Tensor,
    expected_keys: torch.Tensor,
    expected_vals: torch.Tensor,
) -> None:
    actual_pairs = sorted(
        zip(actual_keys.cpu().tolist(), actual_vals.cpu().tolist(), strict=True)
    )
    expected_pairs = sorted(
        zip(expected_keys.cpu().tolist(), expected_vals.cpu().tolist(), strict=True)
    )
    assert actual_pairs == expected_pairs


@pytest.mark.evidence_for(
    "group.radix_sort_pairs",
    backend="cutlass",
    evidence="runtime",
)
def test_group_radix_sort_pairs_scalar_matches_independent_oracle():
    cutlass.cuda.initialize_cuda_context()

    block_x = 32
    keys_host = torch.tensor(
        [((idx * 17 + (idx % 5) * 7) % 97) - 48 for idx in range(block_x)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32) * 11 + 3
    keys_out = torch.zeros_like(keys_host, device="cuda")
    vals_out = torch.zeros_like(vals_host, device="cuda")

    _run_group_radix_sort_pairs_scalar(
        from_dlpack(keys_host.cuda()),
        from_dlpack(vals_host.cuda()),
        from_dlpack(keys_out),
        from_dlpack(vals_out),
        block_x,
    )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=4,
        end_bit=12,
        descending=True,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.evidence_for(
    "group.radix_sort_pairs",
    backend="cutlass",
    evidence="runtime",
)
def test_group_radix_sort_pairs_registers_match_independent_oracle():
    cutlass.cuda.initialize_cuda_context()

    block_x = 32
    item_count = block_x * 3
    keys_host = torch.tensor(
        [((idx * 13 + (idx % 7) * 5) % 64) - 32 for idx in range(item_count)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(item_count, dtype=torch.int32) * 7 + 1
    keys_out = torch.zeros_like(keys_host, device="cuda")
    vals_out = torch.zeros_like(vals_host, device="cuda")

    _run_group_radix_sort_pairs_register(
        from_dlpack(keys_host.cuda()),
        from_dlpack(vals_host.cuda()),
        from_dlpack(keys_out),
        from_dlpack(vals_out),
        block_x,
    )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=0,
        end_bit=4,
        descending=False,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


def _expected_radix_ranks(
    keys: torch.Tensor,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
) -> torch.Tensor:
    width_bits = 64 if keys.dtype in {torch.int64, torch.uint64} else 32
    full_mask = (1 << width_bits) - 1
    raw_unsigned = [(int(key) & full_mask) for key in keys.tolist()]
    sign_flip = (
        0 if keys.dtype in {torch.uint32, torch.uint64} else 1 << (width_bits - 1)
    )
    ordered = [value ^ sign_flip for value in raw_unsigned]
    mask = (1 << (end_bit - begin_bit)) - 1
    digits = [int((value >> begin_bit) & mask) for value in ordered]
    ranks = torch.empty((len(digits),), dtype=torch.int32)
    for idx, digit in enumerate(digits):
        rank = 0
        for peer_idx, peer_digit in enumerate(digits):
            if descending:
                before = peer_digit > digit
            else:
                before = peer_digit < digit
            if before or (peer_digit == digit and peer_idx < idx):
                rank += 1
        ranks[idx] = rank
    return ranks


def _expected_radix_digit_prefix(
    keys: torch.Tensor,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    block_threads: int,
    bins_per_thread: int,
) -> torch.Tensor:
    width_bits = 32
    full_mask = (1 << width_bits) - 1
    raw_unsigned = [(int(key) & full_mask) for key in keys.tolist()]
    sign_flip = 1 << (width_bits - 1)
    ordered = [value ^ sign_flip for value in raw_unsigned]
    radix_digits = 1 << (end_bit - begin_bit)
    mask = radix_digits - 1
    digits = [int((value >> begin_bit) & mask) for value in ordered]

    counts = [0 for _ in range(radix_digits)]
    for digit in digits:
        counts[digit] += 1

    prefix = [0 for _ in range(radix_digits)]
    running = 0
    digit_iter = range(radix_digits - 1, -1, -1) if descending else range(radix_digits)
    for digit in digit_iter:
        prefix[digit] = running
        running += counts[digit]

    expected = torch.full(
        (block_threads, bins_per_thread),
        -1,
        dtype=torch.int32,
    )
    for tid in range(block_threads):
        for track in range(bins_per_thread):
            bin_idx = tid * bins_per_thread + track
            if block_threads == radix_digits or bin_idx < radix_digits:
                expected[tid, track] = prefix[bin_idx]
    return expected.reshape(-1)


def test_provider_radix_sort_accepts_runtime_bit_bounds():
    cutlass.cuda.initialize_cuda_context()

    block_x = 32
    begin_bit = 4
    end_bit = 12
    keys_host = torch.tensor(
        [((idx * 17 + (idx % 5) * 7) % 97) - 48 for idx in range(block_x)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    _run_radix_sort_dynamic_bits(
        from_dlpack(keys_in),
        from_dlpack(vals_in),
        from_dlpack(keys_out),
        from_dlpack(vals_out),
        from_dlpack(keys_only_out),
        cutlass.Int32(begin_bit),
        cutlass.Int32(end_bit),
        block_x,
    )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=True,
    )

    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_radix_sort_runtime_descending_bit_slice(
    block_x: int, use_temp_storage: bool
):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [((idx * 17 + (idx % 5) * 7) % 97) - 48 for idx in range(block_x)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_radix_sort_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_radix_sort(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=4,
        end_bit=12,
        descending=True,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_radix_sort_runtime_int64_high_bit_slice(
    block_x: int, use_temp_storage: bool
):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    high_stride = 1 << 34
    keys_host = torch.tensor(
        [
            ((idx * high_stride) + ((idx % 11) << 36) - (1 << 43))
            for idx in range(block_x)
        ],
        dtype=torch.int64,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int64, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int64, device="cuda")

    if use_temp_storage:
        _run_radix_sort_64bit_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_radix_sort_64bit(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=33,
        end_bit=45,
        descending=False,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_radix_sort_runtime_uint32_keys(block_x: int, use_temp_storage: bool):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [((idx * 37 + (idx % 5) * 4099) & 0xFFFFFFFF) for idx in range(block_x)],
        dtype=torch.uint32,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.uint32, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.uint32, device="cuda")

    if use_temp_storage:
        _run_radix_sort_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_radix_sort(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=4,
        end_bit=12,
        descending=True,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_radix_sort_runtime_uint64_high_bit_slice(
    block_x: int, use_temp_storage: bool
):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [
            ((idx * (1 << 34)) + ((idx % 11) << 36) + (1 << 40))
            for idx in range(block_x)
        ],
        dtype=torch.uint64,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.uint64, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.uint64, device="cuda")

    if use_temp_storage:
        _run_radix_sort_64bit_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_radix_sort_64bit(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=33,
        end_bit=45,
        descending=False,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_radix_sort_runtime_float64_values(
    block_x: int,
    use_temp_storage: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [((idx * 17 + (idx % 5) * 7) % 97) - 48 for idx in range(block_x)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(block_x, dtype=torch.float64) * 1.25 + 0.5
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.float64, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_radix_sort_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_radix_sort(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=4,
        end_bit=12,
        descending=True,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype",
    [torch.int32, torch.uint32, torch.int64, torch.uint64],
)
def test_provider_radix_rank_runtime_signed_and_unsigned_keys(dtype: torch.dtype):
    cutlass.cuda.initialize_cuda_context()

    block_x = 32
    width_bits = 64 if dtype in {torch.int64, torch.uint64} else 32
    full_mask = (1 << width_bits) - 1
    if dtype in {torch.uint32, torch.uint64}:
        key_values = [
            ((idx * (1 << (width_bits - 4))) + idx * 37 + 11) & full_mask
            for idx in range(block_x)
        ]
    else:
        magnitude_mask = (1 << (width_bits - 2)) - 1
        magnitudes = [
            ((idx * (1 << (width_bits - 5))) + idx * 37 + 11) & magnitude_mask
            for idx in range(block_x)
        ]
        key_values = [
            -(magnitude + 1) if idx % 2 else magnitude
            for idx, magnitude in enumerate(magnitudes)
        ]
    keys_host = torch.tensor(key_values, dtype=dtype)
    keys_in = keys_host.cuda()
    rank_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    runner = _run_radix_rank_64bit if width_bits == 64 else _run_radix_rank_32bit
    descending = width_bits == 64

    runner(from_dlpack(keys_in), from_dlpack(rank_out), block_x)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        rank_out.cpu(),
        _expected_radix_ranks(
            keys_host,
            begin_bit=width_bits - 4,
            end_bit=width_bits,
            descending=descending,
        ),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "block_x,use_temp_storage,descending",
    _LAUNCH_DESCENDING_CASES,
)
def test_provider_radix_rank_runtime_exclusive_digit_prefix(
    block_x: int, use_temp_storage: bool, descending: bool
):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [(idx * 5 + (idx % 7) * 3) & 31 for idx in range(block_x)],
        dtype=torch.int32,
    )
    keys_in = keys_host.cuda()
    rank_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    prefix_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_radix_rank_prefix_temp(
            from_dlpack(keys_in),
            from_dlpack(rank_out),
            from_dlpack(prefix_out),
            block_x,
        )
    else:
        _run_radix_rank_prefix(
            from_dlpack(keys_in),
            from_dlpack(rank_out),
            from_dlpack(prefix_out),
            block_x,
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        rank_out.cpu(),
        _expected_radix_ranks(
            keys_host,
            begin_bit=0,
            end_bit=4,
            descending=descending,
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        prefix_out.cpu(),
        _expected_radix_digit_prefix(
            keys_host,
            begin_bit=0,
            end_bit=4,
            descending=descending,
            block_threads=block_x,
            bins_per_thread=1,
        ),
        atol=0,
        rtol=0,
    )


def test_provider_radix_rank_runtime_exclusive_digit_prefix_multi_track():
    cutlass.cuda.initialize_cuda_context()

    block_x = 16
    bins_per_thread = 4
    keys_host = torch.tensor(
        [(idx * 13 + (idx % 3) * 5) & 63 for idx in range(block_x)],
        dtype=torch.int32,
    )
    keys_in = keys_host.cuda()
    prefix_out = torch.zeros(
        (block_x * bins_per_thread,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_radix_rank_prefix_multi_track(
        from_dlpack(keys_in),
        from_dlpack(prefix_out),
        block_x,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        prefix_out.cpu(),
        _expected_radix_digit_prefix(
            keys_host,
            begin_bit=0,
            end_bit=6,
            descending=False,
            block_threads=block_x,
            bins_per_thread=bins_per_thread,
        ),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "block_x,use_temp_storage,descending,use_register_payload",
    [
        (*case, use_register_payload)
        for case in _LAUNCH_DESCENDING_CASES
        for use_register_payload in (False, True)
    ],
)
def test_provider_radix_rank_runtime_multi_item_payloads(
    block_x: int,
    use_temp_storage: bool,
    descending: bool,
    use_register_payload: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    items_per_thread = 3
    total_items = block_x * items_per_thread
    keys_host = torch.tensor(
        [(idx * 5 + (idx % 7) * 3) & 31 for idx in range(total_items)],
        dtype=torch.int32,
    )
    keys_in = keys_host.cuda()
    rank_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    prefix_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_radix_rank_multi_item_payload_temp(
            from_dlpack(keys_in),
            from_dlpack(rank_out),
            from_dlpack(prefix_out),
            block_x,
            use_register_payload,
        )
    else:
        _run_radix_rank_multi_item_payload(
            from_dlpack(keys_in),
            from_dlpack(rank_out),
            from_dlpack(prefix_out),
            block_x,
            use_register_payload,
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        rank_out.cpu(),
        _expected_radix_ranks(
            keys_host,
            begin_bit=0,
            end_bit=4,
            descending=descending,
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        prefix_out.cpu(),
        _expected_radix_digit_prefix(
            keys_host,
            begin_bit=0,
            end_bit=4,
            descending=descending,
            block_threads=block_x,
            bins_per_thread=1,
        ),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "block_x,use_temp_storage,descending,use_register_payload",
    [
        (*case, use_register_payload)
        for case in _LAUNCH_DESCENDING_CASES
        for use_register_payload in (False, True)
    ],
)
def test_provider_radix_sort_runtime_multi_item_payloads(
    block_x: int,
    use_temp_storage: bool,
    descending: bool,
    use_register_payload: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _RADIX_TEMP_STORAGE.reset_uses()

    items_per_thread = 3
    total_items = block_x * items_per_thread
    keys_host = torch.tensor(
        [((idx * 11 + (idx % 7) * 5) % 53) - 26 for idx in range(total_items)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(total_items, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_radix_sort_multi_item_payload_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            use_register_payload,
        )
    else:
        _run_radix_sort_multi_item_payload(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            use_register_payload,
        )
    torch.cuda.synchronize()

    expected_keys, expected_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=0,
        end_bit=4,
        descending=descending,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(vals_out.cpu(), expected_vals, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize("block_x", [16, 32, 64])
def test_provider_topk_runtime_score_window(block_x: int):
    cutlass.cuda.initialize_cuda_context()
    _TOPK_TEMP_STORAGE.reset_uses()

    keys_data = [64 + ((idx * 37 + 11) % 128) for idx in range(block_x)]
    duplicate_min_keys = [1, 1, 2, 2, 3, 3, 4]
    keys_data[: len(duplicate_min_keys)] = duplicate_min_keys
    keys_host = torch.tensor(keys_data, dtype=torch.int32)
    vals_host = torch.arange(block_x, dtype=torch.int32) * 13 + 5
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    max_keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    min_pair_keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    min_pair_vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    values_only_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    _run_topk_score_window(
        from_dlpack(keys_in),
        from_dlpack(vals_in),
        from_dlpack(max_keys_out),
        from_dlpack(min_pair_keys_out),
        from_dlpack(min_pair_vals_out),
        block_x,
    )
    _run_topk_pair_values_only(
        from_dlpack(keys_in),
        from_dlpack(vals_in),
        from_dlpack(values_only_out),
        block_x,
    )
    torch.cuda.synchronize()

    k = min(_TOPK_SCORE_K, block_x)
    expected_max_keys, _ = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=0,
        end_bit=8,
        descending=True,
    )
    expected_min_keys, expected_min_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=0,
        end_bit=8,
        descending=False,
    )

    _assert_topk_keys_unordered(max_keys_out[:k], expected_max_keys[:k])
    _assert_topk_pairs_unordered(
        min_pair_keys_out[:k],
        min_pair_vals_out[:k],
        expected_min_keys[:k],
        expected_min_vals[:k],
    )
    _assert_pairs_still_match_input(
        keys_host,
        vals_host,
        min_pair_keys_out[:k],
        min_pair_vals_out[:k],
    )
    torch.testing.assert_close(
        torch.sort(values_only_out[:k].cpu()).values,
        torch.sort(expected_min_vals[:k]).values,
        atol=0,
        rtol=0,
    )


def test_provider_topk_runtime_float32_keys_float64_values_partial_tile():
    cutlass.cuda.initialize_cuda_context()
    _TOPK_TEMP_STORAGE.reset_uses()

    block_x = 32
    valid_items = 21
    k = 7
    keys_host = torch.tensor(
        [((idx * 17 + 5) % 97) * 0.25 + 0.5 for idx in range(block_x)],
        dtype=torch.float32,
    )
    vals_host = torch.arange(block_x, dtype=torch.float64) * 1.25 + 2.5
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    max_keys_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    min_pair_keys_out = torch.zeros((block_x,), dtype=torch.float32, device="cuda")
    min_pair_vals_out = torch.zeros((block_x,), dtype=torch.float64, device="cuda")

    _run_topk_float_partial(
        from_dlpack(keys_in),
        from_dlpack(vals_in),
        from_dlpack(max_keys_out),
        from_dlpack(min_pair_keys_out),
        from_dlpack(min_pair_vals_out),
        block_x,
        valid_items,
    )
    torch.cuda.synchronize()

    valid_keys = keys_host[:valid_items]
    valid_vals = vals_host[:valid_items]
    max_order = sorted(
        range(valid_items),
        key=lambda idx: (-float(valid_keys[idx].item()), idx),
    )
    min_order = sorted(
        range(valid_items),
        key=lambda idx: (float(valid_keys[idx].item()), idx),
    )
    expected_max_keys = valid_keys[torch.tensor(max_order[:k], dtype=torch.long)]
    expected_min_indices = torch.tensor(min_order[:k], dtype=torch.long)
    expected_min_keys = valid_keys[expected_min_indices]
    expected_min_vals = valid_vals[expected_min_indices]

    _assert_topk_keys_unordered(max_keys_out[:k], expected_max_keys)
    _assert_topk_pairs_unordered(
        min_pair_keys_out[:k],
        min_pair_vals_out[:k],
        expected_min_keys,
        expected_min_vals,
    )


@pytest.mark.parametrize("use_register_payload", [False, True])
def test_provider_topk_runtime_multi_item_payload_score_window(
    use_register_payload: bool,
):
    cutlass.cuda.initialize_cuda_context()

    block_x = 32
    items_per_thread = 3
    total_items = block_x * items_per_thread
    keys_host = torch.tensor(
        [((idx * 29 + 17) % 251) for idx in range(total_items)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(total_items, dtype=torch.int32) * 7 + 3
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    max_keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    min_pair_keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    min_pair_vals_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")

    _run_topk_multi_item_payload_score_window(
        from_dlpack(keys_in),
        from_dlpack(vals_in),
        from_dlpack(max_keys_out),
        from_dlpack(min_pair_keys_out),
        from_dlpack(min_pair_vals_out),
        block_x,
        use_register_payload,
    )
    torch.cuda.synchronize()

    expected_max_keys, _ = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=0,
        end_bit=8,
        descending=True,
    )
    expected_min_keys, expected_min_vals = _expected_radix_order(
        keys_host,
        vals_host,
        begin_bit=0,
        end_bit=8,
        descending=False,
    )

    _assert_topk_keys_unordered(
        max_keys_out[:_TOPK_SCORE_K],
        expected_max_keys[:_TOPK_SCORE_K],
    )
    _assert_topk_pairs_unordered(
        min_pair_keys_out[:_TOPK_SCORE_K],
        min_pair_vals_out[:_TOPK_SCORE_K],
        expected_min_keys[:_TOPK_SCORE_K],
        expected_min_vals[:_TOPK_SCORE_K],
    )
    _assert_pairs_still_match_input(
        keys_host,
        vals_host,
        min_pair_keys_out[:_TOPK_SCORE_K],
        min_pair_vals_out[:_TOPK_SCORE_K],
    )
