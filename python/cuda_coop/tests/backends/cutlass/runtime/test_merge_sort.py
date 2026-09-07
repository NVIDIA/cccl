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
    MERGE_TEMP_STORAGE as _MERGE_TEMP_STORAGE,
)
from ..support.runtime import (
    Float64,
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
def _merge_sort_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    descending = cutlass.const_expr(True)
    group = coop.this_block()
    if cutlass.const_expr(use_register_payload):
        key = cute.make_rmem_tensor(1, Int32)
        key[0] = keys_in[tidx]
        val = cute.make_rmem_tensor(1, Int32)
        val[0] = vals_in[tidx]
        sorted_key, sorted_val = coop.merge_sort_pairs(
            group, key, val.load(), descending=descending
        )
        sorted_key_only = coop.merge_sort_keys(group, key.load(), descending=descending)
        if tidx < block_x:
            keys_out[tidx] = sorted_key[0]
            vals_out[tidx] = sorted_val[0]
            keys_only_out[tidx] = sorted_key_only[0]
    else:
        key = keys_in[tidx]
        val = vals_in[tidx]
        sorted_key, sorted_val = coop.merge_sort_pairs(
            group, key, val, descending=descending
        )
        sorted_key_only = coop.merge_sort_keys(group, key, descending=descending)
        if tidx < block_x:
            keys_out[tidx] = sorted_key
            vals_out[tidx] = sorted_val
            keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _merge_sort_temp_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    descending = cutlass.const_expr(True)
    if cutlass.const_expr(use_register_payload):
        key = cute.make_rmem_tensor(1, Int32)
        key[0] = keys_in[tidx]
        val = cute.make_rmem_tensor(1, Int32)
        val[0] = vals_in[tidx]
        sorted_key, sorted_val = coop._block.merge_sort_pairs(
            key,
            val.load(),
            descending=descending,
            temp_storage=_MERGE_TEMP_STORAGE,
        )
        sorted_key_only = coop._block.merge_sort_keys(
            key.load(),
            descending=descending,
            temp_storage=_MERGE_TEMP_STORAGE,
        )
        if tidx < block_x:
            keys_out[tidx] = sorted_key[0]
            vals_out[tidx] = sorted_val[0]
            keys_only_out[tidx] = sorted_key_only[0]
    else:
        key = keys_in[tidx]
        val = vals_in[tidx]
        sorted_key, sorted_val = coop._block.merge_sort_pairs(
            key, val, descending=descending, temp_storage=_MERGE_TEMP_STORAGE
        )
        sorted_key_only = coop._block.merge_sort_keys(
            key, descending=descending, temp_storage=_MERGE_TEMP_STORAGE
        )
        if tidx < block_x:
            keys_out[tidx] = sorted_key
            vals_out[tidx] = sorted_val
            keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _merge_sort_partial_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    valid_items: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    descending = cutlass.const_expr(True)
    oob_default = cutlass.const_expr(-1000000)
    group = coop.this_block()
    sorted_key, sorted_val = coop.merge_sort_pairs(
        group,
        key,
        val,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
    )
    sorted_key_only = coop.merge_sort_keys(
        group,
        key,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val
        keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _merge_sort_partial_temp_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    valid_items: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    key = keys_in[tidx]
    val = vals_in[tidx]
    descending = cutlass.const_expr(True)
    oob_default = cutlass.const_expr(-1000000)
    sorted_key, sorted_val = coop._block.merge_sort_pairs(
        key,
        val,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=_MERGE_TEMP_STORAGE,
    )
    sorted_key_only = coop._block.merge_sort_keys(
        key,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=_MERGE_TEMP_STORAGE,
    )
    if tidx < block_x:
        keys_out[tidx] = sorted_key
        vals_out[tidx] = sorted_val
        keys_only_out[tidx] = sorted_key_only


@cute.kernel
def _merge_sort_thread_data_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
):
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    descending = cutlass.const_expr(False)
    sorted_keys, sorted_vals = coop._block.merge_sort_pairs(
        keys, vals, descending=descending
    )
    sorted_keys_only = coop._block.merge_sort_keys(keys, descending=descending)
    coop._block.store(keys_out, sorted_keys)
    coop._block.store(vals_out, sorted_vals)
    coop._block.store(keys_only_out, sorted_keys_only)


@cute.kernel
def _merge_sort_thread_data_temp_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
):
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    descending = cutlass.const_expr(True)
    sorted_keys, sorted_vals = coop._block.merge_sort_pairs(
        keys,
        vals,
        descending=descending,
        temp_storage=_MERGE_TEMP_STORAGE,
    )
    sorted_keys_only = coop._block.merge_sort_keys(
        keys,
        descending=descending,
        temp_storage=_MERGE_TEMP_STORAGE,
    )
    coop._block.store(keys_out, sorted_keys)
    coop._block.store(vals_out, sorted_vals)
    coop._block.store(keys_only_out, sorted_keys_only)


@cute.kernel
def _merge_sort_thread_data_partial_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    valid_items: cutlass.Constexpr,
):
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    descending = cutlass.const_expr(True)
    oob_default = cutlass.const_expr(-1000000)
    sorted_keys, sorted_vals = coop._block.merge_sort_pairs(
        keys,
        vals,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
    )
    sorted_keys_only = coop._block.merge_sort_keys(
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
    )
    coop._block.store(keys_out, sorted_keys)
    coop._block.store(vals_out, sorted_vals)
    coop._block.store(keys_only_out, sorted_keys_only)


@cute.kernel
def _merge_sort_thread_data_partial_temp_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    valid_items: cutlass.Constexpr,
):
    keys = coop._block.load(keys_in, items_per_thread=3, dtype=Int32)
    vals = coop._block.load(vals_in, items_per_thread=3, dtype=Int32)
    descending = cutlass.const_expr(True)
    oob_default = cutlass.const_expr(-1000000)
    sorted_keys, sorted_vals = coop._block.merge_sort_pairs(
        keys,
        vals,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=_MERGE_TEMP_STORAGE,
    )
    sorted_keys_only = coop._block.merge_sort_keys(
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=_MERGE_TEMP_STORAGE,
    )
    coop._block.store(keys_out, sorted_keys)
    coop._block.store(vals_out, sorted_vals)
    coop._block.store(keys_only_out, sorted_keys_only)


@cute.jit
def _run_merge_sort(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr = False,
):
    _merge_sort_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
        use_register_payload,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_merge_sort_temp(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    use_register_payload: cutlass.Constexpr = False,
):
    _merge_sort_temp_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
        use_register_payload,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_merge_sort_partial(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    valid_items: cutlass.Constexpr,
):
    _merge_sort_partial_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_merge_sort_partial_temp(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    valid_items: cutlass.Constexpr,
):
    _merge_sort_partial_temp_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        block_x,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_merge_sort_thread_data(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _merge_sort_thread_data_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_merge_sort_thread_data_temp(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
):
    _merge_sort_thread_data_temp_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_merge_sort_thread_data_partial(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    valid_items: cutlass.Constexpr,
):
    _merge_sort_thread_data_partial_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.jit
def _run_merge_sort_thread_data_partial_temp(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    keys_out: cute.Tensor,
    vals_out: cute.Tensor,
    keys_only_out: cute.Tensor,
    block_x: cutlass.Constexpr,
    valid_items: cutlass.Constexpr,
):
    _merge_sort_thread_data_partial_temp_kernel(
        keys_in,
        vals_in,
        keys_out,
        vals_out,
        keys_only_out,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(block_x, 1, 1))


@cute.kernel
def _warp_merge_sort_register_payload_kernel(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    desc_keys_out: cute.Tensor,
    pair_keys_out: cute.Tensor,
    pair_vals_out: cute.Tensor,
):
    keys = coop._warp.load(
        keys_in,
        items_per_thread=2,
        dtype=Int32,
        threads_in_warp=16,
    )
    vals = coop._warp.load(
        vals_in,
        items_per_thread=2,
        dtype=Float64,
        threads_in_warp=16,
    )
    key_fragment = cute.make_rmem_tensor((1, 2), Int32)
    key_fragment[0] = keys[0]
    key_fragment[1] = keys[1]
    value_fragment = cute.make_rmem_tensor((1, 2), Float64)
    value_fragment[0] = vals[0]
    value_fragment[1] = vals[1]
    desc_keys = coop._warp.merge_sort_keys(
        key_fragment,
        compare_op=">",
        threads_in_warp=16,
    )
    pair_keys, pair_vals = coop._warp.merge_sort_pairs(
        key_fragment.load(),
        value_fragment,
        compare_op="<",
        threads_in_warp=16,
    )
    coop._warp.store(desc_keys_out, desc_keys, threads_in_warp=16)
    coop._warp.store(pair_keys_out, pair_keys, threads_in_warp=16)
    coop._warp.store(pair_vals_out, pair_vals, threads_in_warp=16)


@cute.jit
def _run_warp_merge_sort_register_payload(
    keys_in: cute.Tensor,
    vals_in: cute.Tensor,
    desc_keys_out: cute.Tensor,
    pair_keys_out: cute.Tensor,
    pair_vals_out: cute.Tensor,
):
    _warp_merge_sort_register_payload_kernel(
        keys_in,
        vals_in,
        desc_keys_out,
        pair_keys_out,
        pair_vals_out,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1))


def _expected_merge_order(
    keys: torch.Tensor, values: torch.Tensor, *, descending: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    keys_list = keys.tolist()
    idx = list(range(len(keys_list)))
    if descending:
        idx = sorted(idx, key=lambda i: (-keys_list[i], i))
    else:
        idx = sorted(idx, key=lambda i: (keys_list[i], i))
    return _gather_cpu_tensor(keys, idx), _gather_cpu_tensor(values, idx)


def _expected_merge_order_partial(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    descending: bool,
    valid_items: int,
    oob_default: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_keys = torch.full_like(keys, oob_default)
    expected_values = torch.empty_like(values)
    sorted_keys, sorted_values = _expected_merge_order(
        keys[:valid_items],
        values[:valid_items],
        descending=descending,
    )
    expected_keys[:valid_items] = sorted_keys
    expected_values[:valid_items] = sorted_values
    expected_values[valid_items:] = values[valid_items:]
    return expected_keys, expected_values


def _expected_warp_merge_order(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    threads_in_warp: int,
    items_per_thread: int,
    descending: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_keys = torch.empty_like(keys)
    expected_values = torch.empty_like(values)
    tile_items = threads_in_warp * items_per_thread
    for tile_base in range(0, len(keys), tile_items):
        tile_end = tile_base + tile_items
        tile_keys, tile_values = _expected_merge_order(
            keys[tile_base:tile_end],
            values[tile_base:tile_end],
            descending=descending,
        )
        expected_keys[tile_base:tile_end] = tile_keys
        expected_values[tile_base:tile_end] = tile_values
    return expected_keys, expected_values


@pytest.mark.parametrize(
    "block_x,use_temp_storage,use_register_payload",
    [
        (block_x, use_temp_storage, use_register_payload)
        for block_x, use_temp_storage in _LAUNCH_CASES
        for use_register_payload in (False, True)
    ],
)
def test_provider_merge_sort_runtime_descending(
    block_x: int,
    use_temp_storage: bool,
    use_register_payload: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _MERGE_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [((idx * 23 + (idx % 9) * 11) % 113) - 56 for idx in range(block_x)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_merge_sort_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            use_register_payload,
        )
    else:
        _run_merge_sort(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            use_register_payload,
        )
    torch.cuda.synchronize()

    expected_keys, _ = _expected_merge_order(keys_host, vals_host, descending=True)
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)
    assert _MERGE_TEMP_STORAGE.uses == ()


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_merge_sort_runtime_int64_keys(
    block_x: int,
    use_temp_storage: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _MERGE_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [((idx * (1 << 33)) - (1 << 42) + ((idx % 7) * 19)) for idx in range(block_x)],
        dtype=torch.int64,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int64, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int64, device="cuda")

    if use_temp_storage:
        _run_merge_sort_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_merge_sort(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, _ = _expected_merge_order(
        keys_host,
        vals_host,
        descending=True,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "torch_dtype",
    [
        pytest.param(torch.uint32, id="uint32"),
        pytest.param(torch.uint64, id="uint64"),
    ],
)
@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_merge_sort_runtime_unsigned_keys(
    block_x: int,
    use_temp_storage: bool,
    torch_dtype: torch.dtype,
):
    cutlass.cuda.initialize_cuda_context()
    _MERGE_TEMP_STORAGE.reset_uses()

    if torch_dtype == torch.uint32:
        key_values = [
            ((idx * 37 + (idx % 7) * 4099) & 0xFFFFFFFF) for idx in range(block_x)
        ]
    else:
        key_values = [
            ((idx * (1 << 34)) + ((idx % 11) << 36) + (1 << 40))
            for idx in range(block_x)
        ]
    keys_host = torch.tensor(key_values, dtype=torch_dtype)
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch_dtype, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch_dtype, device="cuda")

    if use_temp_storage:
        _run_merge_sort_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_merge_sort(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, _ = _expected_merge_order(
        keys_host,
        vals_host,
        descending=True,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "block_x,use_temp_storage",
    _LAUNCH_CASES,
)
def test_provider_merge_sort_runtime_float64_values(
    block_x: int,
    use_temp_storage: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _MERGE_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [((idx * 23 + (idx % 9) * 11) % 113) - 56 for idx in range(block_x)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(block_x, dtype=torch.float64) * 0.75 - 12.5
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.float64, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_merge_sort_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_merge_sort(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, _ = _expected_merge_order(
        keys_host,
        vals_host,
        descending=True,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)


@pytest.mark.parametrize(
    "block_x,use_temp_storage,valid_items",
    [
        (32, False, 21),
        (64, True, 47),
    ],
)
def test_provider_merge_sort_runtime_partial_tiles(
    block_x: int,
    use_temp_storage: bool,
    valid_items: int,
):
    cutlass.cuda.initialize_cuda_context()
    _MERGE_TEMP_STORAGE.reset_uses()

    keys_host = torch.tensor(
        [((idx * 23 + (idx % 9) * 11) % 113) - 56 for idx in range(block_x)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(block_x, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((block_x,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_merge_sort_partial_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            valid_items,
        )
    else:
        _run_merge_sort_partial(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            valid_items,
        )
    torch.cuda.synchronize()

    expected_keys, _ = _expected_merge_order_partial(
        keys_host,
        vals_host,
        descending=True,
        valid_items=valid_items,
        oob_default=-1000000,
    )
    torch.testing.assert_close(
        keys_out[:valid_items].cpu(), expected_keys[:valid_items], atol=0, rtol=0
    )
    torch.testing.assert_close(
        keys_only_out[:valid_items].cpu(),
        expected_keys[:valid_items],
        atol=0,
        rtol=0,
    )
    _assert_pairs_still_match_input(
        keys_host[:valid_items],
        vals_host[:valid_items],
        keys_out[:valid_items],
        vals_out[:valid_items],
    )
    assert _MERGE_TEMP_STORAGE.uses == ()


@pytest.mark.parametrize(
    "block_x,use_temp_storage,descending",
    _LAUNCH_DESCENDING_CASES,
)
def test_provider_merge_sort_runtime_thread_data_multi_item(
    block_x: int,
    use_temp_storage: bool,
    descending: bool,
):
    cutlass.cuda.initialize_cuda_context()
    _MERGE_TEMP_STORAGE.reset_uses()

    items_per_thread = 3
    total_items = block_x * items_per_thread
    keys_host = torch.tensor(
        [((idx * 23 + (idx % 9) * 11) % 113) - 56 for idx in range(total_items)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(total_items, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_merge_sort_thread_data_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    else:
        _run_merge_sort_thread_data(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
        )
    torch.cuda.synchronize()

    expected_keys, _ = _expected_merge_order(
        keys_host,
        vals_host,
        descending=descending,
    )
    torch.testing.assert_close(keys_out.cpu(), expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(keys_only_out.cpu(), expected_keys, atol=0, rtol=0)
    _assert_pairs_still_match_input(keys_host, vals_host, keys_out, vals_out)
    assert _MERGE_TEMP_STORAGE.uses == ()


@pytest.mark.parametrize(
    "block_x,use_temp_storage,valid_items",
    [
        (32, False, 73),
        (64, True, 151),
    ],
)
def test_provider_merge_sort_runtime_thread_data_partial_tiles(
    block_x: int,
    use_temp_storage: bool,
    valid_items: int,
):
    cutlass.cuda.initialize_cuda_context()
    _MERGE_TEMP_STORAGE.reset_uses()

    items_per_thread = 3
    total_items = block_x * items_per_thread
    keys_host = torch.tensor(
        [((idx * 23 + (idx % 9) * 11) % 113) - 56 for idx in range(total_items)],
        dtype=torch.int32,
    )
    vals_host = torch.arange(total_items, dtype=torch.int32)
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    vals_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    keys_only_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")

    if use_temp_storage:
        _run_merge_sort_thread_data_partial_temp(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            valid_items,
        )
    else:
        _run_merge_sort_thread_data_partial(
            from_dlpack(keys_in),
            from_dlpack(vals_in),
            from_dlpack(keys_out),
            from_dlpack(vals_out),
            from_dlpack(keys_only_out),
            block_x,
            valid_items,
        )
    torch.cuda.synchronize()

    expected_keys, _ = _expected_merge_order_partial(
        keys_host,
        vals_host,
        descending=True,
        valid_items=valid_items,
        oob_default=-1000000,
    )
    torch.testing.assert_close(
        keys_out[:valid_items].cpu(), expected_keys[:valid_items], atol=0, rtol=0
    )
    torch.testing.assert_close(
        keys_only_out[:valid_items].cpu(),
        expected_keys[:valid_items],
        atol=0,
        rtol=0,
    )
    _assert_pairs_still_match_input(
        keys_host[:valid_items],
        vals_host[:valid_items],
        keys_out[:valid_items],
        vals_out[:valid_items],
    )
    assert _MERGE_TEMP_STORAGE.uses == ()


def test_provider_warp_merge_sort_runtime_register_payload_subwarp_tiles():
    cutlass.cuda.initialize_cuda_context()

    block_x = 32
    threads_in_warp = 16
    items_per_thread = 2
    tile_items = threads_in_warp * items_per_thread
    total_items = block_x * items_per_thread
    key_values = []
    for tile_base in range(0, total_items, tile_items):
        key_values.extend(
            tile_base + ((local_idx * 7 + 3) % tile_items) - 11
            for local_idx in range(tile_items)
        )
    keys_host = torch.tensor(key_values, dtype=torch.int32)
    vals_host = torch.arange(total_items, dtype=torch.float64) * 0.5 - 13.0
    keys_in = keys_host.cuda()
    vals_in = vals_host.cuda()
    desc_keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    pair_keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    pair_vals_out = torch.zeros((total_items,), dtype=torch.float64, device="cuda")

    _run_warp_merge_sort_register_payload(
        from_dlpack(keys_in),
        from_dlpack(vals_in),
        from_dlpack(desc_keys_out),
        from_dlpack(pair_keys_out),
        from_dlpack(pair_vals_out),
    )
    torch.cuda.synchronize()

    expected_desc_keys, _ = _expected_warp_merge_order(
        keys_host,
        vals_host,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        descending=True,
    )
    expected_pair_keys, expected_pair_vals = _expected_warp_merge_order(
        keys_host,
        vals_host,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        descending=False,
    )
    torch.testing.assert_close(desc_keys_out.cpu(), expected_desc_keys, atol=0, rtol=0)
    torch.testing.assert_close(pair_keys_out.cpu(), expected_pair_keys, atol=0, rtol=0)
    torch.testing.assert_close(pair_vals_out.cpu(), expected_pair_vals, atol=0, rtol=0)
