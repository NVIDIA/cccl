# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path


def write_prims_api_vector_sort_topk_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str = "cuda.coop.cutlass",
    call_mode: str = "factory",
) -> None:
    """Write a source-backed CuTe/Prims block smoke for a CUTLASS API.

    CUTLASS DSL kernels need inspectable source, so tests write this script to
    disk and run it as a subprocess instead of using ``python -c``.
    """

    if call_mode not in {"factory", "direct"}:
        raise ValueError("call_mode must be 'factory' or 'direct'")

    block_scope_expr = "coop._block"

    if call_mode == "factory":
        primitive_setup = """
radix_sort_keys = block_scope.make_radix_sort_keys(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    begin_bit=FACTORY_DEFAULT_BEGIN_BIT,
    end_bit=FACTORY_DEFAULT_END_BIT,
    descending=True,
    temp_storage=radix_temp_storage,
)
radix_sort_pairs = block_scope.make_radix_sort_pairs(
    cutlass.Int32,
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    begin_bit=FACTORY_DEFAULT_BEGIN_BIT,
    end_bit=FACTORY_DEFAULT_END_BIT,
    descending=True,
    temp_storage=radix_temp_storage,
)
topk_max_keys = block_scope.make_topk_max_keys(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    num_valid=FACTORY_DEFAULT_TOPK_VALID_ITEMS,
    begin_bit=FACTORY_DEFAULT_BEGIN_BIT,
    end_bit=FACTORY_DEFAULT_END_BIT,
    temp_storage=topk_temp_storage,
)
topk_min_pairs = block_scope.make_topk_min_pairs(
    cutlass.Int32,
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    num_valid=FACTORY_DEFAULT_TOPK_VALID_ITEMS,
    begin_bit=FACTORY_DEFAULT_BEGIN_BIT,
    end_bit=FACTORY_DEFAULT_END_BIT,
    temp_storage=topk_temp_storage,
)
"""
        primitive_calls = """
    sorted_keys = radix_sort_keys(
        keys_vec,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=False,
    )
    top_keys = topk_max_keys(
        keys_vec,
        topk_k,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    sorted_pair_keys, sorted_pair_values = radix_sort_pairs(
        keys_vec,
        values_vec,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=False,
    )
    top_pair_keys, top_pair_values = topk_min_pairs(
        keys_vec,
        values_vec,
        topk_k,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
"""
        scope_metadata = """
    "factory_scopes": [
        radix_sort_keys.scope,
        radix_sort_pairs.scope,
        topk_max_keys.scope,
        topk_min_pairs.scope,
    ],
"""
    else:
        primitive_setup = ""
        primitive_calls = """
    sorted_keys = block_scope.radix_sort_keys(
        keys_vec,
        threads_per_block=BLOCK_THREADS,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=False,
        temp_storage=radix_temp_storage,
    )
    top_keys = block_scope.topk_max_keys(
        keys_vec,
        topk_k,
        num_valid=num_valid,
        threads_per_block=BLOCK_THREADS,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=topk_temp_storage,
    )
    sorted_pair_keys, sorted_pair_values = block_scope.radix_sort_pairs(
        keys_vec,
        values_vec,
        threads_per_block=BLOCK_THREADS,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=False,
        temp_storage=radix_temp_storage,
    )
    top_pair_keys, top_pair_values = block_scope.topk_min_pairs(
        keys_vec,
        values_vec,
        topk_k,
        num_valid=num_valid,
        threads_per_block=BLOCK_THREADS,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=topk_temp_storage,
    )
"""
        scope_metadata = """
    "primitive_modules": [
        block_scope.radix_sort_keys.__module__,
        block_scope.radix_sort_pairs.__module__,
        block_scope.topk_max_keys.__module__,
        block_scope.topk_min_pairs.__module__,
        block_scope.store.__module__,
    ],
"""

    script_path.write_text(
        f"""
import json
import importlib
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_pair_sort_topk import (
    _assert_topk_pairs_unordered,
    _expected_radix_pairs,
)
from examples.cutlass.prims_vector_sort_topk import (
    _assert_topk_keys_unordered,
    _expected_radix_keys,
)

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})
block_scope = {block_scope_expr}

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
TOPK_K = 5
TOPK_VALID_ITEMS = TOTAL_ITEMS - 9
FACTORY_DEFAULT_BEGIN_BIT = 4
FACTORY_DEFAULT_END_BIT = 12
FACTORY_DEFAULT_TOPK_VALID_ITEMS = TOTAL_ITEMS

radix_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
topk_temp_storage = block_scope.TempStorage(size_in_bytes=16384, sharing="shared")
{primitive_setup}


@cute.kernel
def _prims_api_vector_sort_topk_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    sorted_keys_out: cute.Tensor,
    sorted_pair_keys_out: cute.Tensor,
    sorted_values_out: cute.Tensor,
    top_keys_out: cute.Tensor,
    top_pair_keys_out: cute.Tensor,
    top_pair_values_out: cute.Tensor,
    topk_k: cutlass.Int32,
    num_valid: cutlass.Int32,
    begin_bit: cutlass.Int32,
    end_bit: cutlass.Int32,
    items_per_thread: cutlass.Constexpr,
):
    keys_vec = block_scope.load(
        keys_in,
        payload=coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )
    values_vec = block_scope.load(
        values_in,
        payload=coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )

{primitive_calls}

    block_scope.store(sorted_keys_out, sorted_keys, payload=coop.Payload.PRIMS)
    block_scope.store(
        sorted_pair_keys_out, sorted_pair_keys, payload=coop.Payload.PRIMS
    )
    block_scope.store(
        sorted_values_out, sorted_pair_values, payload=coop.Payload.PRIMS
    )
    block_scope.store(top_keys_out, top_keys, payload=coop.Payload.PRIMS)
    block_scope.store(
        top_pair_keys_out, top_pair_keys, payload=coop.Payload.PRIMS
    )
    block_scope.store(
        top_pair_values_out, top_pair_values, payload=coop.Payload.PRIMS
    )


@cute.jit
def _run_prims_api_vector_sort_topk(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    sorted_keys_out: cute.Tensor,
    sorted_pair_keys_out: cute.Tensor,
    sorted_values_out: cute.Tensor,
    top_keys_out: cute.Tensor,
    top_pair_keys_out: cute.Tensor,
    top_pair_values_out: cute.Tensor,
    topk_k: cutlass.Int32,
    num_valid: cutlass.Int32,
    begin_bit: cutlass.Int32,
    end_bit: cutlass.Int32,
):
    _prims_api_vector_sort_topk_kernel(
        keys_in,
        values_in,
        sorted_keys_out,
        sorted_pair_keys_out,
        sorted_values_out,
        top_keys_out,
        top_pair_keys_out,
        top_pair_values_out,
        topk_k,
        num_valid,
        begin_bit,
        end_bit,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
begin_bit = 0
end_bit = 8
keys_host = torch.tensor(
    [((idx * 17 + 23) % 251) for idx in range(TOTAL_ITEMS)],
    dtype=torch.int32,
)
values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 13 + 7
keys_in = keys_host.cuda()
values_in = values_host.cuda()
sorted_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
sorted_pair_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
sorted_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
top_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
top_pair_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
top_pair_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_prims_api_vector_sort_topk(
    from_dlpack(keys_in),
    from_dlpack(values_in),
    from_dlpack(sorted_keys_out),
    from_dlpack(sorted_pair_keys_out),
    from_dlpack(sorted_values_out),
    from_dlpack(top_keys_out),
    from_dlpack(top_pair_keys_out),
    from_dlpack(top_pair_values_out),
    cutlass.Int32(TOPK_K),
    cutlass.Int32(TOPK_VALID_ITEMS),
    cutlass.Int32(begin_bit),
    cutlass.Int32(end_bit),
)
torch.cuda.synchronize()

expected_sorted = _expected_radix_keys(
    keys_host,
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=False,
    torch=torch,
)
expected_top = _expected_radix_keys(
    keys_host[:TOPK_VALID_ITEMS],
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=True,
    torch=torch,
)
torch.testing.assert_close(sorted_keys_out.cpu(), expected_sorted, atol=0, rtol=0)
_assert_topk_keys_unordered(
    top_keys_out[:TOPK_K],
    expected_top[:TOPK_K],
    torch=torch,
)
expected_pair_keys, expected_pair_values = _expected_radix_pairs(
    keys_host,
    values_host,
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=False,
    torch=torch,
)
torch.testing.assert_close(sorted_values_out.cpu(), expected_pair_values, atol=0, rtol=0)
torch.testing.assert_close(
    sorted_pair_keys_out.cpu(),
    expected_pair_keys,
    atol=0,
    rtol=0,
)
expected_top_pair_keys, expected_top_pair_values = _expected_radix_pairs(
    keys_host[:TOPK_VALID_ITEMS],
    values_host[:TOPK_VALID_ITEMS],
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=False,
    torch=torch,
)
_assert_topk_pairs_unordered(
    top_pair_keys_out[:TOPK_K],
    top_pair_values_out[:TOPK_K],
    expected_top_pair_keys[:TOPK_K],
    expected_top_pair_values[:TOPK_K],
)
print(json.dumps({{
    "api_module": coop.__name__,
{scope_metadata}
    "topk_valid_items": int(TOPK_VALID_ITEMS),
    "sorted_keys": [int(x) for x in sorted_keys_out[:TOPK_K].cpu().tolist()],
    "top_keys": [int(x) for x in top_keys_out[:TOPK_K].cpu().tolist()],
    "sorted_pairs": [
        [int(key), int(value)]
        for key, value in zip(
            sorted_pair_keys_out[:TOPK_K].cpu().tolist(),
            sorted_values_out[:TOPK_K].cpu().tolist(),
            strict=True,
        )
    ],
    "top_pairs": [
        [int(key), int(value)]
        for key, value in zip(
            top_pair_keys_out[:TOPK_K].cpu().tolist(),
            top_pair_values_out[:TOPK_K].cpu().tolist(),
            strict=True,
        )
    ],
}}, sort_keys=True))
	""",
        encoding="utf-8",
    )


def write_cutlass_mixed_payload_factory_smoke(
    script_path: Path,
    *,
    source_root: Path,
) -> None:
    """Write a source-backed root-factory smoke for mixed Prims/CuTe payloads."""

    script_path.write_text(
        f"""
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_sort_topk import (
    _assert_topk_keys_unordered,
    _expected_radix_keys,
)

cutlass, cute, torch, from_dlpack, coop, Int32 = require_runtime(include_int32=True)

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
TOPK_K = 5
TOPK_VALID_ITEMS = TOTAL_ITEMS - 9

vector_sort_temp_storage = coop._block.TempStorage(size_in_bytes=8192, sharing="shared")
vector_topk_temp_storage = coop._block.TempStorage(size_in_bytes=16384, sharing="shared")
fragment_sort_temp_storage = coop._block.TempStorage(size_in_bytes=8192, sharing="shared")

vector_load = coop._block.make_load(
    cutlass.Int32,
    payload=coop.Payload.PRIMS,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
vector_store = coop._block.make_store(
    cutlass.Int32,
    payload=coop.Payload.PRIMS,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
vector_radix_sort = coop._block.make_radix_sort_keys(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    temp_storage=vector_sort_temp_storage,
)
vector_topk_max = coop._block.make_topk_max_keys(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    temp_storage=vector_topk_temp_storage,
)
fragment_radix_sort = coop._block.make_radix_sort_keys(
    Int32,
    threads_per_block=BLOCK_THREADS,
    temp_storage=fragment_sort_temp_storage,
)


@cute.kernel
def _mixed_payload_factory_kernel(
    vector_keys_in: cute.Tensor,
    fragment_keys_in: cute.Tensor,
    sorted_vector_keys_out: cute.Tensor,
    top_vector_keys_out: cute.Tensor,
    sorted_fragment_keys_out: cute.Tensor,
    topk_k: cutlass.Int32,
    num_valid: cutlass.Int32,
    begin_bit: cutlass.Int32,
    end_bit: cutlass.Int32,
    items_per_thread: cutlass.Constexpr,
):
    cute_tidx, _, _ = cute.arch.thread_idx()
    fragment_base = cute_tidx * items_per_thread

    vector_keys = vector_load(vector_keys_in)
    fragment_keys = cute.make_rmem_tensor((items_per_thread,), Int32)
    for item in cutlass.range_constexpr(items_per_thread):
        fragment_keys[item] = fragment_keys_in[fragment_base + item]

    sorted_vector_keys = vector_radix_sort(
        vector_keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=False,
    )
    top_vector_keys = vector_topk_max(
        vector_keys,
        topk_k,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    sorted_fragment_keys = fragment_radix_sort(
        fragment_keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=False,
    )

    vector_store(sorted_vector_keys_out, sorted_vector_keys)
    vector_store(top_vector_keys_out, top_vector_keys)
    for item in cutlass.range_constexpr(items_per_thread):
        sorted_fragment_keys_out[fragment_base + item] = sorted_fragment_keys[item]


@cute.jit
def _run_mixed_payload_factory(
    vector_keys_in: cute.Tensor,
    fragment_keys_in: cute.Tensor,
    sorted_vector_keys_out: cute.Tensor,
    top_vector_keys_out: cute.Tensor,
    sorted_fragment_keys_out: cute.Tensor,
    topk_k: cutlass.Int32,
    num_valid: cutlass.Int32,
    begin_bit: cutlass.Int32,
    end_bit: cutlass.Int32,
):
    _mixed_payload_factory_kernel(
        vector_keys_in,
        fragment_keys_in,
        sorted_vector_keys_out,
        top_vector_keys_out,
        sorted_fragment_keys_out,
        topk_k,
        num_valid,
        begin_bit,
        end_bit,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
begin_bit = 0
end_bit = 8
vector_keys_host = torch.tensor(
    [((idx * 37 + (idx % 13) * 5) % 251) for idx in range(TOTAL_ITEMS)],
    dtype=torch.int32,
)
fragment_keys_host = torch.tensor(
    [((idx * 19 + 7) % 239) for idx in range(TOTAL_ITEMS)],
    dtype=torch.int32,
)
vector_keys_in = vector_keys_host.cuda()
fragment_keys_in = fragment_keys_host.cuda()
sorted_vector_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
top_vector_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
sorted_fragment_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_mixed_payload_factory(
    from_dlpack(vector_keys_in),
    from_dlpack(fragment_keys_in),
    from_dlpack(sorted_vector_keys_out),
    from_dlpack(top_vector_keys_out),
    from_dlpack(sorted_fragment_keys_out),
    cutlass.Int32(TOPK_K),
    cutlass.Int32(TOPK_VALID_ITEMS),
    cutlass.Int32(begin_bit),
    cutlass.Int32(end_bit),
)
torch.cuda.synchronize()

expected_vector_sorted = _expected_radix_keys(
    vector_keys_host,
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=False,
    torch=torch,
)
expected_vector_top = _expected_radix_keys(
    vector_keys_host[:TOPK_VALID_ITEMS],
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=True,
    torch=torch,
)
expected_fragment_sorted = _expected_radix_keys(
    fragment_keys_host,
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=False,
    torch=torch,
)

torch.testing.assert_close(
    sorted_vector_keys_out.cpu(),
    expected_vector_sorted,
    atol=0,
    rtol=0,
)
_assert_topk_keys_unordered(
    top_vector_keys_out[:TOPK_K],
    expected_vector_top[:TOPK_K],
    torch=torch,
)
torch.testing.assert_close(
    sorted_fragment_keys_out.cpu(),
    expected_fragment_sorted,
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
    "factory_scopes": [
        vector_load.scope,
        vector_store.scope,
        vector_radix_sort.scope,
        vector_topk_max.scope,
        fragment_radix_sort.scope,
    ],
    "topk_valid_items": int(TOPK_VALID_ITEMS),
    "sorted_vector_keys": [
        int(x) for x in sorted_vector_keys_out[:TOPK_K].cpu().tolist()
    ],
    "top_vector_keys": [
        int(x) for x in top_vector_keys_out[:TOPK_K].cpu().tolist()
    ],
    "sorted_fragment_keys": [
        int(x) for x in sorted_fragment_keys_out[:TOPK_K].cpu().tolist()
    ],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_cutlass_api_load_store_factory_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str,
) -> None:
    """Write a source-backed CuTe load/store factory smoke for the CUTLASS API."""

    block_scope_expr = "coop._block"
    value_dtype_expr = "Int32"
    valid_type_expr = "cutlass.Int32"
    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime

cutlass, cute, torch, from_dlpack, _root_coop, Int32 = require_runtime(
    include_int32=True,
)
coop = importlib.import_module({api_module!r})

block_scope = {block_scope_expr}

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
VALID_ITEMS = TOTAL_ITEMS - 5
FACTORY_DEFAULT_VALID_ITEMS = TOTAL_ITEMS
OOB_DEFAULT = -7
STORE_SENTINEL = -99

load_direct = block_scope.make_load(
    {value_dtype_expr},
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    valid_items=FACTORY_DEFAULT_VALID_ITEMS,
    oob_default={value_dtype_expr}(OOB_DEFAULT),
)
load_striped = block_scope.make_load(
    {value_dtype_expr},
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
)
exclusive_sum = block_scope.make_exclusive_sum(
    {value_dtype_expr},
    threads_per_block=BLOCK_THREADS,
)
store_direct = block_scope.make_store(
    {value_dtype_expr},
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    valid_items=FACTORY_DEFAULT_VALID_ITEMS,
)
store_striped = block_scope.make_store(
    {value_dtype_expr},
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
)


@cute.kernel
def _cutlass_api_load_store_factory_kernel(
    values_in: cute.Tensor,
    direct_copy_out: cute.Tensor,
    striped_copy_out: cute.Tensor,
    partial_copy_out: cute.Tensor,
    valid_store_out: cute.Tensor,
    exclusive_out: cute.Tensor,
    valid_items: {valid_type_expr},
):
    full_items = load_direct(values_in)
    store_direct(direct_copy_out, full_items)

    striped_items = load_striped(values_in)
    store_striped(striped_copy_out, striped_items)

    partial_items = load_direct(
        values_in,
        valid_items=valid_items,
        oob_default={value_dtype_expr}(OOB_DEFAULT),
    )
    store_direct(partial_copy_out, partial_items)
    store_direct(valid_store_out, partial_items, valid_items=valid_items)

    prefix_items = exclusive_sum(full_items)
    store_direct(exclusive_out, prefix_items)


@cute.jit
def _run_cutlass_api_load_store_factory(
    values_in: cute.Tensor,
    direct_copy_out: cute.Tensor,
    striped_copy_out: cute.Tensor,
    partial_copy_out: cute.Tensor,
    valid_store_out: cute.Tensor,
    exclusive_out: cute.Tensor,
    valid_items: {valid_type_expr},
):
    _cutlass_api_load_store_factory_kernel(
        values_in,
        direct_copy_out,
        striped_copy_out,
        partial_copy_out,
        valid_store_out,
        exclusive_out,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
values_host = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.int32)
values_in = values_host.cuda()
direct_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
striped_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
partial_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
valid_store_out = torch.full(
    (TOTAL_ITEMS,),
    STORE_SENTINEL,
    dtype=torch.int32,
    device="cuda",
)
exclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_cutlass_api_load_store_factory(
    from_dlpack(values_in),
    from_dlpack(direct_copy_out),
    from_dlpack(striped_copy_out),
    from_dlpack(partial_copy_out),
    from_dlpack(valid_store_out),
    from_dlpack(exclusive_out),
    {valid_type_expr}(VALID_ITEMS),
)
torch.cuda.synchronize()

expected_partial = values_host.clone()
expected_partial[VALID_ITEMS:] = OOB_DEFAULT
expected_valid_store = torch.full_like(values_host, STORE_SENTINEL)
expected_valid_store[:VALID_ITEMS] = values_host[:VALID_ITEMS]
expected_exclusive = torch.cumsum(values_host.to(torch.int64), dim=0).to(torch.int32)
expected_exclusive = expected_exclusive - values_host

torch.testing.assert_close(direct_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(striped_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(partial_copy_out.cpu(), expected_partial, atol=0, rtol=0)
torch.testing.assert_close(valid_store_out.cpu(), expected_valid_store, atol=0, rtol=0)
torch.testing.assert_close(exclusive_out.cpu(), expected_exclusive, atol=0, rtol=0)

print(json.dumps({{
    "api_module": coop.__name__,
    "factory_scopes": [
        load_direct.scope,
        load_striped.scope,
        exclusive_sum.scope,
        store_direct.scope,
        store_striped.scope,
    ],
    "direct_copy": [int(x) for x in direct_copy_out[:4].cpu().tolist()],
    "striped_copy": [int(x) for x in striped_copy_out[:4].cpu().tolist()],
    "partial_copy": [int(x) for x in partial_copy_out[-6:].cpu().tolist()],
    "valid_store": [int(x) for x in valid_store_out[-6:].cpu().tolist()],
    "exclusive": [int(x) for x in exclusive_out[:6].cpu().tolist()],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_cutlass_prims_array_load_store_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str,
) -> None:
    """Write a Prims array load/store smoke for the CUTLASS API."""

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()

coop = importlib.import_module({api_module!r})
prims_block = coop._block
prims_warp = coop._warp
prims_payload = coop.Payload.PRIMS

BLOCK_THREADS = 32
WARP_THREADS = 32
ITEMS_PER_THREAD = 3
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
DYNAMIC_ITEMS_PER_THREAD = 2
DYNAMIC_TOTAL_ITEMS = BLOCK_THREADS * DYNAMIC_ITEMS_PER_THREAD
VALID_ITEMS = TOTAL_ITEMS - 5
OOB_DEFAULT = -17
PARTIAL_OFFSET = 3
PARTIAL_STORE_SENTINEL = -101
assert PARTIAL_OFFSET + VALID_ITEMS <= TOTAL_ITEMS

block_load_direct = prims_block.make_load(
    cutlass.Int32,
    payload=prims_payload,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
block_store_direct = prims_block.make_store(
    cutlass.Int32,
    payload=prims_payload,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
warp_load_direct = prims_warp.make_load(
    cutlass.Int32,
    payload=prims_payload,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
warp_store_direct = prims_warp.make_store(
    cutlass.Int32,
    payload=prims_payload,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
array_load_direct = coop._block.make_load(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
array_store_direct = coop._block.make_store(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
warp_array_load_direct = coop._warp.make_load(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
warp_array_store_direct = coop._warp.make_store(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
)
root_payload_alias_load = coop._block.make_load(
    cutlass.Int32,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
    payload=prims_payload,
    launch_metadata={{"threads_per_block": BLOCK_THREADS}},
)
root_payload_alias_store = coop._block.make_store(
    cutlass.Int32,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
    payload=prims_payload,
    launch_config={{"block": (BLOCK_THREADS, 1, 1)}},
)
warp_root_load = coop._warp.make_load(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
    payload=prims_payload,
)
warp_root_store = coop._warp.make_store(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
    payload=prims_payload,
)
root_implicit_control_load = coop._block.make_load(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    offset=0,
    bounds_check=True,
)
root_implicit_control_store = coop._block.make_store(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    offset=0,
    bounds_check=True,
)
warp_implicit_control_load = coop._warp.make_load(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    offset=0,
    bounds_check=True,
)
warp_implicit_control_store = coop._warp.make_store(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    offset=0,
    bounds_check=True,
)


@cute.kernel
def _cutlass_api_prims_scoped_load_kernel(
    values_in: cute.Tensor,
    striped_copy_out: cute.Tensor,
    metadata_alias_copy_out: cute.Tensor,
    root_payload_alias_copy_out: cute.Tensor,
    root_payload_factory_alias_copy_out: cute.Tensor,
    warp_striped_copy_out: cute.Tensor,
    warp_root_copy_out: cute.Tensor,
    warp_root_factory_copy_out: cute.Tensor,
    partial_copy_out: cute.Tensor,
    partial_valid_store_out: cute.Tensor,
    factory_copy_out: cute.Tensor,
    warp_factory_copy_out: cute.Tensor,
    root_array_copy_out: cute.Tensor,
    root_warp_array_copy_out: cute.Tensor,
    factory_array_copy_out: cute.Tensor,
    warp_factory_array_copy_out: cute.Tensor,
    root_implicit_control_copy_out: cute.Tensor,
    root_implicit_control_factory_copy_out: cute.Tensor,
    root_implicit_control_prefix_out: cute.Tensor,
    warp_implicit_control_copy_out: cute.Tensor,
    warp_implicit_control_factory_copy_out: cute.Tensor,
    warp_implicit_control_prefix_out: cute.Tensor,
    valid_items: cutlass.Int32,
    offset: cutlass.Int32,
):
    striped_items = prims_block.load(
        values_in,
        payload=prims_payload,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        algorithm="striped",
        threads_per_block=BLOCK_THREADS,
    )
    metadata_alias_items = prims_block.load(
        values_in,
        payload=prims_payload,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        algorithm="striped",
        launch_metadata={{"threads_per_block": BLOCK_THREADS}},
    )
    root_payload_alias_items = coop._block.load(
        values_in,
        payload=prims_payload,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        algorithm="striped",
        launch_meta={{"block_dim": (BLOCK_THREADS, 1, 1)}},
    )
    root_payload_factory_alias_items = root_payload_alias_load(values_in)
    root_implicit_control_items = coop._block.load(
        values_in,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        offset=0,
        bounds_check=True,
        threads_per_block=BLOCK_THREADS,
    )
    root_implicit_control_factory_items = root_implicit_control_load(values_in)
    root_implicit_control_prefix_items = coop._block.exclusive_sum(
        root_implicit_control_items,
        threads_per_block=BLOCK_THREADS,
    )
    warp_striped_items = prims_warp.load(
        values_in,
        payload=prims_payload,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        algorithm="striped",
        threads_in_warp=WARP_THREADS,
    )
    warp_root_items = coop._warp.load(
        values_in,
        payload=prims_payload,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        algorithm="striped",
        threads_in_warp=WARP_THREADS,
    )
    warp_root_factory_items = warp_root_load(values_in)
    warp_implicit_control_items = coop._warp.load(
        values_in,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        offset=0,
        bounds_check=True,
        threads_in_warp=WARP_THREADS,
    )
    warp_implicit_control_factory_items = warp_implicit_control_load(values_in)
    warp_implicit_control_prefix_items = coop._warp.exclusive_sum(
        warp_implicit_control_items,
        threads_in_warp=WARP_THREADS,
    )
    partial_items = prims_block.load(
        values_in,
        payload=prims_payload,
        items_per_thread=ITEMS_PER_THREAD,
        dtype=cutlass.Int32,
        offset=offset,
        valid_items=valid_items,
        oob_default=cutlass.Int32(OOB_DEFAULT),
    )
    factory_items = block_load_direct(values_in)
    warp_factory_items = warp_load_direct(values_in)
    coop._block.store(
        striped_copy_out,
        striped_items,
        payload=prims_payload,
        algorithm="striped",
        threads_per_block=BLOCK_THREADS,
    )
    prims_block.store(
        metadata_alias_copy_out,
        metadata_alias_items,
        payload=prims_payload,
        algorithm="striped",
        launch_config={{"block": (BLOCK_THREADS, 1, 1)}},
    )
    coop._block.store(
        root_payload_alias_copy_out,
        root_payload_alias_items,
        payload=prims_payload,
        algorithm="striped",
        launch={{"threads_per_block": BLOCK_THREADS}},
    )
    root_payload_alias_store(
        root_payload_factory_alias_copy_out,
        root_payload_factory_alias_items,
    )
    coop._block.store(
        root_implicit_control_copy_out,
        root_implicit_control_items,
        offset=0,
        bounds_check=True,
        threads_per_block=BLOCK_THREADS,
    )
    root_implicit_control_store(
        root_implicit_control_factory_copy_out,
        root_implicit_control_factory_items,
    )
    coop._block.store(
        root_implicit_control_prefix_out,
        root_implicit_control_prefix_items,
        offset=0,
        bounds_check=True,
        threads_per_block=BLOCK_THREADS,
    )
    coop._warp.store(
        warp_striped_copy_out,
        warp_striped_items,
        payload=prims_payload,
        algorithm="striped",
        threads_in_warp=WARP_THREADS,
    )
    coop._warp.store(
        warp_root_copy_out,
        warp_root_items,
        payload=prims_payload,
        algorithm="striped",
        threads_in_warp=WARP_THREADS,
    )
    warp_root_store(
        warp_root_factory_copy_out,
        warp_root_factory_items,
    )
    coop._warp.store(
        warp_implicit_control_copy_out,
        warp_implicit_control_items,
        offset=0,
        bounds_check=True,
        threads_in_warp=WARP_THREADS,
    )
    warp_implicit_control_store(
        warp_implicit_control_factory_copy_out,
        warp_implicit_control_factory_items,
    )
    coop._warp.store(
        warp_implicit_control_prefix_out,
        warp_implicit_control_prefix_items,
        offset=0,
        bounds_check=True,
        threads_in_warp=WARP_THREADS,
    )
    coop._block.store(
        partial_copy_out,
        partial_items,
        payload=prims_payload,
        threads_per_block=BLOCK_THREADS,
    )
    coop._block.store(
        partial_valid_store_out,
        partial_items,
        payload=prims_payload,
        offset=offset,
        valid_items=valid_items,
        threads_per_block=BLOCK_THREADS,
    )
    block_store_direct(factory_copy_out, factory_items)
    warp_store_direct(warp_factory_copy_out, warp_factory_items)

    root_array = cutlass.make_array_view(values_in)
    root_array_out = cutlass.make_array_view(root_array_copy_out)
    assert isinstance(root_array, cutlass.Array)
    assert isinstance(root_array_out, cutlass.Array)
    root_array_items = coop._block.load(
        root_array,
        items_per_thread=ITEMS_PER_THREAD,
        threads_per_block=BLOCK_THREADS,
    )
    coop._block.store(
        root_array_out,
        root_array_items,
        threads_per_block=BLOCK_THREADS,
    )

    root_warp_array = cutlass.make_array_view(values_in)
    root_warp_array_out = cutlass.make_array_view(root_warp_array_copy_out)
    root_warp_array_items = coop._warp.load(
        root_warp_array,
        items_per_thread=ITEMS_PER_THREAD,
        threads_in_warp=WARP_THREADS,
    )
    coop._warp.store(
        root_warp_array_out,
        root_warp_array_items,
        threads_in_warp=WARP_THREADS,
    )

    factory_array = cutlass.make_array_view(values_in)
    factory_array_out = cutlass.make_array_view(factory_array_copy_out)
    factory_array_items = array_load_direct(factory_array)
    array_store_direct(factory_array_out, factory_array_items)

    warp_factory_array = cutlass.make_array_view(values_in)
    warp_factory_array_out = cutlass.make_array_view(warp_factory_array_copy_out)
    warp_factory_array_items = warp_array_load_direct(warp_factory_array)
    warp_array_store_direct(warp_factory_array_out, warp_factory_array_items)


@cute.jit
def _run_cutlass_api_prims_scoped_load(
    values_in: cute.Tensor,
    striped_copy_out: cute.Tensor,
    metadata_alias_copy_out: cute.Tensor,
    root_payload_alias_copy_out: cute.Tensor,
    root_payload_factory_alias_copy_out: cute.Tensor,
    warp_striped_copy_out: cute.Tensor,
    warp_root_copy_out: cute.Tensor,
    warp_root_factory_copy_out: cute.Tensor,
    partial_copy_out: cute.Tensor,
    partial_valid_store_out: cute.Tensor,
    factory_copy_out: cute.Tensor,
    warp_factory_copy_out: cute.Tensor,
    root_array_copy_out: cute.Tensor,
    root_warp_array_copy_out: cute.Tensor,
    factory_array_copy_out: cute.Tensor,
    warp_factory_array_copy_out: cute.Tensor,
    root_implicit_control_copy_out: cute.Tensor,
    root_implicit_control_factory_copy_out: cute.Tensor,
    root_implicit_control_prefix_out: cute.Tensor,
    warp_implicit_control_copy_out: cute.Tensor,
    warp_implicit_control_factory_copy_out: cute.Tensor,
    warp_implicit_control_prefix_out: cute.Tensor,
    valid_items: cutlass.Int32,
    offset: cutlass.Int32,
):
    _cutlass_api_prims_scoped_load_kernel(
        values_in,
        striped_copy_out,
        metadata_alias_copy_out,
        root_payload_alias_copy_out,
        root_payload_factory_alias_copy_out,
        warp_striped_copy_out,
        warp_root_copy_out,
        warp_root_factory_copy_out,
        partial_copy_out,
        partial_valid_store_out,
        factory_copy_out,
        warp_factory_copy_out,
        root_array_copy_out,
        root_warp_array_copy_out,
        factory_array_copy_out,
        warp_factory_array_copy_out,
        root_implicit_control_copy_out,
        root_implicit_control_factory_copy_out,
        root_implicit_control_prefix_out,
        warp_implicit_control_copy_out,
        warp_implicit_control_factory_copy_out,
        warp_implicit_control_prefix_out,
        valid_items,
        offset,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


@cute.kernel
def _dynamic_offset_array_copy_kernel(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
    literal_values_out: cute.Tensor,
    offset: cutlass.Int32,
):
    input_array = cutlass.make_array_view(values_in)
    output_array = cutlass.make_array_view(values_out)
    literal_output_array = cutlass.make_array_view(literal_values_out)
    assert input_array.align == 4
    assert output_array.align == 4
    assert literal_output_array.align == 4
    literal_items = coop._block.load(
        input_array,
        items_per_thread=DYNAMIC_ITEMS_PER_THREAD,
        offset=0,
        threads_per_block=BLOCK_THREADS,
    )
    coop._block.store(
        literal_output_array,
        literal_items,
        offset=0,
        threads_per_block=BLOCK_THREADS,
    )
    thread_items = coop._block.load(
        input_array,
        items_per_thread=DYNAMIC_ITEMS_PER_THREAD,
        offset=offset,
        threads_per_block=BLOCK_THREADS,
    )
    coop._block.store(
        output_array,
        thread_items,
        offset=offset,
        threads_per_block=BLOCK_THREADS,
    )


@cute.jit
def _run_dynamic_offset_array_copy(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
    literal_values_out: cute.Tensor,
    offset: cutlass.Int32,
):
    _dynamic_offset_array_copy_kernel(
        values_in,
        values_out,
        literal_values_out,
        offset,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
values_host = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.int32)
values_in = values_host.cuda()
dynamic_values_storage_host = torch.arange(
    0,
    DYNAMIC_TOTAL_ITEMS + PARTIAL_OFFSET + 1,
    dtype=torch.int32,
)
dynamic_values_host = dynamic_values_storage_host[1:]
dynamic_values_storage = dynamic_values_storage_host.cuda()
dynamic_values_in = dynamic_values_storage[1:]
dynamic_offset_copy_storage = torch.full(
    (DYNAMIC_TOTAL_ITEMS + PARTIAL_OFFSET + 1,),
    PARTIAL_STORE_SENTINEL,
    dtype=torch.int32,
    device="cuda",
)
dynamic_offset_copy_out = dynamic_offset_copy_storage[1:]
literal_offset_copy_storage = torch.full(
    (DYNAMIC_TOTAL_ITEMS + PARTIAL_OFFSET + 1,),
    PARTIAL_STORE_SENTINEL,
    dtype=torch.int32,
    device="cuda",
)
literal_offset_copy_out = literal_offset_copy_storage[1:]
striped_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
metadata_alias_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
root_payload_alias_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
root_payload_factory_alias_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
warp_striped_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
warp_root_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
warp_root_factory_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
partial_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
partial_valid_store_out = torch.full(
    (TOTAL_ITEMS + PARTIAL_OFFSET,),
    PARTIAL_STORE_SENTINEL,
    dtype=torch.int32,
    device="cuda",
)
factory_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
warp_factory_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
root_array_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
root_warp_array_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
factory_array_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
warp_factory_array_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
root_implicit_control_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
root_implicit_control_factory_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
root_implicit_control_prefix_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
warp_implicit_control_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
warp_implicit_control_factory_copy_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)
warp_implicit_control_prefix_out = torch.zeros(
    (TOTAL_ITEMS,),
    dtype=torch.int32,
    device="cuda",
)

_run_cutlass_api_prims_scoped_load(
    from_dlpack(values_in),
    from_dlpack(striped_copy_out),
    from_dlpack(metadata_alias_copy_out),
    from_dlpack(root_payload_alias_copy_out),
    from_dlpack(root_payload_factory_alias_copy_out),
    from_dlpack(warp_striped_copy_out),
    from_dlpack(warp_root_copy_out),
    from_dlpack(warp_root_factory_copy_out),
    from_dlpack(partial_copy_out),
    from_dlpack(partial_valid_store_out),
    from_dlpack(factory_copy_out),
    from_dlpack(warp_factory_copy_out),
    from_dlpack(root_array_copy_out),
    from_dlpack(root_warp_array_copy_out),
    from_dlpack(factory_array_copy_out),
    from_dlpack(warp_factory_array_copy_out),
    from_dlpack(root_implicit_control_copy_out),
    from_dlpack(root_implicit_control_factory_copy_out),
    from_dlpack(root_implicit_control_prefix_out),
    from_dlpack(warp_implicit_control_copy_out),
    from_dlpack(warp_implicit_control_factory_copy_out),
    from_dlpack(warp_implicit_control_prefix_out),
    cutlass.Int32(VALID_ITEMS),
    cutlass.Int32(PARTIAL_OFFSET),
)
_run_dynamic_offset_array_copy(
    from_dlpack(dynamic_values_in),
    from_dlpack(dynamic_offset_copy_out),
    from_dlpack(literal_offset_copy_out),
    cutlass.Int32(PARTIAL_OFFSET),
)
torch.cuda.synchronize()

torch.testing.assert_close(striped_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(metadata_alias_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(
    root_payload_alias_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    root_payload_factory_alias_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(warp_striped_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(
    warp_root_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    warp_root_factory_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
partial_values = values_host[
    PARTIAL_OFFSET : PARTIAL_OFFSET + VALID_ITEMS
]
expected_partial = torch.full((TOTAL_ITEMS,), OOB_DEFAULT, dtype=torch.int32)
expected_partial[:VALID_ITEMS] = partial_values
torch.testing.assert_close(partial_copy_out.cpu(), expected_partial, atol=0, rtol=0)
expected_partial_valid_store = torch.full(
    (TOTAL_ITEMS + PARTIAL_OFFSET,),
    PARTIAL_STORE_SENTINEL,
    dtype=torch.int32,
)
expected_partial_valid_store[
    PARTIAL_OFFSET : PARTIAL_OFFSET + VALID_ITEMS
] = partial_values
torch.testing.assert_close(
    partial_valid_store_out.cpu(),
    expected_partial_valid_store,
    atol=0,
    rtol=0,
)
expected_dynamic_offset_copy = torch.full(
    (DYNAMIC_TOTAL_ITEMS + PARTIAL_OFFSET,),
    PARTIAL_STORE_SENTINEL,
    dtype=torch.int32,
)
expected_dynamic_offset_copy[PARTIAL_OFFSET:] = dynamic_values_host[PARTIAL_OFFSET:]
torch.testing.assert_close(
    dynamic_offset_copy_out.cpu(),
    expected_dynamic_offset_copy,
    atol=0,
    rtol=0,
)
assert int(dynamic_offset_copy_storage[0].cpu().item()) == PARTIAL_STORE_SENTINEL
expected_literal_offset_copy = torch.full(
    (DYNAMIC_TOTAL_ITEMS + PARTIAL_OFFSET,),
    PARTIAL_STORE_SENTINEL,
    dtype=torch.int32,
)
expected_literal_offset_copy[:DYNAMIC_TOTAL_ITEMS] = dynamic_values_host[
    :DYNAMIC_TOTAL_ITEMS
]
torch.testing.assert_close(
    literal_offset_copy_out.cpu(),
    expected_literal_offset_copy,
    atol=0,
    rtol=0,
)
assert int(literal_offset_copy_storage[0].cpu().item()) == PARTIAL_STORE_SENTINEL
torch.testing.assert_close(factory_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(warp_factory_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(root_array_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(
    root_warp_array_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(factory_array_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(
    warp_factory_array_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    root_implicit_control_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    root_implicit_control_factory_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    warp_implicit_control_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    warp_implicit_control_factory_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
expected_prefix = torch.cumsum(values_host.to(torch.int64), dim=0).to(torch.int32)
expected_prefix = expected_prefix - values_host
torch.testing.assert_close(
    root_implicit_control_prefix_out.cpu(),
    expected_prefix,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    warp_implicit_control_prefix_out.cpu(),
    expected_prefix,
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
    "factory_scopes": [
        block_load_direct.scope,
        block_store_direct.scope,
        warp_load_direct.scope,
        warp_store_direct.scope,
    ],
    "array_factory_scopes": [
        array_load_direct.scope,
        array_store_direct.scope,
        warp_array_load_direct.scope,
        warp_array_store_direct.scope,
    ],
    "load_module": prims_block.load.__module__,
    "store_module": coop._block.store.__module__,
    "warp_load_module": prims_warp.load.__module__,
    "warp_store_module": coop._warp.store.__module__,
    "factory_copy": [int(x) for x in factory_copy_out[:8].cpu().tolist()],
    "warp_factory_copy": [
        int(x) for x in warp_factory_copy_out[:8].cpu().tolist()
    ],
    "root_array_copy": [int(x) for x in root_array_copy_out[:8].cpu().tolist()],
    "root_warp_array_copy": [
        int(x) for x in root_warp_array_copy_out[:8].cpu().tolist()
    ],
    "factory_array_copy": [
        int(x) for x in factory_array_copy_out[:8].cpu().tolist()
    ],
    "warp_factory_array_copy": [
        int(x) for x in warp_factory_array_copy_out[:8].cpu().tolist()
    ],
    "root_implicit_control_copy": [
        int(x) for x in root_implicit_control_copy_out[:8].cpu().tolist()
    ],
    "root_implicit_control_factory_copy": [
        int(x)
        for x in root_implicit_control_factory_copy_out[:8].cpu().tolist()
    ],
    "root_implicit_control_prefix": [
        int(x) for x in root_implicit_control_prefix_out[:8].cpu().tolist()
    ],
    "warp_implicit_control_copy": [
        int(x) for x in warp_implicit_control_copy_out[:8].cpu().tolist()
    ],
    "warp_implicit_control_factory_copy": [
        int(x)
        for x in warp_implicit_control_factory_copy_out[:8].cpu().tolist()
    ],
    "warp_implicit_control_prefix": [
        int(x) for x in warp_implicit_control_prefix_out[:8].cpu().tolist()
    ],
    "striped_copy": [int(x) for x in striped_copy_out[:8].cpu().tolist()],
    "metadata_alias_copy": [
        int(x) for x in metadata_alias_copy_out[:8].cpu().tolist()
    ],
    "root_payload_alias_copy": [
        int(x) for x in root_payload_alias_copy_out[:8].cpu().tolist()
    ],
    "root_payload_factory_alias_copy": [
        int(x) for x in root_payload_factory_alias_copy_out[:8].cpu().tolist()
    ],
    "warp_striped_copy": [int(x) for x in warp_striped_copy_out[:8].cpu().tolist()],
    "warp_root_copy": [
        int(x) for x in warp_root_copy_out[:8].cpu().tolist()
    ],
    "warp_root_factory_copy": [
        int(x) for x in warp_root_factory_copy_out[:8].cpu().tolist()
    ],
    "partial_tail": [int(x) for x in partial_copy_out[-8:].cpu().tolist()],
    "partial_valid_store_tail": [
        int(x) for x in partial_valid_store_out[-8:].cpu().tolist()
    ],
    "dynamic_offset_copy": [
        int(x) for x in dynamic_offset_copy_out[:8].cpu().tolist()
    ],
    "literal_offset_copy": [
        int(x) for x in literal_offset_copy_out[:8].cpu().tolist()
    ],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_cutlass_api_warp_load_store_factory_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str,
) -> None:
    """Write a source-backed CuTe warp load/store factory smoke for the CUTLASS API."""

    warp_scope_expr = "coop._warp"
    value_dtype_expr = "Int32"
    valid_type_expr = "cutlass.Int32"
    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime

cutlass, cute, torch, from_dlpack, _root_coop, Int32 = require_runtime(
    include_int32=True,
)
coop = importlib.import_module({api_module!r})

warp_scope = {warp_scope_expr}

BLOCK_THREADS = 32
WARP_THREADS = 16
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
VALID_ITEMS = TOTAL_ITEMS - 5
FACTORY_DEFAULT_VALID_ITEMS = TOTAL_ITEMS
OOB_DEFAULT = -13
STORE_SENTINEL = -101

load_direct = warp_scope.make_load(
    {value_dtype_expr},
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    valid_items=FACTORY_DEFAULT_VALID_ITEMS,
    oob_default={value_dtype_expr}(OOB_DEFAULT),
)
load_striped = warp_scope.make_load(
    {value_dtype_expr},
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
)
exclusive_sum = warp_scope.make_exclusive_sum(
    {value_dtype_expr},
    threads_in_warp=WARP_THREADS,
)
store_direct = warp_scope.make_store(
    {value_dtype_expr},
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    valid_items=FACTORY_DEFAULT_VALID_ITEMS,
)
store_striped = warp_scope.make_store(
    {value_dtype_expr},
    threads_in_warp=WARP_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    algorithm="striped",
)


@cute.kernel
def _cutlass_api_warp_load_store_factory_kernel(
    values_in: cute.Tensor,
    direct_copy_out: cute.Tensor,
    striped_copy_out: cute.Tensor,
    partial_copy_out: cute.Tensor,
    valid_store_out: cute.Tensor,
    exclusive_out: cute.Tensor,
    valid_items: {valid_type_expr},
):
    full_items = load_direct(values_in)
    store_direct(direct_copy_out, full_items)

    striped_items = load_striped(values_in)
    store_striped(striped_copy_out, striped_items)

    partial_items = load_direct(
        values_in,
        valid_items=valid_items,
        oob_default={value_dtype_expr}(OOB_DEFAULT),
    )
    store_direct(partial_copy_out, partial_items)
    store_direct(valid_store_out, partial_items, valid_items=valid_items)

    prefix_items = exclusive_sum(full_items)
    store_direct(exclusive_out, prefix_items)


@cute.jit
def _run_cutlass_api_warp_load_store_factory(
    values_in: cute.Tensor,
    direct_copy_out: cute.Tensor,
    striped_copy_out: cute.Tensor,
    partial_copy_out: cute.Tensor,
    valid_store_out: cute.Tensor,
    exclusive_out: cute.Tensor,
    valid_items: {valid_type_expr},
):
    _cutlass_api_warp_load_store_factory_kernel(
        values_in,
        direct_copy_out,
        striped_copy_out,
        partial_copy_out,
        valid_store_out,
        exclusive_out,
        valid_items,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


def _expected_thread_data_warp_prefix(values, *, torch):
    expected = torch.empty_like(values)
    tile_items = WARP_THREADS * ITEMS_PER_THREAD
    for tile_base in range(0, int(values.numel()), tile_items):
        tile = values[tile_base : tile_base + tile_items]
        prefix = torch.cumsum(tile.to(torch.int64), dim=0).to(torch.int32) - tile
        expected[tile_base : tile_base + tile_items] = prefix
    return expected


cutlass.cuda.initialize_cuda_context()
values_host = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.int32)
values_in = values_host.cuda()
direct_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
striped_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
partial_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
valid_store_out = torch.full(
    (TOTAL_ITEMS,),
    STORE_SENTINEL,
    dtype=torch.int32,
    device="cuda",
)
exclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_cutlass_api_warp_load_store_factory(
    from_dlpack(values_in),
    from_dlpack(direct_copy_out),
    from_dlpack(striped_copy_out),
    from_dlpack(partial_copy_out),
    from_dlpack(valid_store_out),
    from_dlpack(exclusive_out),
    {valid_type_expr}(VALID_ITEMS),
)
torch.cuda.synchronize()

expected_partial = values_host.clone()
expected_partial[VALID_ITEMS:] = OOB_DEFAULT
expected_valid_store = torch.full_like(values_host, STORE_SENTINEL)
expected_valid_store[:VALID_ITEMS] = values_host[:VALID_ITEMS]
expected_exclusive = _expected_thread_data_warp_prefix(values_host, torch=torch)

torch.testing.assert_close(direct_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(striped_copy_out.cpu(), values_host, atol=0, rtol=0)
torch.testing.assert_close(partial_copy_out.cpu(), expected_partial, atol=0, rtol=0)
torch.testing.assert_close(valid_store_out.cpu(), expected_valid_store, atol=0, rtol=0)
torch.testing.assert_close(exclusive_out.cpu(), expected_exclusive, atol=0, rtol=0)

print(json.dumps({{
    "api_module": coop.__name__,
    "factory_scopes": [
        load_direct.scope,
        load_striped.scope,
        exclusive_sum.scope,
        store_direct.scope,
        store_striped.scope,
    ],
    "direct_copy": [int(x) for x in direct_copy_out[:4].cpu().tolist()],
    "striped_copy": [int(x) for x in striped_copy_out[:4].cpu().tolist()],
    "partial_copy": [int(x) for x in partial_copy_out[-6:].cpu().tolist()],
    "valid_store": [int(x) for x in valid_store_out[-6:].cpu().tolist()],
    "exclusive": [int(x) for x in exclusive_out[:8].cpu().tolist()],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_cutlass_api_row_sum_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str,
) -> None:
    """Write a source-backed scalar row-reduce smoke for a CUTLASS API."""

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})

BLOCK_THREADS = 128
ROWS_PER_BLOCK = 1
WARPS_PER_ROW = 4

row_sum_temp_storage = coop._block.TempStorage(size_in_bytes=20, sharing="shared")


@cute.kernel
def _cutlass_api_row_sum_kernel(
    values_in: cute.Tensor,
    total_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    total = coop._block.row_sum(
        value,
        rows_per_block=ROWS_PER_BLOCK,
        warps_per_row=WARPS_PER_ROW,
        temp_storage=row_sum_temp_storage,
        launch_metadata={{"threads_per_block": BLOCK_THREADS}},
    )
    total_out[tidx] = total


@cute.jit
def _run_cutlass_api_row_sum(
    values_in: cute.Tensor,
    total_out: cute.Tensor,
):
    _cutlass_api_row_sum_kernel(
        values_in,
        total_out,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
values_host = torch.arange(BLOCK_THREADS, dtype=torch.float32)
values_in = values_host.cuda()
total_out = torch.zeros((BLOCK_THREADS,), dtype=torch.float32, device="cuda")

_run_cutlass_api_row_sum(from_dlpack(values_in), from_dlpack(total_out))
torch.cuda.synchronize()

expected_total = torch.full(
    (BLOCK_THREADS,),
    float(values_host.sum().item()),
    dtype=torch.float32,
)
torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)

print(json.dumps({{
    "api_module": coop.__name__,
    "primitive_module": coop._block.row_sum.__module__,
    "temp_storage_scope": row_sum_temp_storage.scope,
    "row_total": float(total_out[0].cpu().item()),
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_prims_api_vector_rank_merge_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str = "cuda.coop.cutlass",
    call_mode: str = "factory",
) -> None:
    """Write a source-backed CuTe/Prims rank/merge smoke for a CUTLASS API."""

    if call_mode not in {"factory", "direct"}:
        raise ValueError("call_mode must be 'factory' or 'direct'")

    block_scope_expr = "coop._block"

    if call_mode == "factory":
        primitive_setup = """
radix_rank = block_scope.make_radix_rank(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    begin_bit=FACTORY_DEFAULT_BEGIN_BIT,
    radix_bits=FACTORY_DEFAULT_RADIX_BITS,
    descending=True,
    temp_storage=radix_temp_storage,
)
merge_sort_keys = block_scope.make_merge_sort_keys(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    descending=True,
    valid_items=FACTORY_DEFAULT_VALID_ITEMS,
    oob_default=Int32(FACTORY_DEFAULT_MERGE_OOB_DEFAULT),
    temp_storage=merge_temp_storage,
)
merge_sort_pairs = block_scope.make_merge_sort_pairs(
    cutlass.Int32,
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    descending=True,
    valid_items=FACTORY_DEFAULT_VALID_ITEMS,
    oob_default=Int32(FACTORY_DEFAULT_MERGE_OOB_DEFAULT),
    temp_storage=merge_temp_storage,
)
"""
        primitive_calls = """
    ranks = radix_rank(
        keys_vec,
        begin_bit=RANK_BEGIN_BIT,
        end_bit=RANK_END_BIT,
        descending=False,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )
    merge_keys, merge_values = merge_sort_pairs(
        keys_vec,
        values_vec,
        valid_items=valid_items,
        oob_default=Int32(MERGE_OOB_DEFAULT),
    )
    merge_keys_only = merge_sort_keys(
        keys_vec,
        valid_items=valid_items,
        oob_default=Int32(MERGE_OOB_DEFAULT),
    )
"""
        scope_metadata = """
    "factory_scopes": [
        radix_rank.scope,
        merge_sort_keys.scope,
        merge_sort_pairs.scope,
    ],
"""
    else:
        primitive_setup = ""
        primitive_calls = """
    ranks = block_scope.radix_rank(
        keys_vec,
        threads_per_block=BLOCK_THREADS,
        begin_bit=RANK_BEGIN_BIT,
        end_bit=RANK_END_BIT,
        descending=False,
        exclusive_digit_prefix=exclusive_digit_prefix,
        temp_storage=radix_temp_storage,
    )
    merge_keys, merge_values = block_scope.merge_sort_pairs(
        keys_vec,
        values_vec,
        threads_per_block=BLOCK_THREADS,
        descending=True,
        valid_items=valid_items,
        oob_default=Int32(MERGE_OOB_DEFAULT),
        temp_storage=merge_temp_storage,
    )
    merge_keys_only = block_scope.merge_sort_keys(
        keys_vec,
        threads_per_block=BLOCK_THREADS,
        descending=True,
        valid_items=valid_items,
        oob_default=Int32(MERGE_OOB_DEFAULT),
        temp_storage=merge_temp_storage,
    )
"""
        scope_metadata = """
    "primitive_modules": [
        block_scope.radix_rank.__module__,
        block_scope.merge_sort_keys.__module__,
        block_scope.merge_sort_pairs.__module__,
        block_scope.store.__module__,
    ],
"""

    script_path.write_text(
        f"""
import json
import importlib
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_rank_merge import (
    MERGE_OOB_DEFAULT,
    _expected_merge_order_partial,
    _expected_radix_digit_prefix,
    _expected_radix_ranks,
)

cutlass, cute, torch, from_dlpack, _root_coop, Int32 = require_runtime(
    include_int32=True
)
coop = importlib.import_module({api_module!r})
block_scope = {block_scope_expr}

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
FACTORY_DEFAULT_BEGIN_BIT = 1
FACTORY_DEFAULT_RADIX_BITS = 3
RANK_BEGIN_BIT = 0
RANK_END_BIT = 4
FACTORY_DEFAULT_VALID_ITEMS = TOTAL_ITEMS
FACTORY_DEFAULT_MERGE_OOB_DEFAULT = MERGE_OOB_DEFAULT + 17

radix_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
merge_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
{primitive_setup}


@cute.kernel
def _prims_api_vector_rank_merge_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    merge_keys_out: cute.Tensor,
    merge_values_out: cute.Tensor,
    merge_keys_only_out: cute.Tensor,
    valid_items: cutlass.Int32,
    begin_bit: cutlass.Constexpr,
    end_bit: cutlass.Constexpr,
    items_per_thread: cutlass.Constexpr,
):
    keys_vec = block_scope.load(
        keys_in,
        payload=coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )
    values_vec = block_scope.load(
        values_in,
        payload=coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )
    exclusive_digit_prefix = coop.ThreadData(1, dtype=Int32)

{primitive_calls}

    block_scope.store(rank_out, ranks, payload=coop.Payload.PRIMS)
    block_scope.store(
        prefix_out, exclusive_digit_prefix, payload=coop.Payload.PRIMS
    )
    block_scope.store(merge_keys_out, merge_keys, payload=coop.Payload.PRIMS)
    block_scope.store(merge_values_out, merge_values, payload=coop.Payload.PRIMS)
    block_scope.store(
        merge_keys_only_out, merge_keys_only, payload=coop.Payload.PRIMS
    )


@cute.jit
def _run_prims_api_vector_rank_merge(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    rank_out: cute.Tensor,
    prefix_out: cute.Tensor,
    merge_keys_out: cute.Tensor,
    merge_values_out: cute.Tensor,
    merge_keys_only_out: cute.Tensor,
    valid_items: cutlass.Int32,
    begin_bit: cutlass.Constexpr,
    end_bit: cutlass.Constexpr,
):
    _prims_api_vector_rank_merge_kernel(
        keys_in,
        values_in,
        rank_out,
        prefix_out,
        merge_keys_out,
        merge_values_out,
        merge_keys_only_out,
        valid_items,
        begin_bit,
        end_bit,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
begin_bit = 0
end_bit = 4
valid_items = TOTAL_ITEMS - 9
keys_host = torch.tensor(
    [((idx * 11 + (idx % 7) * 5) % 53) - 26 for idx in range(TOTAL_ITEMS)],
    dtype=torch.int32,
)
values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 7 + 3
keys_in = keys_host.cuda()
values_in = values_host.cuda()
rank_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
prefix_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
merge_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
merge_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
merge_keys_only_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_prims_api_vector_rank_merge(
    from_dlpack(keys_in),
    from_dlpack(values_in),
    from_dlpack(rank_out),
    from_dlpack(prefix_out),
    from_dlpack(merge_keys_out),
    from_dlpack(merge_values_out),
    from_dlpack(merge_keys_only_out),
    cutlass.Int32(valid_items),
    begin_bit,
    end_bit,
)
torch.cuda.synchronize()

expected_ranks = _expected_radix_ranks(
    keys_host,
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=False,
    torch=torch,
)
expected_prefix = _expected_radix_digit_prefix(
    keys_host,
    begin_bit=begin_bit,
    end_bit=end_bit,
    descending=False,
    block_threads=BLOCK_THREADS,
    bins_per_thread=1,
    torch=torch,
)
expected_merge_keys, expected_merge_values = _expected_merge_order_partial(
    keys_host,
    values_host,
    descending=True,
    valid_items=valid_items,
    oob_default=MERGE_OOB_DEFAULT,
    torch=torch,
)

torch.testing.assert_close(rank_out.cpu(), expected_ranks, atol=0, rtol=0)
torch.testing.assert_close(prefix_out.cpu(), expected_prefix, atol=0, rtol=0)
torch.testing.assert_close(
    merge_keys_out[:valid_items].cpu(),
    expected_merge_keys[:valid_items],
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    merge_keys_only_out[:valid_items].cpu(),
    expected_merge_keys[:valid_items],
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    merge_values_out[:valid_items].cpu(),
    expected_merge_values[:valid_items],
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
{scope_metadata}
    "ranks": [int(x) for x in rank_out[:8].cpu().tolist()],
    "prefix": [int(x) for x in prefix_out[:8].cpu().tolist()],
    "merge_pairs": [
        [int(key), int(value)]
        for key, value in zip(
            merge_keys_out[:8].cpu().tolist(),
            merge_values_out[:8].cpu().tolist(),
            strict=True,
        )
    ],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_prims_api_vector_block_prefix_segment_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str = "cuda.coop.cutlass",
    call_mode: str = "factory",
) -> None:
    """Write a source-backed CuTe/Prims scan/reduce/segment smoke."""

    if call_mode not in {"factory", "direct"}:
        raise ValueError("call_mode must be 'factory' or 'direct'")

    block_scope_expr = "coop._block"

    if call_mode == "factory":
        primitive_setup = """
exclusive_sum = block_scope.make_exclusive_sum(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    temp_storage=scan_temp_storage,
)
inclusive_sum = block_scope.make_inclusive_sum(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    temp_storage=scan_temp_storage,
)
inclusive_scan_xor = block_scope.make_inclusive_scan(
    cutlass.Int32,
    scan_op="bit_xor",
    threads_per_block=BLOCK_THREADS,
    temp_storage=scan_temp_storage,
)
sum_values = block_scope.make_sum(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    algorithm=None,
    temp_storage=reduce_temp_storage,
)
reduce_xor = block_scope.make_reduce(
    cutlass.Int32,
    binary_op="bit_xor",
    threads_per_block=BLOCK_THREADS,
    algorithm=None,
    temp_storage=reduce_temp_storage,
)
adjacent_difference = block_scope.make_adjacent_difference(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    temp_storage=segment_temp_storage,
)
discontinuity = block_scope.make_discontinuity(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    temp_storage=segment_temp_storage,
)
shuffle_down = block_scope.make_shuffle(
    cutlass.Int32,
    block_shuffle_type=block_scope.BlockShuffleType.Down,
    distance=SHUFFLE_DISTANCE,
    threads_per_block=BLOCK_THREADS,
    temp_storage=shuffle_temp_storage,
)
"""
        primitive_calls = """
    sum_aggregate = coop.ThreadData(1, dtype=cutlass.Int32)
    xor_aggregate = coop.ThreadData(1, dtype=cutlass.Int32)
    shuffle_prefix = coop.ThreadData(1, dtype=cutlass.Int32)
    exclusive = exclusive_sum(values_vec)
    inclusive = inclusive_sum(values_vec, block_aggregate=sum_aggregate)
    xor_prefix = inclusive_scan_xor(
        values_vec,
        block_aggregate=xor_aggregate,
    )
    total = sum_values(values_vec)
    xor_total = reduce_xor(values_vec)
    local_sum = values_vec[0] + values_vec[1]
    local_xor = values_vec[0] ^ values_vec[1]
    partial_total = sum_values(local_sum, num_valid=num_valid)
    partial_xor_total = reduce_xor(local_xor, num_valid=num_valid)
    diff = adjacent_difference(segments_vec)
    diff_right = adjacent_difference(
        segments_vec,
        block_adjacent_difference_type=block_scope.BlockAdjacentDifferenceType.SubtractRight,
    )
    head = discontinuity(
        segments_vec,
        block_discontinuity_type=block_scope.BlockDiscontinuityType.HEADS,
    )
    tail = discontinuity(
        segments_vec,
        block_discontinuity_type=block_scope.BlockDiscontinuityType.TAILS,
    )
    head_pair, tail_pair = discontinuity(
        segments_vec,
        block_discontinuity_type=block_scope.BlockDiscontinuityType.HEADS_AND_TAILS,
    )
    shuffled = shuffle_down(values_vec, block_prefix=shuffle_prefix)
"""
        scope_metadata = """
    "factory_scopes": [
        exclusive_sum.scope,
        inclusive_sum.scope,
        inclusive_scan_xor.scope,
        sum_values.scope,
        reduce_xor.scope,
        adjacent_difference.scope,
        discontinuity.scope,
        shuffle_down.scope,
    ],
"""
    else:
        primitive_setup = ""
        primitive_calls = """
    sum_aggregate = coop.ThreadData(1, dtype=cutlass.Int32)
    xor_aggregate = coop.ThreadData(1, dtype=cutlass.Int32)
    shuffle_prefix = coop.ThreadData(1, dtype=cutlass.Int32)
    exclusive = block_scope.exclusive_sum(
        values_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=scan_temp_storage,
    )
    inclusive = block_scope.inclusive_sum(
        values_vec,
        block_aggregate=sum_aggregate,
        threads_per_block=BLOCK_THREADS,
        temp_storage=scan_temp_storage,
    )
    xor_prefix = block_scope.inclusive_scan(
        values_vec,
        scan_op="bit_xor",
        block_aggregate=xor_aggregate,
        threads_per_block=BLOCK_THREADS,
        temp_storage=scan_temp_storage,
    )
    total = block_scope.sum(
        values_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=reduce_temp_storage,
    )
    xor_total = block_scope.reduce(
        values_vec,
        binary_op="bit_xor",
        threads_per_block=BLOCK_THREADS,
        temp_storage=reduce_temp_storage,
    )
    local_sum = values_vec[0] + values_vec[1]
    local_xor = values_vec[0] ^ values_vec[1]
    partial_total = block_scope.sum(
        local_sum,
        num_valid=num_valid,
        threads_per_block=BLOCK_THREADS,
        temp_storage=reduce_temp_storage,
    )
    partial_xor_total = block_scope.reduce(
        local_xor,
        binary_op="bit_xor",
        num_valid=num_valid,
        threads_per_block=BLOCK_THREADS,
        temp_storage=reduce_temp_storage,
    )
    diff = block_scope.adjacent_difference(
        segments_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=segment_temp_storage,
    )
    diff_right = block_scope.adjacent_difference(
        segments_vec,
        block_adjacent_difference_type=block_scope.BlockAdjacentDifferenceType.SubtractRight,
        threads_per_block=BLOCK_THREADS,
        temp_storage=segment_temp_storage,
    )
    head = block_scope.discontinuity(
        segments_vec,
        block_discontinuity_type=block_scope.BlockDiscontinuityType.HEADS,
        threads_per_block=BLOCK_THREADS,
        temp_storage=segment_temp_storage,
    )
    tail = block_scope.discontinuity(
        segments_vec,
        block_discontinuity_type=block_scope.BlockDiscontinuityType.TAILS,
        threads_per_block=BLOCK_THREADS,
        temp_storage=segment_temp_storage,
    )
    head_pair, tail_pair = block_scope.discontinuity(
        segments_vec,
        block_discontinuity_type=block_scope.BlockDiscontinuityType.HEADS_AND_TAILS,
        threads_per_block=BLOCK_THREADS,
        temp_storage=segment_temp_storage,
    )
    shuffled = block_scope.shuffle_down(
        values_vec,
        distance=SHUFFLE_DISTANCE,
        block_prefix=shuffle_prefix,
        threads_per_block=BLOCK_THREADS,
        temp_storage=shuffle_temp_storage,
    )
"""
        scope_metadata = """
    "primitive_modules": [
        block_scope.exclusive_sum.__module__,
        block_scope.inclusive_sum.__module__,
        block_scope.inclusive_scan.__module__,
        block_scope.sum.__module__,
        block_scope.reduce.__module__,
        block_scope.adjacent_difference.__module__,
        block_scope.discontinuity.__module__,
        block_scope.shuffle_down.__module__,
        block_scope.store.__module__,
    ],
"""

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_block_prefix_segment import (
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    SHUFFLE_DISTANCE,
    TOTAL_ITEMS,
    _expected_diff_heads_tails,
    _expected_reduce,
    _expected_scan,
    _expected_shuffle_down,
    _expected_shuffle_prefix,
    _expected_xor_scan,
)

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})
block_scope = {block_scope_expr}

scan_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
reduce_temp_storage = block_scope.TempStorage(size_in_bytes=4096, sharing="shared")
segment_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
shuffle_temp_storage = block_scope.TempStorage(size_in_bytes=4096, sharing="shared")
{primitive_setup}


@cute.kernel
def _prims_api_vector_block_prefix_segment_kernel(
    values_in: cute.Tensor,
    segments_in: cute.Tensor,
    exclusive_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    xor_prefix_out: cute.Tensor,
    sum_aggregate_out: cute.Tensor,
    xor_aggregate_out: cute.Tensor,
    sum_out: cute.Tensor,
    xor_out: cute.Tensor,
    partial_sum_out: cute.Tensor,
    partial_xor_out: cute.Tensor,
    diff_out: cute.Tensor,
    diff_right_out: cute.Tensor,
    head_out: cute.Tensor,
    tail_out: cute.Tensor,
    head_pair_out: cute.Tensor,
    tail_pair_out: cute.Tensor,
    shuffle_down_out: cute.Tensor,
    shuffle_prefix_out: cute.Tensor,
    num_valid: cutlass.Int32,
    items_per_thread: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    values_prims = block_scope.load(
        values_in,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
        payload=_root_coop.Payload.PRIMS,
    )
    segments_prims = block_scope.load(
        segments_in,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
        payload=_root_coop.Payload.PRIMS,
    )
    values_vec = coop.ThreadData.from_fn(
        items_per_thread,
        lambda item: values_prims[item],
        dtype=cutlass.Int32,
    )
    segments_vec = coop.ThreadData.from_fn(
        items_per_thread,
        lambda item: segments_prims[item],
        dtype=cutlass.Int32,
    )

{primitive_calls}

    block_scope.store(exclusive_out, exclusive, payload=_root_coop.Payload.PRIMS)
    block_scope.store(inclusive_out, inclusive, payload=_root_coop.Payload.PRIMS)
    block_scope.store(xor_prefix_out, xor_prefix, payload=_root_coop.Payload.PRIMS)
    block_scope.store(
        sum_aggregate_out, sum_aggregate, payload=_root_coop.Payload.PRIMS
    )
    block_scope.store(
        xor_aggregate_out, xor_aggregate, payload=_root_coop.Payload.PRIMS
    )
    sum_out[tidx] = total
    xor_out[tidx] = xor_total
    if tidx == 0:
        partial_sum_out[0] = partial_total
        partial_xor_out[0] = partial_xor_total
    block_scope.store(diff_out, diff, payload=_root_coop.Payload.PRIMS)
    block_scope.store(diff_right_out, diff_right, payload=_root_coop.Payload.PRIMS)
    block_scope.store(head_out, head, payload=_root_coop.Payload.PRIMS)
    block_scope.store(tail_out, tail, payload=_root_coop.Payload.PRIMS)
    block_scope.store(head_pair_out, head_pair, payload=_root_coop.Payload.PRIMS)
    block_scope.store(tail_pair_out, tail_pair, payload=_root_coop.Payload.PRIMS)
    block_scope.store(
        shuffle_down_out, shuffled, payload=_root_coop.Payload.PRIMS
    )
    block_scope.store(
        shuffle_prefix_out, shuffle_prefix, payload=_root_coop.Payload.PRIMS
    )


@cute.jit
def _run_prims_api_vector_block_prefix_segment(
    values_in: cute.Tensor,
    segments_in: cute.Tensor,
    exclusive_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    xor_prefix_out: cute.Tensor,
    sum_aggregate_out: cute.Tensor,
    xor_aggregate_out: cute.Tensor,
    sum_out: cute.Tensor,
    xor_out: cute.Tensor,
    partial_sum_out: cute.Tensor,
    partial_xor_out: cute.Tensor,
    diff_out: cute.Tensor,
    diff_right_out: cute.Tensor,
    head_out: cute.Tensor,
    tail_out: cute.Tensor,
    head_pair_out: cute.Tensor,
    tail_pair_out: cute.Tensor,
    shuffle_down_out: cute.Tensor,
    shuffle_prefix_out: cute.Tensor,
    num_valid: cutlass.Int32,
):
    _prims_api_vector_block_prefix_segment_kernel(
        values_in,
        segments_in,
        exclusive_out,
        inclusive_out,
        xor_prefix_out,
        sum_aggregate_out,
        xor_aggregate_out,
        sum_out,
        xor_out,
        partial_sum_out,
        partial_xor_out,
        diff_out,
        diff_right_out,
        head_out,
        tail_out,
        head_pair_out,
        tail_pair_out,
        shuffle_down_out,
        shuffle_prefix_out,
        num_valid,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
values_host = ((torch.arange(TOTAL_ITEMS, dtype=torch.int64) % 17) + 1).to(
    torch.int32
)
NUM_VALID_THREADS = BLOCK_THREADS - 5
segments_host = torch.tensor(
    [((idx // 3) + (idx % 11 == 0)) for idx in range(TOTAL_ITEMS)],
    dtype=torch.int32,
)
values_in = values_host.cuda()
segments_in = segments_host.cuda()
exclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
inclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
xor_prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
sum_aggregate_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
xor_aggregate_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
sum_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
xor_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
partial_sum_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
partial_xor_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
diff_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
diff_right_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
head_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
tail_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
head_pair_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
tail_pair_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
shuffle_down_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
shuffle_prefix_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")

_run_prims_api_vector_block_prefix_segment(
    from_dlpack(values_in),
    from_dlpack(segments_in),
    from_dlpack(exclusive_out),
    from_dlpack(inclusive_out),
    from_dlpack(xor_prefix_out),
    from_dlpack(sum_aggregate_out),
    from_dlpack(xor_aggregate_out),
    from_dlpack(sum_out),
    from_dlpack(xor_out),
    from_dlpack(partial_sum_out),
    from_dlpack(partial_xor_out),
    from_dlpack(diff_out),
    from_dlpack(diff_right_out),
    from_dlpack(head_out),
    from_dlpack(tail_out),
    from_dlpack(head_pair_out),
    from_dlpack(tail_pair_out),
    from_dlpack(shuffle_down_out),
    from_dlpack(shuffle_prefix_out),
    cutlass.Int32(NUM_VALID_THREADS),
)
torch.cuda.synchronize()

expected_exclusive, expected_inclusive = _expected_scan(values_host, torch=torch)
expected_xor_prefix = _expected_xor_scan(values_host, torch=torch)
expected_sum, expected_xor = _expected_reduce(
    values_host,
    block_threads=BLOCK_THREADS,
    torch=torch,
)
expected_partial_sum, expected_partial_xor = _expected_reduce(
    values_host[: NUM_VALID_THREADS * ITEMS_PER_THREAD],
    block_threads=BLOCK_THREADS,
    torch=torch,
)
expected_partial_sum_root_only = torch.zeros_like(expected_partial_sum)
expected_partial_xor_root_only = torch.zeros_like(expected_partial_xor)
expected_partial_sum_root_only[0] = expected_partial_sum[0]
expected_partial_xor_root_only[0] = expected_partial_xor[0]
expected_diff, expected_diff_right, expected_head, expected_tail = (
    _expected_diff_heads_tails(
        segments_host,
        torch=torch,
    )
)
expected_shuffle_down = _expected_shuffle_down(
    values_host,
    distance=SHUFFLE_DISTANCE,
)
expected_shuffle_prefix = _expected_shuffle_prefix(
    values_host,
    block_threads=BLOCK_THREADS,
    torch=torch,
)

torch.testing.assert_close(exclusive_out.cpu(), expected_exclusive, atol=0, rtol=0)
torch.testing.assert_close(inclusive_out.cpu(), expected_inclusive, atol=0, rtol=0)
torch.testing.assert_close(xor_prefix_out.cpu(), expected_xor_prefix, atol=0, rtol=0)
torch.testing.assert_close(sum_aggregate_out.cpu(), expected_sum, atol=0, rtol=0)
torch.testing.assert_close(xor_aggregate_out.cpu(), expected_xor, atol=0, rtol=0)
torch.testing.assert_close(sum_out.cpu(), expected_sum, atol=0, rtol=0)
torch.testing.assert_close(xor_out.cpu(), expected_xor, atol=0, rtol=0)
torch.testing.assert_close(
    partial_sum_out.cpu(), expected_partial_sum_root_only, atol=0, rtol=0
)
torch.testing.assert_close(
    partial_xor_out.cpu(), expected_partial_xor_root_only, atol=0, rtol=0
)
torch.testing.assert_close(diff_out.cpu(), expected_diff, atol=0, rtol=0)
torch.testing.assert_close(diff_right_out.cpu(), expected_diff_right, atol=0, rtol=0)
torch.testing.assert_close(head_out.cpu(), expected_head, atol=0, rtol=0)
torch.testing.assert_close(tail_out.cpu(), expected_tail, atol=0, rtol=0)
torch.testing.assert_close(head_pair_out.cpu(), expected_head, atol=0, rtol=0)
torch.testing.assert_close(tail_pair_out.cpu(), expected_tail, atol=0, rtol=0)
torch.testing.assert_close(
    shuffle_down_out.cpu(),
    expected_shuffle_down,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    shuffle_prefix_out.cpu(),
    expected_shuffle_prefix,
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
{scope_metadata}
    "exclusive": [int(x) for x in exclusive_out[:8].cpu().tolist()],
    "inclusive": [int(x) for x in inclusive_out[:8].cpu().tolist()],
    "xor_prefix": [int(x) for x in xor_prefix_out[:8].cpu().tolist()],
    "sum_aggregate": int(sum_aggregate_out[0].cpu().item()),
    "xor_aggregate": int(xor_aggregate_out[0].cpu().item()),
    "sum": int(sum_out[0].cpu().item()),
    "xor": int(xor_out[0].cpu().item()),
    "partial_valid_threads": int(NUM_VALID_THREADS),
    "partial_sum": int(partial_sum_out[0].cpu().item()),
    "partial_xor": int(partial_xor_out[0].cpu().item()),
    "diff_right": [int(x) for x in diff_right_out[:8].cpu().tolist()],
    "heads": [int(x) for x in head_out[:8].cpu().tolist()],
    "tails": [int(x) for x in tail_out[:8].cpu().tolist()],
    "shuffle_down": [int(x) for x in shuffle_down_out[:8].cpu().tolist()],
    "shuffle_prefix": int(shuffle_prefix_out[0].cpu().item()),
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_prims_api_vector_block_exchange_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str = "cuda.coop.cutlass",
    call_mode: str = "factory",
) -> None:
    """Write a source-backed CuTe/Prims block-exchange smoke."""

    if call_mode not in {"factory", "direct"}:
        raise ValueError("call_mode must be 'factory' or 'direct'")

    block_scope_expr = "coop._block"

    if call_mode == "factory":
        primitive_setup = """
striped_to_blocked = block_scope.make_exchange(
    cutlass.Int32,
    block_exchange_type=block_scope.BlockExchangeType.StripedToBlocked,
    threads_per_block=BLOCK_THREADS,
    temp_storage=exchange_temp_storage,
)
blocked_to_striped = block_scope.make_exchange(
    cutlass.Int32,
    block_exchange_type=block_scope.BlockExchangeType.BlockedToStriped,
    threads_per_block=BLOCK_THREADS,
    temp_storage=exchange_temp_storage,
)
scatter_to_striped = block_scope.make_exchange(
    cutlass.Int32,
    block_exchange_type=block_scope.BlockExchangeType.ScatterToStriped,
    threads_per_block=BLOCK_THREADS,
    temp_storage=exchange_temp_storage,
)
scatter_to_striped_flagged = block_scope.make_exchange(
    cutlass.Int32,
    block_exchange_type=block_scope.BlockExchangeType.ScatterToStripedFlagged,
    threads_per_block=BLOCK_THREADS,
    temp_storage=exchange_temp_storage,
)
"""
        primitive_calls = """
    blocked = striped_to_blocked(striped_vec)
    striped = blocked_to_striped(blocked_vec)
    scatter_striped = scatter_to_striped(
        blocked_vec,
        ranks=reverse_ranks_vec,
    )
    scatter_flagged = scatter_to_striped_flagged(
        blocked_vec,
        ranks=reverse_ranks_vec,
        valid_flags=valid_flags_vec,
    )
"""
        scope_metadata = """
    "factory_scopes": [
        striped_to_blocked.scope,
        blocked_to_striped.scope,
        scatter_to_striped.scope,
        scatter_to_striped_flagged.scope,
    ],
"""
    else:
        primitive_setup = ""
        primitive_calls = """
    blocked = block_scope.exchange_striped_to_blocked(
        striped_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=exchange_temp_storage,
    )
    striped = block_scope.exchange_blocked_to_striped(
        blocked_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=exchange_temp_storage,
    )
    scatter_striped = block_scope.exchange_scatter_to_striped(
        blocked_vec,
        reverse_ranks_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=exchange_temp_storage,
    )
    scatter_flagged = block_scope.exchange(
        blocked_vec,
        ranks=reverse_ranks_vec,
        valid_flags=valid_flags_vec,
        block_exchange_type=block_scope.BlockExchangeType.ScatterToStripedFlagged,
        threads_per_block=BLOCK_THREADS,
        temp_storage=exchange_temp_storage,
    )
"""
        scope_metadata = """
    "primitive_modules": [
        block_scope.exchange_striped_to_blocked.__module__,
        block_scope.exchange_blocked_to_striped.__module__,
        block_scope.exchange_scatter_to_striped.__module__,
        block_scope.exchange.__module__,
        block_scope.store.__module__,
    ],
"""

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_block_exchange import (
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    TOTAL_ITEMS,
    _expected_blocked_to_striped,
)

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})
block_scope = {block_scope_expr}

exchange_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
{primitive_setup}


@cute.kernel
def _prims_api_vector_block_exchange_kernel(
    blocked_values_in: cute.Tensor,
    striped_values_in: cute.Tensor,
    reverse_ranks_in: cute.Tensor,
    valid_flags_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
    scatter_flagged_out: cute.Tensor,
    items_per_thread: cutlass.Constexpr,
):
    blocked_vec = block_scope.load(
        blocked_values_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )
    striped_vec = block_scope.load(
        striped_values_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
        algorithm="striped",
        threads_per_block=BLOCK_THREADS,
    )
    reverse_ranks_vec = block_scope.load(
        reverse_ranks_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )
    valid_flags_vec = block_scope.load(
        valid_flags_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )

{primitive_calls}

    block_scope.store(
        striped_to_blocked_out, blocked, payload=_root_coop.Payload.PRIMS
    )
    block_scope.store(
        blocked_to_striped_out, striped, payload=_root_coop.Payload.PRIMS
    )
    block_scope.store(
        scatter_to_striped_out,
        scatter_striped,
        payload=_root_coop.Payload.PRIMS,
        algorithm="striped",
    )
    block_scope.store(
        scatter_flagged_out,
        scatter_flagged,
        payload=_root_coop.Payload.PRIMS,
        algorithm="striped",
    )


@cute.jit
def _run_prims_api_vector_block_exchange(
    blocked_values_in: cute.Tensor,
    striped_values_in: cute.Tensor,
    reverse_ranks_in: cute.Tensor,
    valid_flags_in: cute.Tensor,
    striped_to_blocked_out: cute.Tensor,
    blocked_to_striped_out: cute.Tensor,
    scatter_to_striped_out: cute.Tensor,
    scatter_flagged_out: cute.Tensor,
):
    _prims_api_vector_block_exchange_kernel(
        blocked_values_in,
        striped_values_in,
        reverse_ranks_in,
        valid_flags_in,
        striped_to_blocked_out,
        blocked_to_striped_out,
        scatter_to_striped_out,
        scatter_flagged_out,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
blocked_values_host = (torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 3) - 17
striped_values_host = blocked_values_host.clone()
blocked_to_striped_expected = _expected_blocked_to_striped(
    blocked_values_host,
    block_threads=BLOCK_THREADS,
    items_per_thread=ITEMS_PER_THREAD,
    torch=torch,
)
reverse_ranks_host = torch.arange(TOTAL_ITEMS - 1, -1, -1, dtype=torch.int32)
valid_flags_host = torch.ones((TOTAL_ITEMS,), dtype=torch.int32)

blocked_values_in = blocked_values_host.cuda()
striped_values_in = striped_values_host.cuda()
reverse_ranks_in = reverse_ranks_host.cuda()
valid_flags_in = valid_flags_host.cuda()
striped_to_blocked_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
blocked_to_striped_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
scatter_to_striped_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
scatter_flagged_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_prims_api_vector_block_exchange(
    from_dlpack(blocked_values_in),
    from_dlpack(striped_values_in),
    from_dlpack(reverse_ranks_in),
    from_dlpack(valid_flags_in),
    from_dlpack(striped_to_blocked_out),
    from_dlpack(blocked_to_striped_out),
    from_dlpack(scatter_to_striped_out),
    from_dlpack(scatter_flagged_out),
)
torch.cuda.synchronize()

reverse_expected = torch.flip(blocked_values_host, dims=(0,))
torch.testing.assert_close(
    striped_to_blocked_out.cpu(),
    blocked_values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    blocked_to_striped_out.cpu(),
    blocked_to_striped_expected,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    scatter_to_striped_out.cpu(),
    reverse_expected,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    scatter_flagged_out.cpu(),
    reverse_expected,
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
{scope_metadata}
    "striped_to_blocked": [int(x) for x in striped_to_blocked_out[:8].cpu().tolist()],
    "blocked_to_striped": [int(x) for x in blocked_to_striped_out[:8].cpu().tolist()],
    "scatter_to_striped": [int(x) for x in scatter_to_striped_out[:8].cpu().tolist()],
    "scatter_flagged": [int(x) for x in scatter_flagged_out[:8].cpu().tolist()],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_prims_api_vector_float64_scan_reduce_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str = "cuda.coop.cutlass",
) -> None:
    """Write a source-backed Prims Float64 scan/reduce smoke for the CUTLASS API."""

    block_scope_expr = "coop._block"

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})
block_scope = {block_scope_expr}

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 3
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD

scan_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
reduce_temp_storage = block_scope.TempStorage(size_in_bytes=4096, sharing="shared")


@cute.kernel
def _prims_api_vector_float64_scan_reduce_kernel(
    values_in: cute.Tensor,
    exclusive_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    sum_out: cute.Tensor,
    reduce_out: cute.Tensor,
    items_per_thread: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    values_vec = block_scope.load(
        values_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Float64,
    )

    exclusive = block_scope.exclusive_sum(
        values_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=scan_temp_storage,
    )
    inclusive = block_scope.inclusive_sum(
        values_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=scan_temp_storage,
    )
    total = block_scope.sum(
        values_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=reduce_temp_storage,
    )
    reduced = block_scope.reduce(
        values_vec,
        threads_per_block=BLOCK_THREADS,
        temp_storage=reduce_temp_storage,
    )

    block_scope.store(exclusive_out, exclusive, payload=_root_coop.Payload.PRIMS)
    block_scope.store(inclusive_out, inclusive, payload=_root_coop.Payload.PRIMS)
    sum_out[tidx] = total
    reduce_out[tidx] = reduced


@cute.jit
def _run_prims_api_vector_float64_scan_reduce(
    values_in: cute.Tensor,
    exclusive_out: cute.Tensor,
    inclusive_out: cute.Tensor,
    sum_out: cute.Tensor,
    reduce_out: cute.Tensor,
):
    _prims_api_vector_float64_scan_reduce_kernel(
        values_in,
        exclusive_out,
        inclusive_out,
        sum_out,
        reduce_out,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
values_host = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.float64)
values_in = values_host.cuda()
exclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.float64, device="cuda")
inclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.float64, device="cuda")
sum_out = torch.zeros((BLOCK_THREADS,), dtype=torch.float64, device="cuda")
reduce_out = torch.zeros((BLOCK_THREADS,), dtype=torch.float64, device="cuda")

_run_prims_api_vector_float64_scan_reduce(
    from_dlpack(values_in),
    from_dlpack(exclusive_out),
    from_dlpack(inclusive_out),
    from_dlpack(sum_out),
    from_dlpack(reduce_out),
)
torch.cuda.synchronize()

expected_inclusive = torch.cumsum(values_host, dim=0)
expected_exclusive = expected_inclusive - values_host
expected_total = torch.full(
    (BLOCK_THREADS,),
    float(values_host.sum().item()),
    dtype=torch.float64,
)

torch.testing.assert_close(exclusive_out.cpu(), expected_exclusive, atol=0, rtol=0)
torch.testing.assert_close(inclusive_out.cpu(), expected_inclusive, atol=0, rtol=0)
torch.testing.assert_close(sum_out.cpu(), expected_total, atol=0, rtol=0)
torch.testing.assert_close(reduce_out.cpu(), expected_total, atol=0, rtol=0)

print(json.dumps({{
    "api_module": coop.__name__,
    "primitive_modules": [
        block_scope.exclusive_sum.__module__,
        block_scope.inclusive_sum.__module__,
        block_scope.sum.__module__,
        block_scope.reduce.__module__,
        block_scope.store.__module__,
    ],
    "exclusive": [float(x) for x in exclusive_out[:8].cpu().tolist()],
    "inclusive": [float(x) for x in inclusive_out[:8].cpu().tolist()],
    "sum": float(sum_out[0].cpu().item()),
    "reduce": float(reduce_out[0].cpu().item()),
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_prims_api_vector_histogram_run_length_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str = "cuda.coop.cutlass",
    call_mode: str = "factory",
) -> None:
    """Write a source-backed CuTe/Prims histogram/RLE smoke for a CUTLASS API."""

    if call_mode not in {"factory", "direct"}:
        raise ValueError("call_mode must be 'factory' or 'direct'")

    block_scope_expr = "coop._block"

    if call_mode == "factory":
        primitive_setup = """
histogram = block_scope.make_histogram(
    cutlass.Int32,
    Int32,
    threads_per_block=BLOCK_THREADS,
    bins=HISTOGRAM_BINS,
    bins_per_thread=HISTOGRAM_BINS_PER_THREAD,
    algorithm="sort",
    temp_storage=histogram_temp_storage,
)
run_length = block_scope.make_run_length(
    cutlass.Int32,
    threads_per_block=BLOCK_THREADS,
    runs_per_thread=ITEMS_PER_THREAD,
    decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
    decoded_offset_dtype=Uint32,
    temp_storage=run_length_temp_storage,
)
"""
        primitive_calls = """
    histogram_counts = histogram(samples_vec)
    run_length_parent = run_length(
        run_values_vec,
        run_lengths_vec,
        total_decoded_size=total_decoded_size,
    )
"""
        scope_metadata = """
    "factory_scopes": [
        histogram.scope,
        run_length.scope,
    ],
"""
    else:
        primitive_setup = ""
        primitive_calls = """
    histogram_counts = block_scope.histogram(
        samples_vec,
        bins=HISTOGRAM_BINS,
        bins_per_thread=HISTOGRAM_BINS_PER_THREAD,
        counter_dtype=Int32,
        algorithm="sort",
        threads_per_block=BLOCK_THREADS,
        temp_storage=histogram_temp_storage,
    )
    run_length_parent = block_scope.run_length(
        run_values_vec,
        run_lengths_vec,
        decoded_items_per_thread=decoded_items_per_thread,
        total_decoded_size=total_decoded_size,
        decoded_offset_dtype=Uint32,
        threads_per_block=BLOCK_THREADS,
        temp_storage=run_length_temp_storage,
    )
"""
        scope_metadata = """
    "primitive_modules": [
        block_scope.histogram.__module__,
        block_scope.run_length.__module__,
        block_scope.store.__module__,
    ],
"""

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from cutlass.base_dsl.typing import Uint32
from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_histogram_run_length import (
    BLOCK_THREADS,
    DECODED_ITEMS_PER_THREAD,
    DECODED_WINDOW_OFFSET,
    HISTOGRAM_BINS,
    HISTOGRAM_BINS_PER_THREAD,
    ITEMS_PER_THREAD,
    TOTAL_ITEMS,
    _expected_run_length_window,
)
cutlass, cute, torch, from_dlpack, _root_coop, Int32 = require_runtime(
    include_int32=True
)
coop = importlib.import_module({api_module!r})
block_scope = {block_scope_expr}

histogram_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
run_length_temp_storage = block_scope.TempStorage(size_in_bytes=8192, sharing="shared")
{primitive_setup}


@cute.kernel
def _prims_api_vector_histogram_run_length_kernel(
    samples_in: cute.Tensor,
    run_values_in: cute.Tensor,
    run_lengths_in: cute.Tensor,
    histogram_out: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    decoded_window_offset: Int32,
    items_per_thread: cutlass.Constexpr,
):
    samples_vec = block_scope.load(
        samples_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )
    run_values_vec = block_scope.load(
        run_values_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
    )
    run_lengths_vec = block_scope.load(
        run_lengths_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Uint32,
    )
    decoded_items_per_thread = cutlass.const_expr(DECODED_ITEMS_PER_THREAD)
    relative_offsets = coop.ThreadData(decoded_items_per_thread, dtype=Uint32)
    total_decoded_size = coop.ThreadData(1, dtype=Uint32)

{primitive_calls}
    decoded = run_length_parent.decode(
        decoded_window_offset=decoded_window_offset,
        relative_offsets=relative_offsets,
    )

    block_scope.store(
        histogram_out,
        histogram_counts,
        payload=_root_coop.Payload.PRIMS,
        algorithm="striped",
    )
    block_scope.store(decoded_out, decoded, payload=_root_coop.Payload.PRIMS)
    block_scope.store(
        offsets_out, relative_offsets, payload=_root_coop.Payload.PRIMS
    )
    block_scope.store(
        total_out, total_decoded_size, payload=_root_coop.Payload.PRIMS
    )


@cute.jit
def _run_prims_api_vector_histogram_run_length(
    samples_in: cute.Tensor,
    run_values_in: cute.Tensor,
    run_lengths_in: cute.Tensor,
    histogram_out: cute.Tensor,
    decoded_out: cute.Tensor,
    offsets_out: cute.Tensor,
    total_out: cute.Tensor,
    decoded_window_offset: Int32,
):
    _prims_api_vector_histogram_run_length_kernel(
        samples_in,
        run_values_in,
        run_lengths_in,
        histogram_out,
        decoded_out,
        offsets_out,
        total_out,
        decoded_window_offset,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
samples_host = torch.tensor(
    [((idx * 7 + idx // 3) % HISTOGRAM_BINS) for idx in range(TOTAL_ITEMS)],
    dtype=torch.int32,
)
run_values_host = (torch.arange(TOTAL_ITEMS, dtype=torch.int64) + 200).to(torch.int32)
run_lengths_host = ((torch.arange(TOTAL_ITEMS, dtype=torch.int64) % 4) + 1).to(
    torch.uint32
)
samples_in = samples_host.cuda()
run_values_in = run_values_host.cuda()
run_lengths_in = run_lengths_host.cuda()
histogram_out = torch.zeros((HISTOGRAM_BINS,), dtype=torch.int32, device="cuda")
decoded_out = torch.zeros(
    (BLOCK_THREADS * DECODED_ITEMS_PER_THREAD,),
    dtype=torch.int32,
    device="cuda",
)
offsets_out = torch.zeros(
    (BLOCK_THREADS * DECODED_ITEMS_PER_THREAD,),
    dtype=torch.uint32,
    device="cuda",
)
total_out = torch.zeros((BLOCK_THREADS,), dtype=torch.uint32, device="cuda")

_run_prims_api_vector_histogram_run_length(
    from_dlpack(samples_in),
    from_dlpack(run_values_in),
    from_dlpack(run_lengths_in),
    from_dlpack(histogram_out),
    from_dlpack(decoded_out),
    from_dlpack(offsets_out),
    from_dlpack(total_out),
    Int32(DECODED_WINDOW_OFFSET),
)
torch.cuda.synchronize()

expected_histogram = torch.bincount(samples_host, minlength=HISTOGRAM_BINS).to(
    torch.int32
)
expected_decoded, expected_offsets, decoded_total = _expected_run_length_window(
    run_values_host,
    run_lengths_host,
    torch=torch,
)
expected_total = torch.full((BLOCK_THREADS,), decoded_total, dtype=torch.uint32)
torch.testing.assert_close(histogram_out.cpu(), expected_histogram, atol=0, rtol=0)
torch.testing.assert_close(decoded_out.cpu(), expected_decoded, atol=0, rtol=0)
torch.testing.assert_close(offsets_out.cpu(), expected_offsets, atol=0, rtol=0)
torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)

print(json.dumps({{
    "api_module": coop.__name__,
{scope_metadata}
    "histogram": [int(x) for x in histogram_out[:8].cpu().tolist()],
    "decoded": [int(x) for x in decoded_out[:8].cpu().tolist()],
    "relative_offsets": [int(x) for x in offsets_out[:8].cpu().tolist()],
    "total_decoded_size": int(total_out[0].cpu().item()),
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_prims_api_vector_warp_prefix_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str = "cuda.coop.cutlass",
) -> None:
    """Write a source-backed CuTe/Prims direct warp smoke for a CUTLASS API."""

    warp_scope_expr = "coop._warp"

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_warp_prefix import (
    _expected_blocked_to_striped,
    _expected_prefix,
    _expected_valid_prefix,
    _expected_valid_warp_max,
    _expected_valid_warp_totals,
    _expected_warp_max,
    _expected_warp_min,
    _expected_warp_totals,
    _expected_warp_xor,
)

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})
warp_scope = {warp_scope_expr}

BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
WARP_THREADS = 32
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
VALID_WARP_LANES = 19


@cute.kernel
def _prims_api_vector_warp_prefix_kernel(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    valid_prefix_out: cute.Tensor,
    valid_prefix_aggregate_by_lane_out: cute.Tensor,
    warp_totals_by_lane_out: cute.Tensor,
    valid_warp_totals_by_lane_out: cute.Tensor,
    warp_min_by_lane_out: cute.Tensor,
    warp_max_by_lane_out: cute.Tensor,
    valid_warp_max_by_lane_out: cute.Tensor,
    warp_xor_by_lane_out: cute.Tensor,
    direct_copy_out: cute.Tensor,
    exchange_out: cute.Tensor,
    valid_lanes: cutlass.Int32,
    items_per_thread: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = cute.arch.lane_idx()
    values_prims = warp_scope.load(
        values_in,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
        threads_in_warp=WARP_THREADS,
        payload=_root_coop.Payload.PRIMS,
    )
    values_vec = coop.ThreadData.from_fn(
        items_per_thread,
        lambda item: values_prims[item],
        dtype=cutlass.Int32,
    )

    prefix_values = warp_scope.exclusive_sum(
        values_vec,
        threads_in_warp=WARP_THREADS,
    )
    valid_prefix_aggregate = coop.ThreadData(1)
    valid_prefix_values = warp_scope.exclusive_sum(
        values_vec,
        threads_in_warp=WARP_THREADS,
        valid_items=valid_lanes,
        warp_aggregate=valid_prefix_aggregate,
    )
    warp_scope.store(
        prefix_out,
        prefix_values,
        payload=_root_coop.Payload.PRIMS,
        algorithm="direct",
        threads_in_warp=WARP_THREADS,
    )
    warp_scope.store(
        valid_prefix_out,
        valid_prefix_values,
        payload=_root_coop.Payload.PRIMS,
        algorithm="direct",
        threads_in_warp=WARP_THREADS,
    )
    warp_scope.store(
        valid_prefix_aggregate_by_lane_out,
        valid_prefix_aggregate,
        payload=_root_coop.Payload.PRIMS,
        algorithm="direct",
        threads_in_warp=WARP_THREADS,
    )
    warp_totals = warp_scope.sum(
        values_vec,
        threads_in_warp=WARP_THREADS,
    )
    local_sum = values_vec[0] + values_vec[1]
    valid_warp_totals = warp_scope.sum(
        local_sum,
        threads_in_warp=WARP_THREADS,
        valid_items=valid_lanes,
    )
    warp_min = warp_scope.min(
        values_vec,
        threads_in_warp=WARP_THREADS,
    )
    warp_max = warp_scope.max(
        values_vec,
        threads_in_warp=WARP_THREADS,
    )
    local_max = cutlass.max(values_vec[0], values_vec[1])
    valid_warp_max = warp_scope.max(
        local_max,
        threads_in_warp=WARP_THREADS,
        valid_items=valid_lanes,
    )
    warp_xor = warp_scope.reduce(
        values_vec,
        binary_op="bit_xor",
        threads_in_warp=WARP_THREADS,
    )
    warp_totals_by_lane_out[tidx] = warp_totals
    warp_min_by_lane_out[tidx] = warp_min
    warp_max_by_lane_out[tidx] = warp_max
    warp_xor_by_lane_out[tidx] = warp_xor
    if lane_id == 0:
        valid_warp_totals_by_lane_out[tidx] = valid_warp_totals
        valid_warp_max_by_lane_out[tidx] = valid_warp_max
    coop._block.store(
        direct_copy_out, values_vec, payload=_root_coop.Payload.PRIMS
    )

    striped_values = warp_scope.exchange_blocked_to_striped(
        values_vec,
        threads_in_warp=WARP_THREADS,
    )
    warp_scope.store(
        exchange_out,
        striped_values,
        payload=_root_coop.Payload.PRIMS,
        algorithm="direct",
        threads_in_warp=WARP_THREADS,
    )


@cute.jit
def _run_prims_api_vector_warp_prefix(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    valid_prefix_out: cute.Tensor,
    valid_prefix_aggregate_by_lane_out: cute.Tensor,
    warp_totals_by_lane_out: cute.Tensor,
    valid_warp_totals_by_lane_out: cute.Tensor,
    warp_min_by_lane_out: cute.Tensor,
    warp_max_by_lane_out: cute.Tensor,
    valid_warp_max_by_lane_out: cute.Tensor,
    warp_xor_by_lane_out: cute.Tensor,
    direct_copy_out: cute.Tensor,
    exchange_out: cute.Tensor,
    valid_lanes: cutlass.Int32,
):
    _prims_api_vector_warp_prefix_kernel(
        values_in,
        prefix_out,
        valid_prefix_out,
        valid_prefix_aggregate_by_lane_out,
        warp_totals_by_lane_out,
        valid_warp_totals_by_lane_out,
        warp_min_by_lane_out,
        warp_max_by_lane_out,
        valid_warp_max_by_lane_out,
        warp_xor_by_lane_out,
        direct_copy_out,
        exchange_out,
        valid_lanes,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
values_host = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.int32)
values_in = values_host.cuda()
prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
valid_prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
valid_prefix_aggregate_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
warp_totals_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
valid_warp_totals_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
warp_min_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
warp_max_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
valid_warp_max_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
warp_xor_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
direct_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
exchange_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_prims_api_vector_warp_prefix(
    from_dlpack(values_in),
    from_dlpack(prefix_out),
    from_dlpack(valid_prefix_out),
    from_dlpack(valid_prefix_aggregate_by_lane_out),
    from_dlpack(warp_totals_by_lane_out),
    from_dlpack(valid_warp_totals_by_lane_out),
    from_dlpack(warp_min_by_lane_out),
    from_dlpack(warp_max_by_lane_out),
    from_dlpack(valid_warp_max_by_lane_out),
    from_dlpack(warp_xor_by_lane_out),
    from_dlpack(direct_copy_out),
    from_dlpack(exchange_out),
    cutlass.Int32(VALID_WARP_LANES),
)
torch.cuda.synchronize()

torch.testing.assert_close(
    prefix_out.cpu(),
    _expected_prefix(values_host, torch=torch),
    atol=0,
    rtol=0,
)
expected_valid_prefix = _expected_valid_prefix(
    values_host,
    VALID_WARP_LANES,
    torch=torch,
)
valid_prefix_cpu = valid_prefix_out.cpu()
for warp_id in range(BLOCK_THREADS // WARP_THREADS):
    tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
    valid_end = tile_base + VALID_WARP_LANES * ITEMS_PER_THREAD
    torch.testing.assert_close(
        valid_prefix_cpu[tile_base:valid_end],
        expected_valid_prefix[tile_base:valid_end],
        atol=0,
        rtol=0,
    )
expected_totals = _expected_warp_totals(values_host, torch=torch)
lane0_indices = torch.arange(
    0,
    BLOCK_THREADS,
    WARP_THREADS,
    dtype=torch.int64,
)
torch.testing.assert_close(
    warp_totals_by_lane_out.cpu(),
    expected_totals,
    atol=0,
    rtol=0,
)
expected_valid_totals = _expected_valid_warp_totals(
    values_host,
    VALID_WARP_LANES,
    torch=torch,
)
valid_prefix_aggregate_cpu = valid_prefix_aggregate_by_lane_out.cpu()
for warp_id in range(BLOCK_THREADS // WARP_THREADS):
    lane_base = warp_id * WARP_THREADS
    valid_lane_end = lane_base + VALID_WARP_LANES
    expected_aggregate = int(expected_valid_totals[lane_base].item())
    torch.testing.assert_close(
        valid_prefix_aggregate_cpu[lane_base:valid_lane_end],
        torch.full(
            (VALID_WARP_LANES,),
            expected_aggregate,
            dtype=torch.int32,
        ),
        atol=0,
        rtol=0,
    )
torch.testing.assert_close(
    valid_warp_totals_by_lane_out.cpu()[lane0_indices],
    expected_valid_totals[lane0_indices],
    atol=0,
    rtol=0,
)
expected_min = _expected_warp_min(values_host, torch=torch)
torch.testing.assert_close(
    warp_min_by_lane_out.cpu(),
    expected_min,
    atol=0,
    rtol=0,
)
expected_max = _expected_warp_max(values_host, torch=torch)
torch.testing.assert_close(
    warp_max_by_lane_out.cpu(),
    expected_max,
    atol=0,
    rtol=0,
)
expected_valid_max = _expected_valid_warp_max(
    values_host,
    VALID_WARP_LANES,
    torch=torch,
)
torch.testing.assert_close(
    valid_warp_max_by_lane_out.cpu()[lane0_indices],
    expected_valid_max[lane0_indices],
    atol=0,
    rtol=0,
)
expected_xor = _expected_warp_xor(values_host, torch=torch)
torch.testing.assert_close(
    warp_xor_by_lane_out.cpu(),
    expected_xor,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    direct_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    exchange_out.cpu(),
    _expected_blocked_to_striped(values_host, torch=torch),
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
    "primitive_modules": [
        coop._block.store.__module__,
        warp_scope.exclusive_sum.__module__,
        warp_scope.sum.__module__,
        warp_scope.min.__module__,
        warp_scope.max.__module__,
        warp_scope.reduce.__module__,
        warp_scope.exchange_blocked_to_striped.__module__,
        warp_scope.store.__module__,
    ],
    "prefix_out": [int(x) for x in prefix_out[:4].cpu().tolist()],
    "valid_prefix_first_warp": [
        int(x)
        for x in valid_prefix_out[: VALID_WARP_LANES * ITEMS_PER_THREAD]
        .cpu()
        .tolist()
    ],
    "valid_prefix_aggregate_first_warp": [
        int(x)
        for x in valid_prefix_aggregate_by_lane_out[:VALID_WARP_LANES]
        .cpu()
        .tolist()
    ],
    "warp_totals": [
        int(x) for x in warp_totals_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "valid_warp_totals": [
        int(x) for x in valid_warp_totals_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "warp_min": [
        int(x) for x in warp_min_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "warp_max": [
        int(x) for x in warp_max_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "valid_warp_max": [
        int(x) for x in valid_warp_max_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "warp_xor": [
        int(x) for x in warp_xor_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "direct_copy": [int(x) for x in direct_copy_out[:4].cpu().tolist()],
    "exchange_out": [int(x) for x in exchange_out[:4].cpu().tolist()],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_prims_api_vector_warp_merge_sort_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str,
    call_mode: str,
) -> None:
    """Write a source-backed CuTe/Prims warp merge-sort smoke for the CUTLASS API."""

    if call_mode not in {"direct", "factory"}:
        raise ValueError("call_mode must be 'direct' or 'factory'")

    warp_scope_expr = "coop._warp"

    if call_mode == "factory":
        setup = """
merge_sort_keys = warp_scope.make_merge_sort_keys(
    cutlass.Int32,
    items_per_thread=ITEMS_PER_THREAD,
    compare_op=">",
    threads_in_warp=WARP_THREADS,
)
merge_sort_pairs = warp_scope.make_merge_sort_pairs(
    cutlass.Int32,
    cutlass.Int32,
    items_per_thread=ITEMS_PER_THREAD,
    threads_in_warp=WARP_THREADS,
)
"""
        sort_calls = """
    desc_keys = merge_sort_keys(keys_vec)
    pair_keys, pair_values = merge_sort_pairs(keys_vec, values_vec)
"""
        scope_metadata = """
    "factory_scopes": [
        merge_sort_keys.scope,
        merge_sort_pairs.scope,
    ],
"""
    else:
        setup = ""
        sort_calls = """
    desc_keys = warp_scope.merge_sort_keys(
        keys_vec,
        compare_op=">",
        threads_in_warp=WARP_THREADS,
    )
    pair_keys, pair_values = warp_scope.merge_sort_pairs(
        keys_vec,
        values_vec,
        threads_in_warp=WARP_THREADS,
    )
"""
        scope_metadata = """
    "primitive_modules": [
        warp_scope.merge_sort_keys.__module__,
        warp_scope.merge_sort_pairs.__module__,
    ],
"""

    script_path.write_text(
        f"""
import importlib
import json
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_warp_merge_sort import (
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    TOTAL_ITEMS,
    WARP_THREADS,
    _make_unique_warp_keys,
    _sort_warp_pairs,
    _sort_warp_tiles,
)

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})
warp_scope = {warp_scope_expr}

{setup}

@cute.kernel
def _prims_api_vector_warp_merge_sort_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    desc_keys_out: cute.Tensor,
    pair_keys_out: cute.Tensor,
    pair_values_out: cute.Tensor,
    items_per_thread: cutlass.Constexpr,
):
    keys_vec = warp_scope.load(
        keys_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
        threads_in_warp=WARP_THREADS,
    )
    values_vec = warp_scope.load(
        values_in,
        payload=_root_coop.Payload.PRIMS,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
        threads_in_warp=WARP_THREADS,
    )
{sort_calls}
    warp_scope.store(
        desc_keys_out,
        desc_keys,
        payload=_root_coop.Payload.PRIMS,
        threads_in_warp=WARP_THREADS,
    )
    warp_scope.store(
        pair_keys_out,
        pair_keys,
        payload=_root_coop.Payload.PRIMS,
        threads_in_warp=WARP_THREADS,
    )
    warp_scope.store(
        pair_values_out,
        pair_values,
        payload=_root_coop.Payload.PRIMS,
        threads_in_warp=WARP_THREADS,
    )


@cute.jit
def _run_prims_api_vector_warp_merge_sort(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    desc_keys_out: cute.Tensor,
    pair_keys_out: cute.Tensor,
    pair_values_out: cute.Tensor,
):
    _prims_api_vector_warp_merge_sort_kernel(
        keys_in,
        values_in,
        desc_keys_out,
        pair_keys_out,
        pair_values_out,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
keys_host = _make_unique_warp_keys(torch=torch)
values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 3 + 11
keys_in = keys_host.cuda()
values_in = values_host.cuda()
desc_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
pair_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
pair_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_prims_api_vector_warp_merge_sort(
    from_dlpack(keys_in),
    from_dlpack(values_in),
    from_dlpack(desc_keys_out),
    from_dlpack(pair_keys_out),
    from_dlpack(pair_values_out),
)
torch.cuda.synchronize()

expected_desc_keys = _sort_warp_tiles(keys_host, torch=torch, descending=True)
expected_pair_keys, expected_pair_values = _sort_warp_pairs(
    keys_host,
    values_host,
    torch=torch,
)
torch.testing.assert_close(desc_keys_out.cpu(), expected_desc_keys, atol=0, rtol=0)
torch.testing.assert_close(pair_keys_out.cpu(), expected_pair_keys, atol=0, rtol=0)
torch.testing.assert_close(
    pair_values_out.cpu(),
    expected_pair_values,
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
{scope_metadata}
    "desc_keys_out": [int(x) for x in desc_keys_out[:4].cpu().tolist()],
    "pair_keys_out": [int(x) for x in pair_keys_out[:4].cpu().tolist()],
    "pair_values_out": [int(x) for x in pair_values_out[:4].cpu().tolist()],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )


def write_cutlass_api_vector_warp_factory_smoke(
    script_path: Path,
    *,
    source_root: Path,
    api_module: str,
) -> None:
    """Write a source-backed CuTe/Prims warp-factory smoke for a CUTLASS API."""

    warp_scope_expr = "coop._warp"

    script_path.write_text(
        f"""
import json
import importlib
import sys
from pathlib import Path

source_root = Path({str(source_root)!r}).resolve()
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_warp_prefix import (
    _expected_blocked_to_striped,
    _expected_prefix,
    _expected_valid_warp_max,
    _expected_valid_warp_totals,
    _expected_warp_max,
    _expected_warp_min,
    _expected_warp_totals,
    _expected_warp_xor,
)

cutlass, cute, torch, from_dlpack, _root_coop = require_runtime()
coop = importlib.import_module({api_module!r})
warp_scope = {warp_scope_expr}

BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
WARP_THREADS = 32
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
VALID_WARP_LANES = 19

exclusive_sum = warp_scope.make_exclusive_sum(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
)
warp_sum = warp_scope.make_sum(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
)
warp_min = warp_scope.make_min(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
)
warp_max = warp_scope.make_max(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
)
warp_reduce_xor = warp_scope.make_reduce(
    cutlass.Int32,
    binary_op="bit_xor",
    threads_in_warp=WARP_THREADS,
)
blocked_to_striped = warp_scope.make_exchange(
    cutlass.Int32,
    threads_in_warp=WARP_THREADS,
    mode="blocked_to_striped",
)
warp_store = warp_scope.make_store(
    cutlass.Int32,
    payload=_root_coop.Payload.PRIMS,
    items_per_thread=ITEMS_PER_THREAD,
    threads_in_warp=WARP_THREADS,
    algorithm="direct",
)
@cute.kernel
def _cutlass_api_vector_warp_factory_kernel(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    warp_totals_by_lane_out: cute.Tensor,
    valid_warp_totals_by_lane_out: cute.Tensor,
    warp_min_by_lane_out: cute.Tensor,
    warp_max_by_lane_out: cute.Tensor,
    valid_warp_max_by_lane_out: cute.Tensor,
    warp_xor_by_lane_out: cute.Tensor,
    direct_copy_out: cute.Tensor,
    exchange_out: cute.Tensor,
    valid_lanes: cutlass.Int32,
    items_per_thread: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = cute.arch.lane_idx()
    values_prims = warp_scope.load(
        values_in,
        items_per_thread=items_per_thread,
        dtype=cutlass.Int32,
        threads_in_warp=WARP_THREADS,
        payload=_root_coop.Payload.PRIMS,
    )
    values_vec = coop.ThreadData.from_fn(
        items_per_thread,
        lambda item: values_prims[item],
        dtype=cutlass.Int32,
    )

    prefix_values = exclusive_sum(values_vec)
    warp_store(prefix_out, prefix_values)
    warp_totals = warp_sum(values_vec)
    local_sum = values_vec[0] + values_vec[1]
    valid_warp_totals = warp_sum(
        local_sum,
        valid_items=valid_lanes,
    )
    warp_min_values = warp_min(values_vec)
    warp_max_values = warp_max(values_vec)
    local_max = cutlass.max(values_vec[0], values_vec[1])
    valid_warp_max = warp_max(
        local_max,
        valid_items=valid_lanes,
    )
    warp_xor = warp_reduce_xor(values_vec)
    warp_totals_by_lane_out[tidx] = warp_totals
    warp_min_by_lane_out[tidx] = warp_min_values
    warp_max_by_lane_out[tidx] = warp_max_values
    warp_xor_by_lane_out[tidx] = warp_xor
    if lane_id == 0:
        valid_warp_totals_by_lane_out[tidx] = valid_warp_totals
        valid_warp_max_by_lane_out[tidx] = valid_warp_max
    coop._block.store(
        direct_copy_out, values_vec, payload=_root_coop.Payload.PRIMS
    )

    striped_values = blocked_to_striped(values_vec)
    warp_store(exchange_out, striped_values)


@cute.jit
def _run_cutlass_api_vector_warp_factory(
    values_in: cute.Tensor,
    prefix_out: cute.Tensor,
    warp_totals_by_lane_out: cute.Tensor,
    valid_warp_totals_by_lane_out: cute.Tensor,
    warp_min_by_lane_out: cute.Tensor,
    warp_max_by_lane_out: cute.Tensor,
    valid_warp_max_by_lane_out: cute.Tensor,
    warp_xor_by_lane_out: cute.Tensor,
    direct_copy_out: cute.Tensor,
    exchange_out: cute.Tensor,
    valid_lanes: cutlass.Int32,
):
    _cutlass_api_vector_warp_factory_kernel(
        values_in,
        prefix_out,
        warp_totals_by_lane_out,
        valid_warp_totals_by_lane_out,
        warp_min_by_lane_out,
        warp_max_by_lane_out,
        valid_warp_max_by_lane_out,
        warp_xor_by_lane_out,
        direct_copy_out,
        exchange_out,
        valid_lanes,
        ITEMS_PER_THREAD,
    ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))


cutlass.cuda.initialize_cuda_context()
values_host = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.int32)
values_in = values_host.cuda()
prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
warp_totals_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
valid_warp_totals_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
warp_min_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
warp_max_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
valid_warp_max_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
warp_xor_by_lane_out = torch.zeros(
    (BLOCK_THREADS,),
    dtype=torch.int32,
    device="cuda",
)
direct_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
exchange_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

_run_cutlass_api_vector_warp_factory(
    from_dlpack(values_in),
    from_dlpack(prefix_out),
    from_dlpack(warp_totals_by_lane_out),
    from_dlpack(valid_warp_totals_by_lane_out),
    from_dlpack(warp_min_by_lane_out),
    from_dlpack(warp_max_by_lane_out),
    from_dlpack(valid_warp_max_by_lane_out),
    from_dlpack(warp_xor_by_lane_out),
    from_dlpack(direct_copy_out),
    from_dlpack(exchange_out),
    cutlass.Int32(VALID_WARP_LANES),
)
torch.cuda.synchronize()

torch.testing.assert_close(
    prefix_out.cpu(),
    _expected_prefix(values_host, torch=torch),
    atol=0,
    rtol=0,
)
expected_totals = _expected_warp_totals(values_host, torch=torch)
lane0_indices = torch.arange(
    0,
    BLOCK_THREADS,
    WARP_THREADS,
    dtype=torch.int64,
)
torch.testing.assert_close(
    warp_totals_by_lane_out.cpu(),
    expected_totals,
    atol=0,
    rtol=0,
)
expected_valid_totals = _expected_valid_warp_totals(
    values_host,
    VALID_WARP_LANES,
    torch=torch,
)
torch.testing.assert_close(
    valid_warp_totals_by_lane_out.cpu()[lane0_indices],
    expected_valid_totals[lane0_indices],
    atol=0,
    rtol=0,
)
expected_min = _expected_warp_min(values_host, torch=torch)
torch.testing.assert_close(
    warp_min_by_lane_out.cpu(),
    expected_min,
    atol=0,
    rtol=0,
)
expected_max = _expected_warp_max(values_host, torch=torch)
torch.testing.assert_close(
    warp_max_by_lane_out.cpu(),
    expected_max,
    atol=0,
    rtol=0,
)
expected_valid_max = _expected_valid_warp_max(
    values_host,
    VALID_WARP_LANES,
    torch=torch,
)
torch.testing.assert_close(
    valid_warp_max_by_lane_out.cpu()[lane0_indices],
    expected_valid_max[lane0_indices],
    atol=0,
    rtol=0,
)
expected_xor = _expected_warp_xor(values_host, torch=torch)
torch.testing.assert_close(
    warp_xor_by_lane_out.cpu(),
    expected_xor,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    direct_copy_out.cpu(),
    values_host,
    atol=0,
    rtol=0,
)
torch.testing.assert_close(
    exchange_out.cpu(),
    _expected_blocked_to_striped(values_host, torch=torch),
    atol=0,
    rtol=0,
)

print(json.dumps({{
    "api_module": coop.__name__,
    "factory_scopes": [
        exclusive_sum.scope,
        warp_sum.scope,
        warp_min.scope,
        warp_max.scope,
        warp_reduce_xor.scope,
        blocked_to_striped.scope,
        warp_store.scope,
    ],
    "prefix_out": [int(x) for x in prefix_out[:4].cpu().tolist()],
    "warp_totals": [
        int(x) for x in warp_totals_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "valid_warp_totals": [
        int(x) for x in valid_warp_totals_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "warp_min": [
        int(x) for x in warp_min_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "warp_max": [
        int(x) for x in warp_max_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "valid_warp_max": [
        int(x) for x in valid_warp_max_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "warp_xor": [
        int(x) for x in warp_xor_by_lane_out.cpu()[lane0_indices].tolist()
    ],
    "direct_copy": [int(x) for x in direct_copy_out[:4].cpu().tolist()],
    "exchange_out": [int(x) for x in exchange_out[:4].cpu().tolist()],
}}, sort_keys=True))
""",
        encoding="utf-8",
    )
