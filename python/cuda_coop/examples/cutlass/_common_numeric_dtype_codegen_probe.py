# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime probe for the common numeric dtype profile."""

from __future__ import annotations

import functools
from typing import Any

from cuda.coop._core.dtype_policy import COMMON_V1_NUMERIC_DTYPE_NAMES
from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
VALUE_SEGMENTS = 10
PORTABLE_NUMERIC_DTYPE_NAMES = COMMON_V1_NUMERIC_DTYPE_NAMES
_COMPILER_DTYPE_NAMES = {
    "uint8": "Uint8",
    "int32": "Int32",
    "uint32": "Uint32",
    "int64": "Int64",
    "uint64": "Uint64",
    "float32": "Float32",
    "float64": "Float64",
}


def _store_items(output, segment: int, thread_rank, items) -> None:
    offset = segment * TOTAL_ITEMS + thread_rank * ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]


@functools.lru_cache(maxsize=len(PORTABLE_NUMERIC_DTYPE_NAMES))
def make_dtype_runner(dtype_name: str) -> tuple[Any, Any, Any, Any, Any]:
    """Build one common-versus-qualified numeric-profile runner."""

    try:
        compiler_dtype_name = _COMPILER_DTYPE_NAMES[dtype_name]
    except KeyError as exc:
        supported = ", ".join(PORTABLE_NUMERIC_DTYPE_NAMES)
        raise ValueError(
            f"dtype_name must be one of {{{supported}}}; got {dtype_name!r}"
        ) from exc

    import numpy as np
    from cutlass.base_dsl import typing as cutlass_typing

    cutlass, cute, torch, from_dlpack, cutlass_coop = require_runtime()
    from cuda import coop as common_coop

    ordinary_dtype = getattr(np, dtype_name)
    compiler_dtype = getattr(cutlass_typing, compiler_dtype_name)
    torch_dtype = getattr(torch, dtype_name)
    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop

    @cute.kernel
    def _kernel(
        source: cute.Tensor,
        values_out: cute.Tensor,
        flags_out: cute.Tensor,
        reduce_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_storage = common_coop.TempStorage()
        qualified_storage = cutlass_coop.TempStorage()

        common_loaded = common_coop.ThreadData(
            ITEMS_PER_THREAD,
            dtype=ordinary_dtype,
        )
        qualified_loaded = cutlass_coop.ThreadData(
            ITEMS_PER_THREAD,
            dtype=compiler_dtype,
        )
        common_coop.load(
            common_group,
            source,
            common_loaded,
            temp_storage=common_storage,
        )
        cutlass_coop.load(
            qualified_group,
            source,
            qualified_loaded,
            temp_storage=qualified_storage,
        )
        common_coop.store(
            common_group,
            values_out,
            common_loaded,
            temp_storage=common_storage,
        )
        cutlass_coop.store(
            qualified_group,
            values_out,
            qualified_loaded,
            offset=TOTAL_ITEMS,
            temp_storage=qualified_storage,
        )

        common_exchange = common_coop.exchange(
            common_group,
            common_loaded,
            mode="blocked_to_striped",
        )
        qualified_exchange = cutlass_coop.exchange(
            qualified_group,
            qualified_loaded,
            mode="blocked_to_striped",
        )
        common_shuffle = common_coop.shuffle(common_group, common_loaded)
        qualified_shuffle = cutlass_coop.shuffle(
            qualified_group,
            qualified_loaded,
        )
        boundary = source[0]
        common_adjacent = common_coop.adjacent_difference(
            common_group,
            common_loaded,
            tile_predecessor_item=boundary,
            temp_storage=common_storage,
        )
        qualified_adjacent = cutlass_coop.adjacent_difference(
            qualified_group,
            qualified_loaded,
            tile_predecessor_item=boundary,
            temp_storage=qualified_storage,
        )
        common_scan = common_coop.inclusive_scan(
            common_group,
            common_loaded,
            scan_op="max",
            temp_storage=common_storage,
        )
        qualified_scan = cutlass_coop.inclusive_scan(
            qualified_group,
            qualified_loaded,
            scan_op="max",
            temp_storage=qualified_storage,
        )
        common_flags = common_coop.discontinuity(
            common_group,
            common_loaded,
            tile_predecessor_item=boundary,
            temp_storage=common_storage,
        )
        qualified_flags = cutlass_coop.discontinuity(
            qualified_group,
            qualified_loaded,
            tile_predecessor_item=boundary,
            temp_storage=qualified_storage,
        )
        common_max = common_coop.reduce(
            common_group,
            common_loaded,
            binary_op="max",
        )
        qualified_max = cutlass_coop.reduce(
            qualified_group,
            qualified_loaded,
            binary_op="max",
        )

        _store_items(values_out, 2, tidx, common_exchange)
        _store_items(values_out, 3, tidx, qualified_exchange)
        _store_items(values_out, 4, tidx, common_shuffle)
        _store_items(values_out, 5, tidx, qualified_shuffle)
        _store_items(values_out, 6, tidx, common_adjacent)
        _store_items(values_out, 7, tidx, qualified_adjacent)
        _store_items(values_out, 8, tidx, common_scan)
        _store_items(values_out, 9, tidx, qualified_scan)
        _store_items(flags_out, 0, tidx, common_flags)
        _store_items(flags_out, 1, tidx, qualified_flags)
        reduce_out[tidx] = common_max
        reduce_out[BLOCK_THREADS + tidx] = qualified_max

    @cute.jit
    def _run(
        source: cute.Tensor,
        values_out: cute.Tensor,
        flags_out: cute.Tensor,
        reduce_out: cute.Tensor,
    ):
        _kernel(source, values_out, flags_out, reduce_out).launch(
            grid=(1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
        )

    return _run, torch, from_dlpack, cutlass, torch_dtype


def run_dtype_example(dtype_name: str) -> dict[str, Any]:
    """Run one portable numeric dtype against independent host oracles."""

    import numpy as np

    run, torch, from_dlpack, cutlass, torch_dtype = make_dtype_runner(dtype_name)
    cutlass.cuda.initialize_cuda_context()

    ordinary_dtype = getattr(np, dtype_name)
    values_numpy = np.arange(1, TOTAL_ITEMS + 1, dtype=ordinary_dtype)
    values_host = torch.tensor(values_numpy.tolist(), dtype=torch_dtype)
    values = values_host.cuda()
    values_out = torch.zeros(
        (VALUE_SEGMENTS * TOTAL_ITEMS,),
        dtype=torch_dtype,
        device="cuda",
    )
    flags_out = torch.full(
        (2 * TOTAL_ITEMS,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    reduce_out = torch.zeros(
        (2 * BLOCK_THREADS,),
        dtype=torch_dtype,
        device="cuda",
    )

    run(
        from_dlpack(values),
        from_dlpack(values_out),
        from_dlpack(flags_out),
        from_dlpack(reduce_out),
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(values.cpu().numpy(), values_numpy)
    segments = values_out.cpu().reshape(VALUE_SEGMENTS, TOTAL_ITEMS).numpy()
    np.testing.assert_array_equal(segments[0], values_numpy)
    np.testing.assert_array_equal(segments[1], values_numpy)

    expected_exchange = np.stack(
        (values_numpy[:BLOCK_THREADS], values_numpy[BLOCK_THREADS:]),
        axis=1,
    ).reshape(-1)
    np.testing.assert_array_equal(segments[2], expected_exchange)
    np.testing.assert_array_equal(segments[3], expected_exchange)

    # BlockShuffle deliberately leaves the final flattened position undefined.
    np.testing.assert_array_equal(segments[4, :-1], values_numpy[1:])
    np.testing.assert_array_equal(segments[5, :-1], values_numpy[1:])

    expected_adjacent = np.ones_like(values_numpy)
    expected_adjacent[0] = 0
    np.testing.assert_array_equal(segments[6], expected_adjacent)
    np.testing.assert_array_equal(segments[7], expected_adjacent)
    np.testing.assert_array_equal(segments[8], values_numpy)
    np.testing.assert_array_equal(segments[9], values_numpy)

    expected_flags = np.ones(TOTAL_ITEMS, dtype=np.int32)
    expected_flags[0] = 0
    flags_numpy = flags_out.cpu().numpy()
    np.testing.assert_array_equal(flags_numpy[:TOTAL_ITEMS], expected_flags)
    np.testing.assert_array_equal(flags_numpy[TOTAL_ITEMS:], expected_flags)
    expected_reduce = np.full(
        (2 * BLOCK_THREADS,),
        values_numpy[-1],
        dtype=ordinary_dtype,
    )
    np.testing.assert_array_equal(reduce_out.cpu().numpy(), expected_reduce)

    return {
        "block_threads": BLOCK_THREADS,
        "dtype": dtype_name,
        "input_preserved": True,
        "items_per_thread": ITEMS_PER_THREAD,
        "operations": (
            "load",
            "store",
            "reduce",
            "inclusive_scan",
            "exchange",
            "shuffle",
            "adjacent_difference",
            "discontinuity",
        ),
    }


def main() -> int:
    for dtype_name in PORTABLE_NUMERIC_DTYPE_NAMES:
        print(run_dtype_example(dtype_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
