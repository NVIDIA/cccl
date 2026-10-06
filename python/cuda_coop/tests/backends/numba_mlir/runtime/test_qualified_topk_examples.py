# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_qualified_topk_examples():
    # qualified-topk-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def select_extremes(
        source,
        count,
        smallest,
        largest,
        min_pairs,
        max_pairs,
        min_ids,
        max_ids,
        items_per_thread,
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        indices = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys, valid_items=count, oob_default=0)
        for item in range(items_per_thread):
            indices[item] = types.int32(block.rank() * items_per_thread + item)
        low = coop.topk_min_keys(block, keys, k=8, valid_items=count)
        high = coop.topk_max_keys(block, keys, k=8, valid_items=count)
        low_keys, low_ids = coop.topk_min_pairs(
            block, keys, indices, k=8, valid_items=count
        )
        high_keys, high_ids = coop.topk_max_pairs(
            block, keys, indices, k=8, valid_items=count
        )
        # Only the selected prefix is defined; its order is unspecified.
        coop.store(block, smallest, low, valid_items=8)
        coop.store(block, largest, high, valid_items=8)
        coop.store(block, min_pairs, low_keys, valid_items=8)
        coop.store(block, max_pairs, high_keys, valid_items=8)
        coop.store(block, min_ids, low_ids, valid_items=8)
        coop.store(block, max_ids, high_ids, valid_items=8)

    for items_per_thread in (1, 4):
        values = (
            np.random.default_rng(42)
            .permutation(64 * items_per_thread - 7)
            .astype(np.int32)
            - 256
        )
        outputs = [cuda.device_array(8, dtype=np.int32) for _ in range(6)]
        select_extremes[1, 64](
            cuda.to_device(values), values.size, *outputs, items_per_thread
        )
        low, high, low_keys, high_keys, low_ids, high_ids = [
            output.copy_to_host() for output in outputs
        ]
        ordered = np.sort(values)
        np.testing.assert_array_equal(np.sort(low), ordered[:8])
        np.testing.assert_array_equal(np.sort(high), ordered[-8:])
        np.testing.assert_array_equal(np.sort(low_keys), ordered[:8])
        np.testing.assert_array_equal(np.sort(high_keys), ordered[-8:])
        for keys, indices in ((low_keys, low_ids), (high_keys, high_ids)):
            assert np.all((indices >= 0) & (indices < values.size))
            np.testing.assert_array_equal(keys, values[indices])
    # qualified-topk-example-end
