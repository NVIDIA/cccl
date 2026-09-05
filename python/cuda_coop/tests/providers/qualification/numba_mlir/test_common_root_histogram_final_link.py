# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import shutil
from collections import Counter

import numpy as np
import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_BLOCK = (8, 4, 2)
_ITEMS_PER_THREAD = 3
_BINS = 97
_BINS_PER_THREAD = 2


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="link")
def test_common_and_qualified_histogram_provider_functions_are_eliminated(
    backend_prerequisite,
):
    if shutil.which("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")

    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    backend_prerequisite(
        "numba_mlir",
        cuda.is_available(),
        "CUDA GPU required for Numba-CUDA-MLIR tests",
    )

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    @cuda.jit
    def kernel(source, output_int32, output_int64):
        tid = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        common_samples = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
        qualified_samples = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_samples[index] = value
            qualified_samples[index] = value

        common_group = coop.this_block()
        common_atomic = coop.histogram(
            common_group,
            common_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=int,
        )
        common_sort = coop.histogram(
            common_group,
            common_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=np.int64,
            algorithm="sort",
        )
        qualified_group = numba_coop.this_block()
        qualified_atomic = numba_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
        )
        qualified_sort = numba_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=types.int64,
            algorithm="sort",
        )

        for index in range(_BINS_PER_THREAD):
            output_index = tid * _BINS_PER_THREAD + index
            output_int32[output_index] = common_atomic[index] + qualified_atomic[index]
            output_int64[output_index] = common_sort[index] + qualified_sort[index]

    source_type = types.Array(types.uint8, 1, "C")
    int32_array_type = types.Array(types.int32, 1, "C")
    int64_array_type = types.Array(types.int64, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        source_type,
        int32_array_type,
        int64_array_type,
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_BLOCK)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    histogram_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockHistogram"
    ]
    assert len(histogram_records) == 4
    assert Counter(record[1] for record in histogram_records) == {
        "InitHistogram": 2,
        "Composite": 2,
    }
    symbols = {record[2] for record in histogram_records}
    assert len(symbols) == 4

    sass = kernel.inspect_sass(inspect_key)
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
    for symbol in symbols:
        assert (
            re.search(
                rf"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)"
                rf"[^\n]*{re.escape(symbol)}",
                sass,
            )
            is None
        )
