# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

import numpy as np
import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_BLOCK = (8, 4, 2)
_ITEMS_PER_THREAD = 3
_BINS = 97
_BINS_PER_THREAD = 2


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="compile")
def test_common_and_qualified_histogram_compile_to_two_cached_provider_plans(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

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

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    histogram_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockHistogram"
    ]
    assert len(histogram_records) == 4
    assert Counter(record[1] for record in histogram_records) == {
        "InitHistogram": 2,
        "Composite": 2,
    }
    assert len({record[2] for record in histogram_records}) == 4
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin


@pytest.mark.parametrize(
    ("sample_dtype", "counter_dtype", "diagnostic"),
    [
        (
            "float32",
            "int32",
            r"cuda\.coop\.histogram common V1 supports sample dtypes uint8, "
            r"int32, uint32, int64, uint64; use a backend-qualified import "
            r"for backend-specific sample dtypes",
        ),
        (
            "int32",
            "uint8",
            r"cuda\.coop\.histogram common V1 supports counter dtypes int32, "
            r"uint32, int64, uint64; use a backend-qualified import for "
            r"backend-specific counter dtypes",
        ),
    ],
)
def test_common_histogram_rejects_dtypes_outside_each_operand_family(
    numba_mlir_cuda_available,
    sample_dtype,
    counter_dtype,
    diagnostic,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    resolved_sample_dtype = getattr(types, sample_dtype)
    resolved_counter_dtype = getattr(types, counter_dtype)

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        samples = coop.ThreadData(1, dtype=resolved_sample_dtype)
        samples[0] = source[tid]
        counts = coop.histogram(
            coop.this_block(),
            samples,
            bins=32,
            counter_dtype=resolved_counter_dtype,
        )
        output[tid] = counts[0]

    signature = cuda_typing.signature(
        types.none,
        types.Array(resolved_sample_dtype, 1, "C"),
        types.Array(resolved_counter_dtype, 1, "C"),
    )
    with pytest.raises(CoopSinglePhaseRewriteError, match=diagnostic):
        compile_for_launch(kernel, signature, block=32)
