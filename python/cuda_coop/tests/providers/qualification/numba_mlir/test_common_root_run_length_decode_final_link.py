# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common Numba-CUDA-MLIR Run Length Decode."""

import re
import shutil
from collections import Counter

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_BLOCK = (8, 4, 2)
_RUNS_PER_THREAD = 2
_DECODED_ITEMS_PER_THREAD = 3


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="numba_mlir",
    evidence="link",
)
def test_common_and_qualified_run_length_decode_providers_are_eliminated(
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
    def kernel(values, lengths, decoded, relative_offsets, window_offset):
        tid = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        common_values = coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint8,
        )
        common_lengths = coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint64,
        )
        qualified_values = numba_coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint8,
        )
        qualified_lengths = numba_coop.ThreadData(
            _RUNS_PER_THREAD,
            dtype=types.uint64,
        )
        for item in range(_RUNS_PER_THREAD):
            index = tid * _RUNS_PER_THREAD + item
            value = values[index]
            length = lengths[index]
            common_values[item] = value
            common_lengths[item] = length
            qualified_values[item] = value
            qualified_lengths[item] = length

        common = coop.run_length_decode(
            coop.this_block(),
            common_values,
            common_lengths,
            decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=window_offset,
        )
        qualified_relative = numba_coop.ThreadData(
            _DECODED_ITEMS_PER_THREAD,
            dtype=types.uint64,
        )
        qualified_total = numba_coop.ThreadData(1, dtype=types.uint64)
        qualified = numba_coop.run_length_decode(
            numba_coop.this_block(),
            qualified_values,
            qualified_lengths,
            decoded_items_per_thread=_DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=window_offset,
            relative_offsets=qualified_relative,
            total_decoded_size=qualified_total,
        )
        for item in range(_DECODED_ITEMS_PER_THREAD):
            index = tid * _DECODED_ITEMS_PER_THREAD + item
            decoded[index] = common[item] + qualified[item]
            relative_offsets[index] = qualified_relative[item] + qualified_total[0]

    uint8_array = types.Array(types.uint8, 1, "C")
    uint64_array = types.Array(types.uint64, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        uint8_array,
        uint64_array,
        uint8_array,
        uint64_array,
        types.uint64,
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_BLOCK)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    run_length_records = [
        record
        for record in records
        if record[0].split("<", 1)[0] == "BlockRunLengthDecodeDriver"
    ]
    assert Counter(record[1] for record in run_length_records) == {
        "DecodeAt": 1,
        "DecodeWithOffsetsAt": 1,
    }
    assert all(len(record[5]) == 1 for record in run_length_records)
    symbols = {record[2] for record in run_length_records}
    assert len(symbols) == 2

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
