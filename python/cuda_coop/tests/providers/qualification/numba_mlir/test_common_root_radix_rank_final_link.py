# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common Numba-CUDA-MLIR Radix Rank."""

import re
import shutil

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_THREADS = 64
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for("group.radix_rank", backend="numba_mlir", evidence="link")
def test_common_and_qualified_radix_rank_providers_are_eliminated(
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
    def kernel(source, output):
        tid = cuda.threadIdx.x
        common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_keys = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_keys[index] = value
            qualified_keys[index] = value

        common_group = coop.this_block()
        qualified_group = numba_coop.this_block()
        common_ascending = coop.radix_rank(
            common_group,
            common_keys,
            begin_bit=28,
            end_bit=32,
        )
        qualified_ascending = numba_coop.radix_rank(
            qualified_group,
            qualified_keys,
            begin_bit=28,
            radix_bits=4,
        )
        common_descending = coop.radix_rank(
            common_group,
            common_keys,
            begin_bit=28,
            radix_bits=4,
            descending=True,
        )
        qualified_descending = numba_coop.radix_rank(
            qualified_group,
            qualified_keys,
            begin_bit=28,
            end_bit=32,
            descending=True,
        )
        output[tid] = (
            common_ascending[0]
            + qualified_ascending[0]
            + common_descending[0]
            + qualified_descending[0]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    radix_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockRadixRank"
    ]
    assert len(radix_records) == 2
    assert {record[1] for record in radix_records} == {"RankKeys"}
    assert all(len(record[5]) == 1 for record in radix_records)
    symbols = tuple(record[2] for record in radix_records)
    assert len(set(symbols)) == 2

    sass = kernel.inspect_sass(inspect_key)
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
    for symbol in set(symbols):
        assert (
            re.search(
                rf"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)"
                rf"[^\n]*{re.escape(symbol)}",
                sass,
            )
            is None
        )
