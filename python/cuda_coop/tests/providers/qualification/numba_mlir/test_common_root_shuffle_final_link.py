# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import shutil

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS_PER_THREAD = 3


@pytest.mark.evidence_for("group.shuffle", backend="numba_mlir", evidence="link")
def test_common_and_qualified_shuffle_providers_are_eliminated(
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
        tid = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        common_items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_items = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_items[index] = value
            qualified_items[index] = value

        common_group = coop.this_block()
        common_down = coop.shuffle(common_group, common_items)
        common_up = coop.shuffle(
            common_group,
            common_items,
            mode="up",
            distance=1,
        )
        qualified_group = numba_coop.this_block()
        qualified_down = numba_coop.shuffle(qualified_group, qualified_items)
        qualified_up = numba_coop.shuffle(
            qualified_group,
            qualified_items,
            mode="up",
            distance=1,
        )
        output[tid] = (
            common_down[0] + common_up[1] + qualified_down[0] + qualified_up[1]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_BLOCK)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    shuffle_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockShuffle"
    ]
    assert len(shuffle_records) == 2
    assert {record[1] for record in shuffle_records} == {"Down", "Up"}
    symbols = {record[2] for record in shuffle_records}
    assert len(symbols) == 1

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


def test_qualified_complex128_shuffle_provider_is_eliminated(backend_prerequisite):
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

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        items = numba_coop.ThreadData(2, dtype=types.complex128)
        items[0] = source[tid * 2]
        items[1] = source[tid * 2 + 1]
        group = numba_coop.this_block()
        down = numba_coop.shuffle(group, items, mode="down")
        up = numba_coop.shuffle(group, items, mode="up")
        output[tid] = down[0] + up[1]

    array_type = types.Array(types.complex128, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    shuffle_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockShuffle"
    ]
    assert len(shuffle_records) == 2
    assert {record[1] for record in shuffle_records} == {"Down", "Up"}
    symbols = {record[2] for record in shuffle_records}
    assert len(symbols) == 1

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
