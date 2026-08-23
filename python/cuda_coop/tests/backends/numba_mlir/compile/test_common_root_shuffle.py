# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS_PER_THREAD = 3


@pytest.mark.evidence_for("group.shuffle", backend="numba_mlir", evidence="compile")
def test_common_and_qualified_shuffle_compile_to_the_same_cached_provider_plan(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

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
        common_default_down = coop.shuffle(common_group, common_items)
        common_explicit_down = coop.shuffle(
            common_group,
            common_items,
            mode="down",
            distance=1,
        )
        common_up = coop.shuffle(
            common_group,
            common_items,
            mode="up",
            distance=1,
        )

        qualified_group = numba_coop.this_block()
        qualified_default_down = numba_coop.shuffle(
            qualified_group,
            qualified_items,
        )
        qualified_explicit_down = numba_coop.shuffle(
            qualified_group,
            qualified_items,
            mode="down",
            distance=1,
        )
        qualified_up = numba_coop.shuffle(
            qualified_group,
            qualified_items,
            mode="up",
            distance=1,
        )

        output[tid] = (
            common_items[0]
            + common_default_down[0]
            + common_explicit_down[0]
            + common_up[1]
            + qualified_items[0]
            + qualified_default_down[0]
            + qualified_explicit_down[0]
            + qualified_up[1]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_BLOCK)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    shuffle_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockShuffle"
    ]
    assert len(shuffle_records) == 2
    assert {record[1] for record in shuffle_records} == {"Down", "Up"}
    assert len({record[2] for record in shuffle_records}) == 1
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin
