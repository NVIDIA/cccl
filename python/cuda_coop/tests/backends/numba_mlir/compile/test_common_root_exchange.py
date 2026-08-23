# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_THREADS = 64
_ITEMS_PER_THREAD = 5


@pytest.mark.evidence_for("group.exchange", backend="numba_mlir", evidence="compile")
def test_common_root_exchange_compiles_both_groups_and_modes_with_provider_plans(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        for index in range(_ITEMS_PER_THREAD):
            items[index] = source[tid * _ITEMS_PER_THREAD + index]

        block = coop.this_block()
        block_blocked = coop.exchange(block, items)
        block_striped = coop.exchange(block, items, mode="blocked_to_striped")
        warp = coop.this_warp()
        warp_blocked = coop.exchange(warp, items)
        warp_striped = coop.exchange(warp, items, mode="blocked_to_striped")
        output[tid] = (
            items[0]
            + block_blocked[0]
            + block_striped[0]
            + warp_blocked[0]
            + warp_striped[0]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    exchange_records = [
        record
        for record in records
        if record[0].split("<", 1)[0] in {"BlockExchange", "WarpExchange"}
    ]
    assert len(exchange_records) == 4
    assert {(record[0].split("<", 1)[0], record[1]) for record in exchange_records} == {
        (class_name, mode)
        for class_name in ("BlockExchange", "WarpExchange")
        for mode in ("StripedToBlocked", "BlockedToStriped")
    }
    assert len({record[2] for record in exchange_records}) == 2
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin
