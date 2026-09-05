# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import shutil

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_THREADS = 64
_ITEMS_PER_THREAD = 5


@pytest.mark.evidence_for("group.exchange", backend="numba_mlir", evidence="link")
@pytest.mark.parametrize(
    ("group_kind", "expected_class"),
    [("block", "BlockExchange"), ("warp", "WarpExchange")],
)
def test_common_and_qualified_exchange_providers_are_eliminated(
    backend_prerequisite,
    group_kind,
    expected_class,
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

    common_constructor = coop.this_block if group_kind == "block" else coop.this_warp
    qualified_constructor = (
        numba_coop.this_block if group_kind == "block" else numba_coop.this_warp
    )

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        common_items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_items = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_items[index] = value
            qualified_items[index] = value

        common_group = common_constructor()
        qualified_group = qualified_constructor()
        common_blocked = coop.exchange(common_group, common_items)
        common_striped = coop.exchange(
            common_group,
            common_items,
            mode="blocked_to_striped",
        )
        qualified_blocked = numba_coop.exchange(
            qualified_group,
            qualified_items,
        )
        qualified_striped = numba_coop.exchange(
            qualified_group,
            qualified_items,
            mode="blocked_to_striped",
        )
        output[tid] = (
            common_blocked[0]
            + common_striped[0]
            + qualified_blocked[0]
            + qualified_striped[0]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    exchange_records = [
        record for record in records if record[0].split("<", 1)[0] == expected_class
    ]
    assert len(exchange_records) == 2
    assert {record[1] for record in exchange_records} == {
        "StripedToBlocked",
        "BlockedToStriped",
    }
    symbols = tuple({record[2] for record in exchange_records})
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


@pytest.mark.parametrize(
    ("group_kind", "expected_class"),
    [("block", "BlockExchange"), ("warp", "WarpExchange")],
)
def test_qualified_aggregate_exchange_provider_is_eliminated(
    backend_prerequisite,
    group_kind,
    expected_class,
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

    qualified_constructor = (
        numba_coop.this_block if group_kind == "block" else numba_coop.this_warp
    )

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        qualified_items = numba_coop.ThreadData(2, dtype=types.complex128)
        qualified_items[0] = source[tid * 2]
        qualified_items[1] = source[tid * 2 + 1]

        qualified_result = numba_coop.exchange(
            qualified_constructor(),
            qualified_items,
        )
        output[tid] = qualified_result[0]

    array_type = types.Array(types.complex128, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    exchange_records = [
        record for record in records if record[0].split("<", 1)[0] == expected_class
    ]
    assert len(exchange_records) == 1
    assert exchange_records[0][1] == "StripedToBlocked"
    symbol = exchange_records[0][2]

    sass = kernel.inspect_sass(inspect_key)
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
    assert (
        re.search(
            rf"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)"
            rf"[^\n]*{re.escape(symbol)}",
            sass,
        )
        is None
    )
