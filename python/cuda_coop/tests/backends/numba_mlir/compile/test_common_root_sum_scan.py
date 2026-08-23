# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu


@pytest.mark.evidence_for("group.reduce", backend="numba_mlir", evidence="compile")
@pytest.mark.evidence_for("group.sum", backend="numba_mlir", evidence="compile")
@pytest.mark.evidence_for("group.scan", backend="numba_mlir", evidence="compile")
@pytest.mark.evidence_for(
    "group.exclusive_sum", backend="numba_mlir", evidence="compile"
)
@pytest.mark.evidence_for(
    "group.inclusive_sum", backend="numba_mlir", evidence="compile"
)
@pytest.mark.evidence_for(
    "group.exclusive_scan", backend="numba_mlir", evidence="compile"
)
@pytest.mark.evidence_for(
    "group.inclusive_scan", backend="numba_mlir", evidence="compile"
)
@pytest.mark.parametrize(
    ("group_kind", "threads", "expected_classes"),
    [
        ("block", 64, {"BlockReduce", "BlockScan"}),
        ("warp", 64, {"WarpReduce", "WarpScan"}),
    ],
)
def test_common_root_reduce_sum_scan_compiles_and_records_cub_provider_plan(
    numba_mlir_cuda_available,
    group_kind,
    threads,
    expected_classes,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop

    if group_kind == "block":

        @cuda.jit
        def kernel(source, output):
            tid = cuda.threadIdx.x
            group = coop.this_block()
            storage = coop.TempStorage()
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = source[tid * 2]
            items[1] = source[tid * 2 + 1]

            output[tid] = coop.sum(group, items)
            partial_sum = coop.sum(
                group,
                source[tid],
                broadcast=False,
                valid_items=threads - 7,
                algorithm="raking",
            )
            if tid == 0:
                output[threads] = partial_sum
            output[2 * threads + tid] = coop.scan(
                group,
                items,
                temp_storage=storage,
            )[0]
            output[3 * threads + tid] = coop.exclusive_sum(
                group,
                items,
                temp_storage=storage,
            )[0]
            output[4 * threads + tid] = coop.inclusive_sum(
                group,
                items,
                temp_storage=storage,
            )[0]
            output[8 * threads + tid] = coop.exclusive_scan(
                group,
                items,
                temp_storage=storage,
            )[0]
            output[9 * threads + tid] = coop.inclusive_scan(
                group,
                items,
                temp_storage=storage,
            )[0]
            output[5 * threads + tid] = coop.exclusive_scan(
                group,
                items,
                scan_op="max",
                initial_value=0,
                temp_storage=storage,
            )[0]
            output[6 * threads + tid] = coop.inclusive_scan(
                group,
                items,
                scan_op="max",
                temp_storage=storage,
            )[0]
            maximum = coop.reduce(
                group,
                source[tid],
                binary_op="max",
                broadcast=False,
                algorithm="raking",
            )
            if tid == 0:
                output[7 * threads] = maximum

    else:

        @cuda.jit
        def kernel(source, output):
            tid = cuda.threadIdx.x
            group = coop.this_warp()
            value = source[tid]

            output[tid] = coop.sum(group, value)
            partial_sum = coop.sum(
                group,
                value,
                broadcast=False,
                valid_items=27,
            )
            if tid % 32 == 0:
                output[threads + tid] = partial_sum
            output[2 * threads + tid] = coop.scan(group, value)
            output[3 * threads + tid] = coop.exclusive_sum(group, value)
            output[4 * threads + tid] = coop.inclusive_sum(group, value)
            output[8 * threads + tid] = coop.exclusive_scan(group, value)
            output[9 * threads + tid] = coop.inclusive_scan(group, value)
            output[5 * threads + tid] = coop.exclusive_scan(
                group,
                value,
                scan_op="max",
                initial_value=0,
            )
            output[6 * threads + tid] = coop.inclusive_scan(
                group,
                value,
                scan_op="max",
            )
            maximum = coop.reduce(
                group,
                value,
                binary_op="max",
                broadcast=False,
                valid_items=27,
            )
            if tid % 32 == 0:
                output[7 * threads + tid] = maximum

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=threads)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    link_plan = result.metadata["link_plan"]
    assert link_plan.has_external_link_items
    assert link_plan.has_ltoir_link_items
    assert result.metadata["linked_external_link_items"]

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    assert {record[0].split("<", 1)[0] for record in records} == expected_classes
    assert any(record[0].split("<", 1)[0].endswith("Reduce") for record in records)
    assert sum(record[0].split("<", 1)[0].endswith("Scan") for record in records) >= 4

    assert kernel.get_metadata(inspect_key)["cubin"] == cubin
