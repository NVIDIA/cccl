# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu


@pytest.mark.evidence_for("group.load", backend="numba_mlir", evidence="compile")
@pytest.mark.evidence_for("group.store", backend="numba_mlir", evidence="compile")
@pytest.mark.parametrize(
    ("group_kind", "threads", "expected_classes"),
    [
        ("block", 32, {"BlockLoad", "BlockStore"}),
        ("warp", 64, {"WarpLoad", "WarpStore"}),
    ],
)
def test_common_root_load_store_compiles_without_launch(
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
        def kernel(source, full, partial_a, partial_b):
            group = coop.this_block()
            storage = coop.TempStorage()
            full_output = coop.ThreadData(2, dtype=types.int32)
            full_loaded = coop.load(
                group,
                source,
                full_output,
                algorithm="transpose",
                temp_storage=storage,
            )
            partial_output = coop.ThreadData(2, dtype=types.int32)
            partial_loaded = coop.load(
                group,
                source,
                partial_output,
                algorithm="transpose",
                valid_items=47,
                oob_default=-1,
                offset=3,
                temp_storage=storage,
            )
            coop.store(
                group,
                full,
                full_loaded,
                algorithm="transpose",
                temp_storage=storage,
            )
            coop.store(
                group,
                partial_a,
                partial_loaded,
                algorithm="transpose",
                valid_items=47,
                offset=3,
                temp_storage=storage,
            )
            coop.store(
                group,
                partial_b,
                partial_loaded,
                algorithm="transpose",
                valid_items=47,
                offset=3,
                temp_storage=storage,
            )

    else:

        @cuda.jit
        def kernel(source, full, partial_a, partial_b):
            group = coop.this_warp()
            full_output = coop.ThreadData(2, dtype=types.int32)
            full_loaded = coop.load(group, source, full_output, algorithm="transpose")
            partial_output = coop.ThreadData(2, dtype=types.int32)
            partial_loaded = coop.load(
                group,
                source,
                partial_output,
                algorithm="transpose",
                valid_items=59,
                oob_default=-1,
                offset=3,
            )
            coop.store(group, full, full_loaded, algorithm="transpose")
            coop.store(
                group,
                partial_a,
                partial_loaded,
                algorithm="transpose",
                valid_items=59,
                offset=3,
            )
            coop.store(
                group,
                partial_b,
                partial_loaded,
                algorithm="transpose",
                valid_items=59,
                offset=3,
            )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, *([array_type] * 4))
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
    assert {record[1] for record in records} == {"Load", "Store"}
    load_records = [record for record in records if record[1] == "Load"]
    assert len(load_records) == 2
    assert load_records[0] != load_records[1]
    store_records = [record for record in records if record[1] == "Store"]
    assert len(store_records) == 2
    assert store_records[0] != store_records[1]
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin
