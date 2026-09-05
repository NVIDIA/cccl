# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
import shutil

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]


@pytest.mark.evidence_for("group.load", backend="numba_mlir", evidence="link")
@pytest.mark.evidence_for("group.store", backend="numba_mlir", evidence="link")
@pytest.mark.parametrize(
    ("group_kind", "threads", "expected_classes"),
    [
        ("block", 32, {"BlockLoad", "BlockStore"}),
        ("warp", 64, {"WarpLoad", "WarpStore"}),
    ],
)
def test_root_load_store_provider_functions_are_eliminated(
    backend_prerequisite,
    group_kind,
    threads,
    expected_classes,
):
    if shutil.which("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")

    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    pytest.importorskip("numba_cuda_mlir")
    backend_prerequisite(
        "numba_mlir",
        cuda.is_available(),
        "CUDA GPU required for Numba-CUDA-MLIR tests",
    )

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
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    assert {record[0].split("<", 1)[0] for record in records} == expected_classes
    assert {record[1] for record in records} == {"Load", "Store"}
    load_records = [record for record in records if record[1] == "Load"]
    assert len(load_records) == 2
    assert load_records[0] != load_records[1]
    store_records = [record for record in records if record[1] == "Store"]
    assert len(store_records) == 2
    assert store_records[0] != store_records[1]
    symbols = tuple(record[2] for record in records)
    assert symbols

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
    ("group_kind", "threads", "expected_classes"),
    [
        ("block", 32, {"BlockLoad", "BlockStore"}),
        ("warp", 64, {"WarpLoad", "WarpStore"}),
    ],
)
def test_qualified_aggregate_load_store_provider_functions_are_eliminated(
    backend_prerequisite,
    group_kind,
    threads,
    expected_classes,
):
    if shutil.which("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")

    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    pytest.importorskip("numba_cuda_mlir")
    backend_prerequisite(
        "numba_mlir",
        cuda.is_available(),
        "CUDA GPU required for Numba-CUDA-MLIR tests",
    )

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def kernel(source, output):
        group = coop.this_block() if group_kind == "block" else coop.this_warp()
        items = coop.ThreadData(2, dtype=types.complex128)
        loaded = coop.load(group, source, items, algorithm="transpose")
        coop.store(group, output, loaded, algorithm="transpose")

    array_type = types.Array(types.complex128, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=threads)
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    assert {record[0].split("<", 1)[0] for record in records} == expected_classes
    assert {record[1] for record in records} == {"Load", "Store"}
    assert len(records) == 2
    symbols = tuple(record[2] for record in records)
    assert symbols

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
