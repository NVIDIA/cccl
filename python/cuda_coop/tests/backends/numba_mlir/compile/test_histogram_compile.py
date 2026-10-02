# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as compiler_tools
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.mark.parametrize("algorithm", ["atomic", "sort"])
@pytest.mark.parametrize("counter_name", ["int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_histogram_links_with_independent_counter_type(
    monkeypatch, algorithm, counter_name, items_per_thread
):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    counter = getattr(types, counter_name)

    @cuda.jit(chip="sm_90")
    def kernel(source, destination, items_per_thread):
        block = coop.this_block()
        samples = coop.ThreadData(items_per_thread)
        coop.load(block, source, samples)
        counts = coop.histogram(
            block,
            samples,
            bins=65,
            bins_per_thread=2,
            counter_dtype=counter,
            algorithm=algorithm,
            temp_storage=coop.TempStorage(),
        )
        coop.store(block, destination, counts, algorithm="striped")

    launch = (
        ("grid", (1, 1, 1)),
        ("block", (64, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        types.void(
            types.uint8[::1],
            counter[::1],
            types.IntegerLiteral(items_per_thread),
        ),
        launch,
    )
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]
    assert "bar.sync" in next(iter(kernel.inspect_lto_ptx().values()))
