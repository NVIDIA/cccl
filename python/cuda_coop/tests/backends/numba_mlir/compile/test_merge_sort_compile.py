# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Merge Sort providers and inferred payloads without a visible GPU.

The tests use real compiler and linker paths. Only device capability queries
are replaced, so failures still expose generated-source or binding problems.
"""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._lowering import _merge_sort

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _fixed_device(monkeypatch):
    """Select SM90 for compilation while requiring every GPU to be hidden."""

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )


@pytest.mark.parametrize("namespace", ["block", "warp"])
@pytest.mark.parametrize("partial", [False, True])
def test_numeric_keys_and_pairs_provider_bundle(namespace, partial):
    """Compile all numeric providers together to expose symbol collisions."""

    collected = []
    context = _nvrtc.resolve_compile_context()
    for dtype in (
        types.int8,
        types.uint8,
        types.int16,
        types.uint16,
        types.int32,
        types.uint32,
        types.int64,
        types.uint64,
        types.float32,
        types.float64,
    ):
        for pairs in (False, True):
            name = (
                f"{namespace}_merge_sort_"
                + ("pairs" if pairs else "keys")
                + ("_partial" if partial else "")
            )
            with _types.collect_specializations() as variants:
                getattr(_merge_sort, name)(
                    key_dtype=dtype,
                    value_dtype=dtype if pairs else None,
                    threads_per_block=(8, 4, 2),
                    threads_in_warp=8,
                    items_per_thread=3,
                    descending=True,
                )
            assert len(variants) == 1
            variants[0]._compile_context = context
            collected.extend(variants)
    ltoir = _types.prepare_ltoir_bundle(
        collected,
        allow_single=True,
    )
    assert isinstance(ltoir, bytes) and ltoir


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_partial_pairs_with_dtype_inference(items_per_thread):
    """Compile inferred pair types with count guards and warp barriers."""

    @cuda.jit(chip="sm_90")
    def kernel(source, values, output, associations, count, items_per_thread):
        keys = coop.ThreadData(items_per_thread)
        payload = coop.ThreadData(items_per_thread)
        thread = cuda.threadIdx.x
        for item in range(items_per_thread):
            keys[item] = source[thread * items_per_thread + item]
            payload[item] = values[thread * items_per_thread + item]
        result, result_values = coop.merge_sort_pairs(
            coop.this_warp().group_by(8),
            keys,
            payload,
            valid_items=count,
            oob_default=1000,
        )
        for item in range(items_per_thread):
            output[thread * items_per_thread + item] = result[item]
            associations[thread * items_per_thread + item] = result_values[item]

    signature = types.void(
        types.int32[::1],
        types.float64[::1],
        types.int32[::1],
        types.float64[::1],
        types.int64,
        types.IntegerLiteral(items_per_thread),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        (
            ("grid", (1, 1, 1)),
            ("block", (64, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )
    assert result.metadata["cubin"]
    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert "trap;" in ptx
    assert "bar.warp.sync" in ptx
