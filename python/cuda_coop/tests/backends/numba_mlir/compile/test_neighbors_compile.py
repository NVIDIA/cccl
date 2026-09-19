# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Neighbor provider and production compilation without a GPU."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._lowering import _neighbors

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _fixed_device(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        cuda, "get_current_device", lambda: SimpleNamespace(compute_capability=(9, 0))
    )
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )


@pytest.mark.parametrize("operation", ["adjacent_difference", "discontinuity"])
def test_all_numeric_provider_overloads(operation):
    variants = []
    context = _nvrtc.resolve_compile_context()
    modes = (
        ("left", "right")
        if operation == "adjacent_difference"
        else ("heads", "tails", "heads_and_tails")
    )
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
        for mode in modes:
            configurations = [(False, False, False)]
            if mode in {"left", "heads", "heads_and_tails"}:
                configurations.append((False, True, False))
            if mode in {"right", "tails", "heads_and_tails"}:
                configurations.append((False, False, True))
            if mode == "heads_and_tails":
                configurations.append((False, True, True))
            if operation == "adjacent_difference":
                configurations.append((True, False, False))
                if mode == "left":
                    configurations.append((True, True, False))
            for partial, predecessor, successor in configurations:
                with _types.collect_specializations() as collected:
                    getattr(
                        _neighbors,
                        "block_"
                        + operation
                        + ("_both" if mode == "heads_and_tails" else ""),
                    )(
                        dtype=dtype,
                        threads_per_block=(5, 3, 2),
                        items_per_thread=3,
                        mode=mode,
                        partial=partial,
                        predecessor=predecessor,
                        successor=successor,
                    )
                collected[0][0]._compile_context = context
                variants.extend(collected)
    ltoir = _types.prepare_ltoir_bundle(
        [item[0] for item in variants],
        bundle_name=operation,
        allow_single=True,
        threads_by_algo={id(item[0]): item[1] for item in variants},
        block_threads_by_algo={id(item[0]): item[2] for item in variants},
    )
    assert ltoir


def _compile(kernel, signature):
    return kernel._compile_launch_config_signature(
        signature,
        (
            ("grid", (1, 1, 1)),
            ("block", (64, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


def test_production_partial_with_inferred_payload():
    @cuda.jit(chip="sm_90")
    def kernel(source, output, count):
        values = coop.ThreadData(3)
        for i in range(3):
            values[i] = source[cuda.threadIdx.x * 3 + i]
        differences = coop.adjacent_difference(
            coop.this_block(), values, valid_items=count
        )
        heads = coop.discontinuity(coop.this_block(), differences)
        for i in range(3):
            output[cuda.threadIdx.x * 3 + i] = heads[i]

    result = _compile(
        kernel, types.void(types.float64[::1], types.int32[::1], types.int64)
    )
    assert result.metadata["cubin"]
    assert "trap;" in next(iter(kernel.inspect_lto_ptx().values()))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"direction": "right", "valid_items": 1, "tile_successor_item": 1}, "partial"),
        ({"valid_items": 193}, "valid_items"),
        ({"valid_items": True}, "valid_items"),
        ({"tile_predecessor_item": 1.5}, "tile_predecessor_item"),
    ],
)
def test_invalid_adjacent_options_rejected(options, match):
    # Bind options into a concrete function: variadic keyword forwarding is
    # intentionally outside the device frontend's contract.
    if "direction" in options:

        @cuda.jit(chip="sm_90")
        def kernel(source):
            values = coop.ThreadData(3, dtype=types.int32)
            coop.load(coop.this_block(), source, values)
            return coop.adjacent_difference(
                coop.this_block(),
                values,
                direction="right",
                valid_items=1,
                tile_successor_item=1,
            )
    elif "valid_items" in options:
        count = options["valid_items"]

        @cuda.jit(chip="sm_90")
        def kernel(source):
            values = coop.ThreadData(3, dtype=types.int32)
            coop.load(coop.this_block(), source, values)
            return coop.adjacent_difference(
                coop.this_block(), values, valid_items=count
            )
    else:

        @cuda.jit(chip="sm_90")
        def kernel(source):
            values = coop.ThreadData(3, dtype=types.int32)
            coop.load(coop.this_block(), source, values)
            return coop.adjacent_difference(
                coop.this_block(), values, tile_predecessor_item=1.5
            )

    with pytest.raises(Exception, match=match):
        _compile(kernel, types.void(types.int32[::1]))
