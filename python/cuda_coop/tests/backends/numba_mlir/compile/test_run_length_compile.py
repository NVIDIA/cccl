# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""GPU-free final-link checks for the complete-operation RLD providers."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as compiler_tools
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.mark.parametrize("bulk,relative", [(False, True), (True, False), (True, True)])
def test_rld_production_compile_and_link(monkeypatch, bulk, relative):
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

    @cuda.jit(chip="sm_90")
    def kernel(source, counts, destination, relative_output, offset):
        block = numba_coop.this_block()
        storage = numba_coop.TempStorage()
        values = numba_coop.ThreadData(2, dtype=types.float64)
        lengths = numba_coop.ThreadData(2, dtype=types.uint64)
        numba_coop.load(block, source, values)
        numba_coop.load(block, counts, lengths)
        if bulk:
            if relative:
                total = numba_coop.run_length_decode_into(
                    block,
                    values,
                    lengths,
                    destination,
                    decoded_items_per_thread=4,
                    destination_offset=offset,
                    relative_offsets=relative_output,
                    decoded_offset_dtype=types.uint64,
                    temp_storage=storage,
                )
            else:
                total = numba_coop.run_length_decode_into(
                    block,
                    values,
                    lengths,
                    destination,
                    decoded_items_per_thread=4,
                    destination_offset=offset,
                    temp_storage=storage,
                )
            counts[cuda.threadIdx.x] = total
        else:
            relative_items = numba_coop.ThreadData(4)
            total = numba_coop.ThreadData(1)
            decoded = numba_coop.run_length_decode(
                block,
                values,
                lengths,
                decoded_items_per_thread=4,
                decoded_window_offset=offset,
                relative_offsets=relative_items,
                total_decoded_size=total,
                decoded_offset_dtype=types.uint64,
                temp_storage=storage,
            )
            numba_coop.store(block, destination, decoded)
            for item in range(4):
                relative_output[cuda.threadIdx.x * 4 + item] = relative_items[item]
            counts[cuda.threadIdx.x] = total[0]

    signature = types.void(
        types.float64[::1],
        types.uint64[::1],
        types.float64[::1],
        types.uint64[::1],
        types.uint64,
    )
    launch = (
        ("grid", (1, 1, 1)),
        ("block", (64, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(signature, launch)
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]
    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert "trap;" in ptx
    assert "bar.sync" in ptx


@pytest.mark.parametrize(
    "destination_type",
    [
        types.float64[::1],
        types.int32[:, ::1],
        types.Array(types.int32, 1, "A"),
        types.Array(types.int32, 1, "C", readonly=True),
    ],
)
def test_bulk_destination_requires_matching_writable_contiguous_array(
    monkeypatch, destination_type
):
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

    @cuda.jit(chip="sm_90")
    def kernel(destination):
        values = numba_coop.ThreadData(1, dtype=types.int32)
        lengths = numba_coop.ThreadData(1, dtype=types.uint32)
        values[0] = 7
        lengths[0] = 1
        numba_coop.run_length_decode_into(
            numba_coop.this_block(),
            values,
            lengths,
            destination,
            decoded_items_per_thread=2,
        )

    launch = (
        ("grid", (1, 1, 1)),
        ("block", (32, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    with pytest.raises(TypeError, match="writable contiguous"):
        kernel._compile_launch_config_signature(types.void(destination_type), launch)
