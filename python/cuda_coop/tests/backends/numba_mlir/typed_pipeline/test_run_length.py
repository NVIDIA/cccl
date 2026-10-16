# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check native decode result extents and payload safety before execution."""

from types import SimpleNamespace

import numba_cuda_mlir.tools as compiler_tools
import numpy as np
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from ._support import requires_typed_pipeline

pytestmark = [
    requires_typed_pipeline,
    pytest.mark.backend_numba_mlir,
    pytest.mark.compile,
]


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch):
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel):
    return kernel._compile_launch_config_signature(
        types.void(types.int32[::1]),
        (
            ("grid", (1, 1, 1)),
            ("block", (32, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


def test_common_decode_returns_the_requested_window_without_auxiliary_outputs():
    decoded_items = np.int64(7)

    @cuda.jit(chip="sm_90")
    def kernel(output):
        values = coop.ThreadData(1, dtype=types.int32)
        lengths = coop.ThreadData(1, dtype=types.uint64)
        values[0] = 7
        lengths[0] = 1
        decoded = coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=decoded_items,
        )
        coop.store(coop.this_block(), output, decoded)

    assert _compile(kernel).metadata["cubin"]


@pytest.mark.parametrize(
    "length_dtype,length_items,message",
    [
        (types.float32, 1, "integer dtype"),
        (types.uint64, 2, "matching fixed extents"),
    ],
)
def test_decode_rejects_invalid_compressed_payloads(
    length_dtype, length_items, message
):
    @cuda.jit(chip="sm_90")
    def kernel(output):
        values = coop.ThreadData(1, dtype=types.int32)
        lengths = coop.ThreadData(length_items, dtype=length_dtype)
        decoded = coop.run_length_decode(
            coop.this_block(), values, lengths, decoded_items_per_thread=4
        )
        coop.store(coop.this_block(), output, decoded)

    with pytest.raises((TypedPipelineError, TypingError), match=message):
        _compile(kernel)
