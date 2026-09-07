# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.mark.parametrize("algorithm", ["atomic", "sort"])
def test_fused_histogram_provider_compiles_ltoir(algorithm) -> None:
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering._histogram import _group_histogram

    invocable = _group_histogram(
        types.uint8,
        types.uint64,
        (32, 2, 1),
        2,
        128,
        algorithm,
    )

    assert invocable.specialization.method_name == "Histogram"
    assert "BlockHistogram" in invocable.specialization.struct_name
    assert len(invocable.files) == 1
    artifact = Path(invocable.files[0])
    assert artifact.suffix == ".ltoir"
    assert artifact.stat().st_size > 0


@pytest.mark.parametrize("with_relative_offsets", [False, True])
def test_fused_run_length_decode_provider_compiles_ltoir(
    with_relative_offsets,
) -> None:
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering._run_length_decode import (
        _group_run_length_decode,
    )

    invocable = _group_run_length_decode(
        types.int32,
        types.uint32,
        types.uint32,
        types.uint32,
        (32, 2, 1),
        2,
        3,
        with_relative_offsets,
        types.uint32 if with_relative_offsets else None,
    )

    expected_method = "DecodeWithOffsetsAt" if with_relative_offsets else "DecodeAt"
    assert invocable.specialization.method_name == expected_method
    assert "BlockRunLengthDecodeDriver" in invocable.specialization.struct_name
    assert len(invocable.files) == 1
    artifact = Path(invocable.files[0])
    assert artifact.suffix == ".ltoir"
    assert artifact.stat().st_size > 0
