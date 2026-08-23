# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
types = pytest.importorskip("numba_cuda_mlir.types")

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as coop  # noqa: E402

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_INDEX_ALIGNMENT = np.int64(16)


@cuda.jit
def _storage_foundation_kernel(output):
    data = coop.ThreadData(2, types.int32, alignment=16)
    scratch = coop.TempStorage(32, alignment=16)
    data[0] = 7
    data[1] = 11
    scratch[0] = 13
    output[0] = data[0] + data[1] + scratch[0]


@cuda.jit
def _storage_alignment_alias_kernel(output):
    data = coop.ThreadData(
        2,
        types.int32,
        alignas=_INDEX_ALIGNMENT,
        alignment=_INDEX_ALIGNMENT,
    )
    data[0] = 7
    data[1] = 11
    output[0] = data[0] + data[1]


@cuda.jit
def _storage_default_alignment_kernel(output):
    data = coop.ThreadData(2, types.int32, alignment=None)
    data[0] = 7
    data[1] = 11
    output[0] = data[0] + data[1]


def test_qualified_storage_constructors_compile_and_run():
    output = np.zeros(1, dtype=np.int32)

    _storage_foundation_kernel[1, 1](output)

    assert output.tolist() == [31]


def test_thread_data_alignment_aliases_compile_and_run():
    output = np.zeros(1, dtype=np.int32)

    _storage_alignment_alias_kernel[1, 1](output)

    assert output.tolist() == [18]


def test_thread_data_explicit_default_alignment_compiles_and_runs():
    output = np.zeros(1, dtype=np.int32)

    _storage_default_alignment_kernel[1, 1](output)

    assert output.tolist() == [18]
