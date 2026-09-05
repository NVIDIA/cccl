# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dataclasses

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

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

_THREADS = 128


@dataclasses.dataclass
class _ScaleBiasTraits:
    scale: np.int32
    bias: np.int32


@cuda.jit(extensions=[coop.gpu_dataclass_argument_handler])
def _scale_bias_kernel(input_values, output_values, traits):
    tid = cuda.threadIdx.x
    output_values[tid] = input_values[tid] * traits.scale + traits.bias


def test_gpu_dataclass_scalar_fields_round_trip():
    input_values = np.arange(_THREADS, dtype=np.int32)
    traits_by_values = [
        coop.gpu_dataclass(
            _ScaleBiasTraits(np.int32(scale), np.int32(bias)),
            compute_temp_storage=False,
        )
        for scale, bias in ((3, 7), (5, -2))
    ]

    for traits in traits_by_values:
        output_values = np.zeros_like(input_values)
        _scale_bias_kernel[1, _THREADS](input_values, output_values, traits)

        np.testing.assert_array_equal(
            output_values,
            input_values * traits.scale + traits.bias,
        )
