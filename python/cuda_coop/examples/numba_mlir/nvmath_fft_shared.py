# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compose cuFFTDx and a cooperative sum with ordinary shared arrays.

Requires nvmath-python with its device dependencies and Numba-CUDA-MLIR.
Run ``python nvmath_fft_shared.py`` from this directory.
"""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import cuda
from nvmath_fft import (
    FFT_SCRATCH_ELEMENTS,
    POWER_STORAGE_ELEMENTS,
    SPECTRUM_SIZE,
    fft,
    run_example,
)

from cuda import coop


# docs: start nvmath-fft-shared-kernel
@cuda.jit
def fft_energy_shared_kernel(data, normalized_power, total_energy):
    fft_scratch = cuda.shared.array(
        FFT_SCRATCH_ELEMENTS, np.complex64, alignment=32
    )
    # Bin powers stay live across coop.sum; the final slot broadcasts its sum.
    powers = cuda.shared.array(POWER_STORAGE_ELEMENTS, np.float32)
    thread_data = cuda.local.array(
        fft.storage_size, fft.value_type, alignment=32
    )
    thread = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
    local_fft = cuda.threadIdx.y

    index = cuda.threadIdx.x
    for i in range(fft.elements_per_thread):
        thread_data[i] = data[local_fft, index]
        index += fft.stride

    fft.execute(thread_data, fft_scratch)
    cuda.syncthreads()

    thread_energy = np.float32(0)
    index = cuda.threadIdx.x
    for i in range(fft.elements_per_thread):
        value = thread_data[i]
        power = value.real * value.real + value.imag * value.imag
        powers[local_fft * fft.size + index] = power
        thread_energy += power
        data[local_fft, index] = value
        index += fft.stride

    energy = coop.sum(coop.this_block(), np.float32(thread_energy))
    if thread == 0:
        powers[SPECTRUM_SIZE] = energy
        total_energy[0] = energy
    cuda.syncthreads()

    index = cuda.threadIdx.x
    for i in range(fft.elements_per_thread):
        energy = powers[SPECTRUM_SIZE]
        power = powers[local_fft * fft.size + index]
        normalized_power[local_fft, index] = (
            power / energy if energy > 0 else np.float32(0)
        )
        index += fft.stride


# docs: end nvmath-fft-shared-kernel


def main() -> int:
    _, normalized_power, energy = run_example(fft_energy_shared_kernel)
    print(f"Spectral energy: {energy:.6f}")
    print(f"Normalized bin powers sum to: {normalized_power.sum():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
