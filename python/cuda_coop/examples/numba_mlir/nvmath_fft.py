# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fuse cuFFTDx FFTs, a cooperative energy sum, and spectral normalization.

Requires nvmath-python with its device dependencies and Numba-CUDA-MLIR.
Run ``python nvmath_fft.py`` from this directory.
"""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import cuda
from nvmath.device import FFT

from cuda import coop

fft = FFT(
    fft_type="c2c",
    size=128,
    precision=np.float32,
    direction="forward",
    elements_per_thread=8,
    ffts_per_block=2,
    execution="Block",
)
FFT_SCRATCH_ELEMENTS = (
    fft.shared_memory_size + np.dtype(np.complex64).itemsize - 1
) // np.dtype(np.complex64).itemsize
SPECTRUM_SIZE = int(fft.ffts_per_block) * int(fft.size)
POWER_STORAGE_ELEMENTS = SPECTRUM_SIZE + 1


# docs: start nvmath-fft-kernel
@cuda.jit
def fft_energy_kernel(data, normalized_power, total_energy):
    # FFT and reduction run in separate phases and reuse this region.
    storage = coop.TempStorage()
    fft_scratch = storage.reserve(
        FFT_SCRATCH_ELEMENTS, np.complex64, alignment=32
    )
    # Bin powers stay live across coop.sum; the final slot broadcasts its sum.
    application = coop.TempStorage()
    powers = application.reserve(POWER_STORAGE_ELEMENTS, np.float32)
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
    # cuFFTDx does not supply a trailing barrier before scratch reuse.
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

    energy = coop.sum(
        coop.this_block(), np.float32(thread_energy), temp_storage=storage
    )
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


# docs: end nvmath-fft-kernel


def run_example(
    kernel=fft_energy_kernel,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the FFTs, normalized bin powers, and combined spectral energy."""

    rng = np.random.default_rng(42)
    shape = (fft.ffts_per_block, fft.size)
    source = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)
    data = cuda.to_device(source)
    normalized_power = cuda.device_array(shape, dtype=np.float32)
    total_energy = cuda.device_array(1, dtype=np.float32)

    kernel[1, fft.block_dim](data, normalized_power, total_energy)
    actual_fft = data.copy_to_host()
    actual_power = normalized_power.copy_to_host()
    actual_energy = float(total_energy.copy_to_host()[0])

    expected_fft = np.fft.fft(source, axis=-1)
    expected_power = np.abs(expected_fft) ** 2
    expected_energy = float(expected_power.sum())
    np.testing.assert_allclose(actual_fft, expected_fft, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=2e-5)
    np.testing.assert_allclose(
        actual_power,
        expected_power / expected_energy,
        rtol=2e-5,
        atol=2e-7,
    )
    return actual_fft, actual_power, actual_energy


def main() -> int:
    _, normalized_power, energy = run_example()
    print(f"Spectral energy: {energy:.6f}")
    print(f"Normalized bin powers sum to: {normalized_power.sum():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
