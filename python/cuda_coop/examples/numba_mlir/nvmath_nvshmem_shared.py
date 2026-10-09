# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compose FFT, cooperative reduction, and NVSHMEM using static shared arrays.

Uses the same dependencies and host verification as ``nvmath_nvshmem.py``.
Run directly for one PE, or with ``mpirun -np 2 python
nvmath_nvshmem_shared.py --mpi`` on two supported GPUs. All user arrays and
the cooperative reduction scratch fit in static shared memory; this case
works with the released compiler and does not require TempStorage.reserve.
"""

import argparse
import ctypes

import cffi
import numpy as np
import nvshmem.core
import nvshmem.device.bindings.numba_cuda_mlir as nvshmem_device
from numba_cuda_mlir import cuda, types
from nvmath.device import FFT
from nvmath_nvshmem import _FFT_SIZE, _FFTS_PER_BLOCK
from nvmath_nvshmem import main as run_example

from cuda import coop

_ffi = cffi.FFI()


def make_kernel():
    """Use fixed shared arrays for FFT, application, and communication data."""
    fft = FFT(
        fft_type="c2c",
        size=_FFT_SIZE,
        precision=np.float32,
        direction="forward",
        elements_per_thread=8,
        ffts_per_block=_FFTS_PER_BLOCK,
        execution="Block",
    )
    fft_scratch_elements = int((fft.shared_memory_size + 7) // 8)
    nvshmem_scratch_bytes = int(
        nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_MINIMUM)
    )
    bins = _FFT_SIZE * _FFTS_PER_BLOCK
    power_slots = bins + 1

    # docs: start nvmath-nvshmem-shared-kernel
    @cuda.jit(lto=True)
    def fft_normalize_send(data, staging, received, total_energy, peer):
        fft_scratch = cuda.shared.array(
            fft_scratch_elements, np.complex64, alignment=32
        )
        power = cuda.shared.array(power_slots, np.float32)
        comms_scratch = cuda.shared.array(
            nvshmem_scratch_bytes, np.uint8, alignment=16
        )

        # All threads register the same shared array for NVSHMEM's lifetime.
        comms_pointer = ctypes.cast(
            _ffi.from_buffer(comms_scratch), ctypes.c_void_p
        )
        nvshmem_device.give_smem(
            comms_pointer, types.uint64(comms_scratch.size)
        )
        cuda.syncthreads()

        values = cuda.local.array(
            fft.storage_size, fft.value_type, alignment=32
        )
        transform = cuda.threadIdx.y
        index = cuda.threadIdx.x
        for item in range(fft.elements_per_thread):
            values[item] = data[transform, index]
            index += fft.stride

        fft.execute(values, fft_scratch)
        cuda.syncthreads()

        local_energy = np.float32(0)
        index = cuda.threadIdx.x
        for item in range(fft.elements_per_thread):
            value = values[item]
            data[transform, index] = value
            magnitude = value.real * value.real + value.imag * value.imag
            power[transform * _FFT_SIZE + index] = magnitude
            local_energy += magnitude
            index += fft.stride

        # cuda.coop allocates its own static scratch for this reduction.
        total = coop.sum(coop.this_block(), np.float32(local_energy))
        thread = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
        if thread == 0:
            power[bins] = total
            total_energy[0] = total
        cuda.syncthreads()

        index = cuda.threadIdx.x
        for item in range(fft.elements_per_thread):
            offset = transform * _FFT_SIZE + index
            staging[offset] = power[offset] / power[bins]
            index += fft.stride
        cuda.syncthreads()

        nvshmem_device.float_put_block(
            _ffi.from_buffer(received),
            _ffi.from_buffer(staging),
            types.uint64(bins),
            types.int32(peer),
        )
        nvshmem_device.quiet()
        cuda.syncthreads()
        nvshmem_device.release_smem()

    # docs: end nvmath-nvshmem-shared-kernel

    return fft_normalize_send, fft.block_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mpi", action="store_true", help="Use one GPU per local MPI rank"
    )
    run_example(parser.parse_args().mpi, kernel_factory=make_kernel)
