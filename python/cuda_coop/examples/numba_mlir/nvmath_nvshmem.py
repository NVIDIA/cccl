# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""FFT, normalize spectral power, and send it to the next NVSHMEM PE.

Requires nvmath-python, NVSHMEM4Py 0.4 with its Numba-CUDA-MLIR bindings,
libNVSHMEM 3.8, and CuPy matching the CUDA major version. Run directly for a
single-PE check, or with ``mpirun -np 2 python nvmath_nvshmem.py --mpi`` on
two supported GPUs. The MPI path additionally requires mpi4py.

NVSHMEM owns its reserved scratch from give_smem through release_smem.
The application owns all synchronization because these consumers share a
TempStorage descriptor with auto_sync=False.
"""

import argparse
import ctypes
import os

import cffi
import numpy as np
import nvshmem.core
import nvshmem.device.bindings.numba_cuda_mlir as nvshmem_device
from cuda.core import Device
from numba_cuda_mlir import cuda, types
from nvmath.device import FFT

from cuda import coop

_FFT_SIZE = 128
_FFTS_PER_BLOCK = 4
_ffi = cffi.FFI()


def make_kernel():
    """Specialize FFT and shared-memory requirements for the current device."""
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

    # example-begin kernel
    @cuda.jit(lto=True)
    def fft_normalize_send(data, staging, received, total_energy, peer):
        # Registered communication scratch and bin powers remain live across
        # the math and cooperative calls, so give each call its own region.
        scratch = coop.TempStorage(auto_sync=False, sharing="exclusive")
        fft_scratch = scratch.reserve(
            fft_scratch_elements, np.complex64, alignment=32
        )
        power = scratch.reserve(power_slots, np.float32)
        comms_scratch = scratch.reserve(
            nvshmem_scratch_bytes, np.uint8, alignment=16
        )

        # NVSHMEM's current MLIR interface takes a void pointer and byte count.
        # Every thread registers the same planned shared-memory array.
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

        total = coop.sum(
            coop.this_block(), np.float32(local_energy), temp_storage=scratch
        )
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

        # All threads participate in this block-scoped transfer. The source
        # is global memory; NVSHMEM can use its registered scratch for staging.
        nvshmem_device.float_put_block(
            _ffi.from_buffer(received),
            _ffi.from_buffer(staging),
            types.uint64(bins),
            types.int32(peer),
        )
        nvshmem_device.quiet()
        cuda.syncthreads()
        nvshmem_device.release_smem()

    # example-end kernel

    return fft_normalize_send, fft.block_dim


def _source(rank):
    rng = np.random.default_rng(42 + rank)
    shape = (_FFTS_PER_BLOCK, _FFT_SIZE)
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)


def main(use_mpi=False, *, kernel_factory=make_kernel):
    """Check FFT, reduction, and the payload received from the preceding PE."""
    rank, nranks, local_rank = 0, 1, 0
    if use_mpi:
        from mpi4py import MPI

        communicator = MPI.COMM_WORLD
        rank, nranks = communicator.Get_rank(), communicator.Get_size()
        local = communicator.Split_type(MPI.COMM_TYPE_SHARED)
        local_rank = local.Get_rank()
        local.Free()

    device = Device(local_rank)
    device.set_current()
    os.environ.setdefault("NVSHMEM_TMA_POLICY", "ENABLE")
    unique_id = nvshmem.core.get_unique_id(empty=rank != 0)
    if use_mpi:
        communicator.Bcast(unique_id._data.view(np.int8), root=0)
    nvshmem.core.init(
        device=device,
        uid=unique_id,
        rank=rank,
        nranks=nranks,
        initializer_method="uid",
    )

    stream = device.create_stream()
    staging = received = None
    try:
        bins = _FFT_SIZE * _FFTS_PER_BLOCK
        staging = nvshmem.core.array((bins,), dtype="float32")
        received = nvshmem.core.array((bins,), dtype="float32")
        received.fill(np.nan)
        device.sync()
        nvshmem.core.barrier_all(stream=stream)
        stream.sync()

        source = _source(rank)
        data = cuda.to_device(source)
        energy = cuda.to_device(np.zeros(1, dtype=np.float32))
        kernel, block = kernel_factory()
        kernel[1, block](data, staging, received, energy, (rank + 1) % nranks)
        device.sync()
        nvshmem.core.barrier_all(stream=stream)
        stream.sync()

        expected_fft = np.fft.fft(source, axis=-1)
        np.testing.assert_allclose(
            data.copy_to_host(), expected_fft, rtol=2e-5, atol=2e-5
        )
        expected_energy = np.square(np.abs(expected_fft)).sum()
        np.testing.assert_allclose(
            energy.copy_to_host(), [expected_energy], rtol=2e-5
        )
        sender_fft = np.fft.fft(_source((rank - 1) % nranks), axis=-1)
        expected_power = np.square(np.abs(sender_fft))
        expected_power /= expected_power.sum()
        np.testing.assert_allclose(
            received.get().reshape(expected_power.shape),
            expected_power,
            rtol=3e-5,
            atol=1e-7,
        )
        print(
            f"PE {rank}/{nranks}: FFT, cooperative reduction, "
            "and NVSHMEM put passed"
        )
    finally:
        if received is not None:
            nvshmem.core.free_array(received)
        if staging is not None:
            nvshmem.core.free_array(staging)
        stream.close()
        nvshmem.core.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mpi", action="store_true", help="Use one GPU per local MPI rank"
    )
    main(parser.parse_args().mpi)
