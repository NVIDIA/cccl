# Numba-CUDA-MLIR examples

Run these scripts from a checkout with `cuda-coop` and its Numba-CUDA-MLIR
backend installed. Each script checks its results against a host reference.

| Example | Demonstrates |
| --- | --- |
| [nvmath_fft.py](nvmath_fft.py) | cuFFTDx FFT, cooperative spectral-energy reduction, and application shared arrays in one kernel. |
| [nvmath_nvshmem.py](nvmath_nvshmem.py) | FFT and reduction with registered NVSHMEM shared scratch, a device transfer, and explicit completion and release. |

## Shared storage for device libraries

Both examples use `coop.TempStorage(auto_sync=False)`. Calls to
`scratch.reserve(count, dtype, alignment=...)` return ordinary shared arrays.
The compiler places each reservation in a separate region alongside the
scratch used by `coop.sum`. Repeating a call site reuses its reservation.

nvmath accepts its array directly. NVSHMEM's MLIR binding accepts a pointer
and byte count; the example adapts the reserved byte array to that interface.
The libraries retain their normal interfaces and completion requirements.
The kernel supplies block barriers and NVSHMEM completion/release calls.

The NVSHMEM example extracts pointers with CFFI's `from_buffer()`, which
preserves each reservation's offset. Numba-CUDA-MLIR 0.5.2 does not apply
sliced-array offsets in `.ctypes.data`; use the demonstrated CFFI conversion
for pointer consumers on that compiler.

Reservation counts, element types, and alignments are compile-time constants.
Supported elements are integers, floating-point numbers, and complex numbers.
A descriptor that has reservations cannot use `auto_sync=True`: the compiler
cannot infer when an external library has finished using those arrays.

## nvmath

For CUDA 13, install the device-library dependencies and run the FFT example:

```sh
python -m pip install -e './python/cuda_coop[numba-cuda-mlir-cu13]' 'nvmath-python[cu13-dx]'
python python/cuda_coop/examples/numba_mlir/nvmath_fft.py
```

For CUDA 12, use the corresponding `cu12` extras. The example computes
complex FFTs, sums their spectral energy with `coop.sum`, and uses an
application reservation to normalize the spectrum after the reduction.
The host checks the FFT, total energy, and normalized powers against NumPy.

## NVSHMEM

The communication example additionally needs NVSHMEM4Py 0.4 or later with its
Numba-CUDA-MLIR device bindings, NVSHMEM 3.8 or later, and CuPy. With CUDA 13:

```sh
python -m pip install 'nvshmem4py-cu13[mlir]>=0.4' 'nvidia-nvshmem-cu13>=3.8' cupy-cuda13x
python python/cuda_coop/examples/numba_mlir/nvmath_nvshmem.py
```

The default launches one processing element (PE) and sends to itself. For a
ring transfer across two GPUs, install an MPI implementation and `mpi4py`,
then launch one process per GPU:

```sh
mpirun -np 2 python python/cuda_coop/examples/numba_mlir/nvmath_nvshmem.py --mpi
```

Each local MPI rank selects its corresponding visible GPU. Use GPUs supported
by the installed CUDA, nvmath, and NVSHMEM versions. Shared-memory TMA
acceleration requires SM90 or newer and a supported peer connection; this
example checks communication results without assuming that TMA was selected.

Its communication scratch is separate from FFT scratch and application data
for the entire kernel. The symmetric communication buffers are global-memory
allocations managed by NVSHMEM; `reserve()` supplies block-local shared memory.

The example completes outgoing transfers before releasing the registered
scratch. Receiving processes still synchronize through NVSHMEM before reading
their buffers; a CUDA block barrier alone does not synchronize GPUs.

NCCL4Py currently supplies CuTe device bindings. These examples use NVSHMEM
for communication from a Numba-CUDA-MLIR kernel.
