# Numba-CUDA-MLIR examples

Run these scripts from a checkout with `cuda-coop` and its Numba-CUDA-MLIR
backend installed. Each script checks its results against a host reference.

| Example | Demonstrates |
| --- | --- |
| [nvmath_fft.py](nvmath_fft.py) | cuFFTDx FFT, cooperative spectral-energy reduction, and application shared arrays in one kernel. |
| [nvmath_nvshmem.py](nvmath_nvshmem.py) | FFT and reduction with registered NVSHMEM shared scratch, a device transfer, and explicit completion and release. |
| [nvmath_fft_shared.py](nvmath_fft_shared.py) | The FFT and reduction with ordinary static `cuda.shared.array` allocations. |
| [nvmath_nvshmem_shared.py](nvmath_nvshmem_shared.py) | The complete FFT, reduction, and NVSHMEM kernel with ordinary static shared arrays. |

## Shared storage for device libraries

Calls to `scratch.reserve(count, dtype, alignment=...)` return ordinary shared
arrays. In `nvmath_fft.py`, FFT and reduction reuse one descriptor's shared
region in separate phases; application data has its own descriptor because
it remains live across the reduction. `nvmath_nvshmem.py` uses
`coop.TempStorage(auto_sync=False, sharing="exclusive")` to keep every
reservation and primitive's scratch separate. Repeating a call site reuses
its reservation.

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
The default `sharing="shared"` lets
reservations alias one another and primitive scratch within the same descriptor;
separate descriptors always have separate storage. Use separate descriptors or
`sharing="exclusive"` for buffers that remain live across other calls.
A descriptor that has reservations cannot use `auto_sync=True`: the compiler
cannot infer when an external library has finished using those arrays.

## Ordinary shared arrays

The `_shared.py` examples allocate FFT scratch, application data, and NVSHMEM
scratch with `cuda.shared.array`. They call `coop.sum` with its default
compiler-owned scratch and use no `TempStorage` descriptor or reservation.
All their arrays have fixed sizes, and the complete static allocation fits
within the device limit. This composition works with released compiler 0.5.2;
it does not depend on the pending compiler shared-memory fix.

`reserve()` is optional for these kernels. It is useful when the application
wants those buffers included in cooperative layout and launch accounting,
including the planner's choice of static or dynamic backing. Ordinary static
arrays are separate allocations accounted for by the compiler.

Current compatibility boundaries:

| User arrays | Cooperative scratch | Status |
| --- | --- | --- |
| Static | Static | Supported; demonstrated by both `_shared.py` examples. |
| Static | Dynamic | Requires the compiler fix and a corresponding relaxation of the cooperative compatibility guard. |
| Dynamic | Static | Requires the compiler fix and a corresponding guard relaxation; the caller supplies dynamic launch bytes. |
| Dynamic | Dynamic | Requires coordinated partitioning and launch accounting, in addition to the compiler fix. |

[Numba-CUDA-MLIR #312](https://github.com/NVIDIA/numba-cuda-mlir/pull/312)
corrects static/dynamic separation, alignment, and dynamic-array extents.
It does not by itself assign disjoint regions to independently requested
cooperative and user dynamic storage. In particular, separate zero-length
`cuda.shared.array(0, ...)` declarations can view the same dynamic memory.
The examples preserve the current compatibility guard.

## nvmath

For CUDA 13, install the device-library dependencies and run the FFT example:

```sh
python -m pip install -e './python/cuda_coop[numba-cuda-mlir-cu13]' 'nvmath-python[cu13-dx]'
python python/cuda_coop/examples/numba_mlir/nvmath_fft.py
python python/cuda_coop/examples/numba_mlir/nvmath_fft_shared.py
```

For CUDA 12, use the corresponding `cu12` extras. Both versions compute
complex FFTs, sum their spectral energy with `coop.sum`, and use an
application shared array to normalize the spectrum after the reduction.
The host checks the FFT, total energy, and normalized powers against NumPy.

## NVSHMEM

The communication example additionally needs NVSHMEM4Py 0.4 or later with its
Numba-CUDA-MLIR device bindings, NVSHMEM 3.8 or later, and CuPy. With CUDA 13:

```sh
python -m pip install 'nvshmem4py-cu13[mlir]>=0.4' 'nvidia-nvshmem-cu13>=3.8' cupy-cuda13x
python python/cuda_coop/examples/numba_mlir/nvmath_nvshmem.py
python python/cuda_coop/examples/numba_mlir/nvmath_nvshmem_shared.py
```

The default launches one processing element (PE) and sends to itself. For a
ring transfer across two GPUs, install an MPI implementation and `mpi4py`,
then launch one process per GPU:

```sh
mpirun -np 2 python python/cuda_coop/examples/numba_mlir/nvmath_nvshmem.py --mpi
mpirun -np 2 python python/cuda_coop/examples/numba_mlir/nvmath_nvshmem_shared.py --mpi
```

Each local MPI rank selects its corresponding visible GPU. Use GPUs supported
by the installed CUDA, nvmath, and NVSHMEM versions. Shared-memory TMA
acceleration requires SM90 or newer and a supported peer connection; this
example checks communication results without assuming that TMA was selected.

Its communication scratch is separate from FFT scratch and application data
for the entire kernel. The symmetric communication buffers are global-memory
allocations managed by NVSHMEM; `reserve()` or `cuda.shared.array` supplies
block-local shared memory.

The example completes outgoing transfers before releasing the registered
scratch. Receiving processes still synchronize through NVSHMEM before reading
their buffers; a CUDA block barrier alone does not synchronize GPUs.

NCCL4Py currently supplies CuTe device bindings. These examples use NVSHMEM
for communication from a Numba-CUDA-MLIR kernel.
