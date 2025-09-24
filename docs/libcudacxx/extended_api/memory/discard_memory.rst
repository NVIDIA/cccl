.. _libcudacxx-extended-api-memory-discard-memory:

``cuda::discard_memory``
========================

Defined in header ``<cuda/memory>``, ``<cuda/discard_memory>`` (deprecated since CCCL 3.2).

.. code:: cuda

   __host__ __device__ void discard_memory(volatile void* ptr, size_t nbytes);

Discard modified cache lines without writing back the cached data to memory. The functionality enables using global memory as temporary scratch space. Does **not** generate any HW store operations.

Equivalent to ``memset(ptr, _indeterminate_, nbytes)``.

**Preconditions**

- ``ptr`` points to a valid allocation in *global memory* of size greater or equal to ``nbytes``.

Example
-------

This kernel needs a scratch pad that does not fit in shared memory, so it uses an allocation in global memory instead:

.. code:: cuda

   #include <cuda/memory>

    __device__ int compute(int* scratch, size_t N);

    __global__ void kernel(const int* in, int* out, int* scratch, size_t N) {
        // Each thread reads N elements into the scratch pad:
        for (int i = 0; i < N; ++i) {
            int idx      = threadIdx.x + i * blockDim.x;
            scratch[idx] = in[idx];
        }
        __syncthreads();
        // All threads compute on the scratch pad:
        int result = compute(scratch, N);
        // All threads discard the scratch pad memory to _hint_ that it does not need to be flushed from the cache:
        cuda::discard_memory(scratch + threadIdx.x * N, N * sizeof(int));
        __syncthreads();
        out[threadIdx.x] = result;
    }
