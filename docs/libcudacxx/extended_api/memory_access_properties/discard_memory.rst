.. _libcudacxx-extended-api-memory-access-properties-discard-memory:

cuda::discard_memory
====================

.. code:: cuda

   __device__ void discard_memory(void volatile* ptr, size_t nbytes);

**Preconditions**: ``ptr`` points to a valid allocation of size greater or equal to ``nbytes``.

**Effects**: equivalent to ``memset(ptr, _indeterminate_, nbytes)``.

**Hint**: to discard modified cache lines without writing back the cached data to memory. Enables using global memory
as temporary scratch space. Does **not** generate any HW store operations.

Example
-------

This kernel needs a scratch pad that does not fit in shared memory, so it uses an allocation in global memory instead:

.. code:: cuda

   #include <cuda/discard_memory>
   __device__ int compute(int* scratch, size_t N);

   __global__ void kernel(int const* in, int* out, int* scratch, size_t N) {
       // Each thread reads N elements into the scratch pad:
       for (int i = 0; i < N; ++i) {
           int idx = threadIdx.x + i * blockDim.x;
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
