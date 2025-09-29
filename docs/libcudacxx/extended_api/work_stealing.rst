.. _libcudacxx-extended-api-work-stealing:

Work stealing
=============

Defined in header ``<cuda/work_stealing>`` if the CUDA compiler supports at least PTX ISA 8.7:

.. code:: cuda

  namespace cuda::device {

  template <int ThreadBlockRank = 3, typename UnaryFunction = /*unspecified*/>
  __device__ void for_each_canceled_block(UnaryFunction uf);

  } // namespace cuda::device

**Note**: On devices with compute capability 10.0 or higher, this function may leverage hardware acceleration.

This API is primarily intended for implementing work-stealing at the thread-block level.


Compared to alternative work distribution techniques, such as  `grid-stride loops <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>`__, which distribute work statically, or dynamic work distribution methods relying on global memory concurrency, this API offers several advantages:

   - It enables dynamic work-stealing: thread blocks that complete their tasks sooner can take on additional work from slower thread blocks.
   - It may cooperate with the GPU work scheduler to respect work priorities and improve load balancing.
   - It may reduce work-stealing latency compared to global memory atomics.

For better performance, extract the shared thread-block prologue and epilogue outside the lambda and reuse them across thread-block iterations:

  - Prologue: Thread-block initialization code and data common to all thread blocks, such as ``__shared__`` memory allocation and initialization.
  - Epilogue: Epilogue: Thread-block finalization code common to all thread blocks, such as writing shared memory back to global memory..

**Mandates**:

   - ``ThreadBlockRank`` equals the rank of the thread block: ``1``, ``2``, or ``3`` for one-dimensional, two-dimensional, and three-dimensional thread blocks, respectively.
   - ``is_invocable_r_v<UnaryFunction, void, dim3>`` is true.

**Preconditions**:

   - All threads within a thread block shall call ``for_each_canceled_block`` **exactly once**.

**Effects**:

   - Invokes ``uf`` with ``blockIdx`` and then repeatedly attempts to cancel the launch of another thread block within the current grid:

      - If successful: invokes ``uf`` with the canceled thread block's ``blockIdx`` and repeats.
      - Otherwise, the function returns; it failed to cancel the launch of another thread block.

Example
-------

This example demonstrates work-stealing at thread-block granularity using this API.

.. code:: cuda

   // Before:

   #include <cuda/math>
   #include <cuda/functional>
   __global__ void vec_add(int* a, int* b, int* c, int n) {
     // Extract common prologue outside the lambda, e.g.,
     // - __shared__ or global (malloc) memory allocation
     // - common initialization code
     // - etc.

     cuda::device::for_each_canceled_block<1>([=](dim3 block_idx) {
       // block_idx may be different than the built-in blockIdx variable, that is:
       // assert(block_idx == blockIdx); // may fail!
       // so we need to use "block_idx" consistently inside for_each_canceled:
       int idx = threadIdx.x + block_idx.x * blockDim.x;
       if (idx < n) {
         c[idx] += a[idx] + b[idx];
       }
     });
     // Note: Calling for_each_canceled_block<1> again from this
     // thread block exhibits undefined behavior.

     // Extract common epilogue outside the lambda, e.g.,
     // - write back shared memory to global memory
     // - external synchronization
     // - global memory deallocation (free)
     // - etc.
   }

   int main() {
    int N = 10000;
    int *a, *b, *c;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));
    for (int i = 0; i < N; ++i) {
      a[i] = i;
      b[i] = 1;
      c[i] = 0;
    }

    const int threads_per_block = 256;
    const int blocks_per_grid = cuda::ceil_div(N, threads_per_block);

    vec_add<<<blocks_per_grid, threads_per_block>>>(a, b, c, N);
    cudaDeviceSynchronize();

    bool success = true;
    for (int i = 0; i < N; ++i) {
      if (c[i] != (1 + i)) {
	std::cerr << "ERROR " << i << ", " << c[i] << std::endl;
	success = false;
      }
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return success? 0 : 1;
   }
