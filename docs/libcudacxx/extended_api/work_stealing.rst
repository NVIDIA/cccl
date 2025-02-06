.. _libcudacxx-extended-api-work-stealing:

Work stealing
=============

.. code:: cuda

   namespace cuda::experimental {
	  
       template <int ThreadBlockRank = 3, typename UnaryFunction = ..unspecified..>
           requires std::is_invocable_r_v<void, UnaryFunction, dim3>
       __device__ void for_each_cancelled_block(UnaryFunction uf);

   } // namespace cuda::experimental

**WARNING**: This is an **Experimental API**.

**WARNING**: This API requires C++20 or newer.

This API is useful to implement work-stealing at thread-block level granularity.
On devices with compute capability 10.0 or higher, it may leverage hardware acceleration for work-stealing.
When compared against alternative work distribution techniques like `grid-stride loops <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>`__, which distribute load statically, or against other dynamic work distribution techniques using global memory concurrency, the main advantages of this API over these alternatives are:

   - It performs work-stealing dynamically: thread blocks that finish work sooner may do more work than thread blocks whose work takes longer.
   - It may cooperate with the GPU work-scheduler to respect work priorities and perform load-balancing.
   - It may have lower work-stealing latency than global memory atomics.

For better performance, extract the thread block prologue - i.e. code and data that is common to all thread blocks (e.g. initialization, __shared__ memory allocations, common constants loaded to shared memory, etc.) - and thread block epilogue (e.g. writing back shared memory to global memory) outside the lambda passed to this API (see example below).

**Mandates**:

   - ``ThreadBlockRank`` equals the rank of the thread block: ``1``, ``2``, or ``3`` for one-dimensional, two-dimensional, and three-dimensional thread blocks, respectively.

**Preconditions**:

   - All threads of current thread-block call ``for_each_cancelled_block`` **exactly once**.

**Effects**:

   - Invokes ``uf`` with ``dim3 == blockIdx``, then repeatedly attempts to cancel the launch of a current grid thread block, and:

      - on success, calls ``uf`` with that thread blocks ``blockIdx`` and repeats,
      - otherwise it failed to cancel the launch of a thread block and it returns.

Example
-------

This example shows how to perform work-stealing at thread-block granularity using this API.

.. code:: cuda

   // Before:

   #include <cuda/math>
   #include <cuda/try_cancel>
   __global__ void vec_add(int* a, int* b, int* c, int n) {
     // Extract common prologue outside the lambda, e.g.,
     // - __shared__ or global (malloc) memory allocation
     // - common initialization code
     // - etc.

     cuda::experimental::for_each_cancelled_block<1>([=](dim3 tb) {
       int idx = threadIdx.x + tb.x * blockDim.x;
       if (idx < n) {
         c[idx] += a[idx] + b[idx];
       }
     });
     // Note: Calling for_each_cancelled_block<1> again from this
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

    int tpb = 256;
    int bpg = cuda::ceil_div(N, tpb);

    vec_add<<<bpg, tpb>>>(a, b, c, N);
    cudaDeviceSynchronize();

    bool success = true;
    for (int i = 0; i < N; ++i) {
      if (c[i] != (1 + i)) {
	std::cerr << "ERROR " << i << ", " << c[i] << std::endl;
	success = false;
      }
    }

    return success? 0 : 1;
   }
