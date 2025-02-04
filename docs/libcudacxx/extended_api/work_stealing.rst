.. _libcudacxx-extended-api-work-stealing:

Work stealing
=============

.. code:: cuda

   namespace cuda::experimental {
	  
       template <int ThreadBlockDim = 3, typename UnaryFunction = ..unspecified..>
           requires std::is_invocable_r_v<void, UnaryFunction, dim3>
       __device__ void try_cancel_blocks(bool is_leader, UnaryFunction uf);

   } // namespace cuda::experimental

**WARNING**: this is an experimental API.

This API is useful to implement work-stealing at thread-block level granularity.
On devices with compute capability 10.0 or higher, it may leverage hardware acceleration for work-stealing.
The main advantage over techniques like `grid-stride loops <>`__ is that this API may cooperate with the GPU work-scheduler to respect work priorities and enable load-balancing.

**Preconditions**:
  - All threads of current thread-block call ``try_cancel_blocks`` exactly once.
  - Exactly one thread-block thread calls ``try_cancel_blocks`` with ``is_leader`` parameter value ``true``.

**Effects**:
  - Invokes ``uf`` with ``dim3 == blockIdx``, then repetedly attempts to cancel the launch of a current grid thread block, and:
    - on success, calls ``uf`` with that thread blocks ``blockIdx``, 
    - otherwise, it returns.

Example
-------

This example shows how to perform work-stealing at thread-block granularity using this API.

.. code:: cuda

   #include <cuda/try_cancel>
   __global__ void vec_add1(int* a, int* b, int* c, int n, int tidx = 0) {
     cuda::experimental::try_cancel_blocks<1>(threadIdx.x == tidx, [=](dim3 tb) {
       int idx = threadIdx.x + tb.x * blockDim.x;
       if (idx < n) {
         c[idx] += a[idx] + b[idx];
       }
     });
   }
