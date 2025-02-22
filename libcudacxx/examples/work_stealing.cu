   #include <iostream>
   #include <cuda/atomic>
   #include <cuda/work_stealing>   
   #include <cuda/math>
   #include <cooperative_groups/reduce.h>
   namespace cg = cooperative_groups
	  
   // Before: process one scalar addition per thread block:
   // __global__ void vec_add_reduce(int* s, int* a, int* b, int* c, int n) {
   //   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   //   int thread_sum = 0;
   //   if (idx < n) {
   //     int sum = a[idx] + b[idx];
   //     c[idx] += sum;
   //     thread_sum += sum;
   //   }
   //   auto block = cg::this_thread_block();
   //   auto tile = cg::tiled_partition<32>(block);  
   //   cg::reduce_update_async(
   //     cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s},
   //     thread_sum, cg::plus<int>{}
   //   );
   // }
   
   // After: process many scalar additions per thread block:
   __global__ void vec_add_reduce(int* s, int* a, int* b, int* c, int n) {
     // Extract common prologue outside the lambda, e.g.,
     // - __shared__ or global (malloc) memory allocation
     // - common initialization code
     // - etc.
     // Here we extract the sum to continue accumulating locally across block indices:
     int thread_sum = 0;

     cuda::for_each_canceled_block<1>([&](dim3 block_idx) {
       // block_idx may be different than the built-in blockIdx variable, that is:
       // assert(block_idx == blockIdx); // may fail!
       // so we need to use "block_idx" consistently inside for_each_canceled:
       int idx = threadIdx.x + block_idx.x * blockDim.x;
       if (idx < n) {
         int sum = a[idx] + b[idx];
         c[idx] += sum;
	 thread_sum += sum;
       }
     });
     // Note: Calling for_each_canceled_block again or calling for_each_canceled_cluster from this
     // thread block exhibits undefined behavior.

     // Extract common epilogue outside the lambda, e.g.,
     // - write back shared memory to global memory
     // - external synchronization
     // - global memory deallocation (free)
     // - etc.
     // Here we extract that the per thread-block tile reduction into the global memory location:
     auto block = cg::this_thread_block();
     auto tile = cg::tiled_partition<32>(block);  
     cg::reduce_update_async(
       cg::tiled_partition<32>(cg::this_thread_block()), cuda::atomic_ref{*s},
       thread_sum, cg::plus<int>{}
     );
   }

   int main() {
    int N = 10000;
    int *sum, *a, *b, *c;
    cudaMallocManaged(&sum, sizeof(int));
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));
    *sum = 0;
    for (int i = 0; i < N; ++i) {
      a[i] = i;
      b[i] = 1;
      c[i] = 0;
    }

    const int threads_per_block = 256;
    const int blocks_per_grid = cuda::ceil_div(N, threads_per_block);

    vec_add_reduce<<<blocks_per_grid, threads_per_block>>>(sum, a, b, c, N);

    bool success = true;    
    if(cudaGetLastError() != cudaSuccess) {
      std::cerr << "LAUNCH ERROR" << std::endl;
      success = false;
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
       std::cerr << "SYNC ERRROR" << std::endl;
       success = false;       
    }

    int should = 0;
    for (int i = 0; i < N; ++i) {
      should += c[i];
      if (c[i] != (1 + i)) {
	std::cerr << "ERROR " << i << ": " << c[i] << " != " << (1+i) << std::endl;
	success = false;
      }
    }

    if (*sum != should) {
      std::cerr << "SUM ERROR " << *sum << " != " << should << std::endl;
      success = false;
    }

    cudaFree(sum);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return success? 0 : 1;
   }
