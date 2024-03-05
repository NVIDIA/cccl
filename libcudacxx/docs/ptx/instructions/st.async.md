# st.async

-  PTX ISA: [`st.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async)

**NOTE.** Alignment of `addr` must be a multiple of vector size. For instance,
the `addr` supplied to the `v2.b32` variant must be aligned to `2 x 4 = 8` bytes.

**st_async**:
```cuda
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type& value,
  uint64_t* remote_bar);

// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type (&value)[2],
  uint64_t* remote_bar);

// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.  PTX ISA 81, SM_90
template <typename B32>
__device__ static inline void st_async(
  B32* addr,
  const B32 (&value)[4],
  uint64_t* remote_bar);
```

**Usage**:
```cuda
#include <cstdio>
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void __cluster_dims__(8, 1, 1) kernel()
{
  using cuda::ptx::sem_release;
  using cuda::ptx::sem_acquire;
  using cuda::ptx::space_cluster;
  using cuda::ptx::space_shared;
  using cuda::ptx::scope_cluster;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;

  __shared__ int receive_buffer[4];
  __shared__ barrier_t bar;
  init(&bar, blockDim.x);

  // Sync cluster to ensure remote barrier is initialized.
  cluster.sync();

  // Get address of remote cluster barrier:
  unsigned int other_block_rank = cluster.block_rank() ^ 1;
  uint64_t * remote_bar = cluster.map_shared_rank(cuda::device::barrier_native_handle(bar), other_block_rank);
  // int * remote_buffer = cluster.map_shared_rank(&receive_buffer, other_block_rank);
  int * remote_buffer = cluster.map_shared_rank(&receive_buffer[0], other_block_rank);

  // Arrive on local barrier:
  uint64_t arrival_token;
  if (threadIdx.x == 0) {
    // Thread 0 arrives and indicates it expects to receive a certain number of bytes as well
    arrival_token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, cuda::device::barrier_native_handle(bar), sizeof(receive_buffer));
  } else {
    arrival_token = cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_shared, cuda::device::barrier_native_handle(bar));
  }

  if (threadIdx.x == 0) {
    printf("[block %d] arrived with expected tx count = %llu\n", cluster.block_rank(), sizeof(receive_buffer));
  }

  // Send bytes to remote buffer, arriving on remote barrier
  if (threadIdx.x == 0) {
    cuda::ptx::st_async(remote_buffer, {int(cluster.block_rank()), 2, 3, 4}, remote_bar);
  }

  if (threadIdx.x == 0) {
    printf("[block %d] st_async to %p, %p\n",
           cluster.block_rank(),
           remote_buffer,
           remote_bar
    );
  }

  // Wait on local barrier:
  while(!cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cluster, cuda::device::barrier_native_handle(bar), arrival_token)) {}

  // Print received values:
  if (threadIdx.x == 0) {
    printf(
      "[block %d] receive_buffer = { %d, %d, %d, %d }\n",
      cluster.block_rank(),
      receive_buffer[0], receive_buffer[1], receive_buffer[2], receive_buffer[3]
    );
  }

}

int main() {
  kernel<<<8, 128>>>();
  cudaDeviceSynchronize();
}
```
