## PTX instructions

The `cuda::ptx` namespace contains functions that map one-to-one to PTX
instructions. These can be used for maximal control of the generated code, or to
experiment with new hardware features before a high-level C++ API is available.

### Shared memory barrier (mbarrier)

| Instruction | Compute capability | CUDA Toolkit |
|----------------------------------------|--------------------|--------------|
| `cuda::ptx::mbarrier_arrive_expect_tx` | 9.0                | CTK 12.4     |


#### [`cuda::ptx::mbarrier_arrive_expect_tx`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)

```cuda
template <dot_scope _Sco>
__device__ inline
uint64_t mbarrier_arrive_expect_tx(sem_release_t sem, scope_t<_Sco> scope, space_shared_t spc, uint64_t* addr, uint32_t tx_count);

template <dot_scope _Sco>
__device__ inline
void mbarrier_arrive_expect_tx(sem_release_t sem, scope_t<_Sco> scope, space_shared_cluster_t spc, uint64_t* addr, uint32_t tx_count);
```

Usage:

```cuda
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void kernel() {
    using cuda::ptx::sem_release;
    using cuda::ptx::space_shared_cluster;
    using cuda::ptx::space_shared;
    using cuda::ptx::scope_cluster;
    using cuda::ptx::scope_cta;

    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier_t bar;
    init(&bar, blockDim.x);
    __syncthreads();

    NV_IF_TARGET(NV_PROVIDES_SM_90, (
        // Arrive on local shared memory barrier:
        uint64_t token;
        token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared, &bar, 1);
        token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, &bar, 1);

        // Get address of remote cluster barrier:
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        unsigned int other_block_rank = cluster.block_rank() ^ 1;
        uint64_t * remote_bar = cluster.map_shared_rank(&bar, other_block_rank);

        // Sync cluster to ensure remote barrier is initialized.
        cluster.sync();

        // Arrive on remote cluster barrier:
        cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared_cluster, remote_bar, 1);
        cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared_cluster, remote_bar, 1);
    )
}
```



