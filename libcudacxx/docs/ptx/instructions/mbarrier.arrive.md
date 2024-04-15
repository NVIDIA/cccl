# mbarrier.arrive

-  PTX ISA: [`mbarrier.arrive`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## mbarrier.arrive

Some of the listed PTX instructions below are semantically equivalent. They
differ in one important way: the shorter instructions are typically supported on
older compilers.

| C++ | PTX |
| [(0)](#0-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.shared.b64` |
| [(1)](#1-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.shared::cta.b64` |
| [(2)](#2-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.release.cta.shared::cta.b64` |
| [(2)](#2-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.release.cluster.shared::cta.b64` |
| [(3)](#3-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.release.cta.shared::cta.b64` |
| [(3)](#3-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.release.cluster.shared::cta.b64` |
| [(4)](#4-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.release.cluster.shared::cluster.b64` |
| [(5)](#5-mbarrier_arrive) `cuda::ptx::mbarrier_arrive`| `mbarrier.arrive.release.cluster.shared::cluster.b64` |


### [(0)](#0-mbarrier_arrive) `mbarrier_arrive`
{: .no_toc }
```cuda
// mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr);
```

### [(1)](#1-mbarrier_arrive) `mbarrier_arrive`
{: .no_toc }
```cuda
// mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr,
  const uint32_t& count);
```

### [(2)](#2-mbarrier_arrive) `mbarrier_arrive`
{: .no_toc }
```cuda
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr];           // 3a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr);
```

### [(3)](#3-mbarrier_arrive) `mbarrier_arrive`
{: .no_toc }
```cuda
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr], count;    // 3b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& count);
```

### [(4)](#4-mbarrier_arrive) `mbarrier_arrive`
{: .no_toc }
```cuda
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr];                // 4a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr);
```

### [(5)](#5-mbarrier_arrive) `mbarrier_arrive`
{: .no_toc }
```cuda
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr], count;         // 4b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count);
```

## mbarrier.arrive.no_complete

| C++ | PTX |
| [(0)](#0-mbarrier_arrive_no_complete) `cuda::ptx::mbarrier_arrive_no_complete`| `mbarrier.arrive.noComplete.shared.b64` |


### [(0)](#0-mbarrier_arrive_no_complete) `mbarrier_arrive_no_complete`
{: .no_toc }
```cuda
// mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive_no_complete(
  uint64_t* addr,
  const uint32_t& count);
```

## mbarrier.arrive.expect_tx

| C++ | PTX |
| [(0)](#0-mbarrier_arrive_expect_tx) `cuda::ptx::mbarrier_arrive_expect_tx`| `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64` |
| [(0)](#0-mbarrier_arrive_expect_tx) `cuda::ptx::mbarrier_arrive_expect_tx`| `mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64` |
| [(1)](#1-mbarrier_arrive_expect_tx) `cuda::ptx::mbarrier_arrive_expect_tx`| `mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64` |


### [(0)](#0-mbarrier_arrive_expect_tx) `mbarrier_arrive_expect_tx`
{: .no_toc }
```cuda
// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64 state, [addr], tx_count; // 8.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& tx_count);
```

### [(1)](#1-mbarrier_arrive_expect_tx) `mbarrier_arrive_expect_tx`
{: .no_toc }
```cuda
// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64   _, [addr], tx_count; // 9.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& tx_count);
```

## Usage

```cuda
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void kernel() {
    using cuda::ptx::sem_release;
    using cuda::ptx::space_cluster;
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
        token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, &bar, 1);

        // Get address of remote cluster barrier:
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        unsigned int other_block_rank = cluster.block_rank() ^ 1;
        uint64_t * remote_bar = cluster.map_shared_rank(&bar, other_block_rank);

        // Sync cluster to ensure remote barrier is initialized.
        cluster.sync();

        // Arrive on remote cluster barrier:
        cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_cluster, remote_bar, 1);
    )
}
```
