# mbarrier.test_wait

- PTX ISA: [`mbarrier.test_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## mbarrier.test_wait

| C++ | PTX |
| [(0)](#0-mbarrier_test_wait) `cuda::ptx::mbarrier_test_wait`| `mbarrier.test_wait.shared.b64` |
| [(1)](#1-mbarrier_test_wait) `cuda::ptx::mbarrier_test_wait`| `mbarrier.test_wait.acquire.cta.shared::cta.b64` |
| [(1)](#1-mbarrier_test_wait) `cuda::ptx::mbarrier_test_wait`| `mbarrier.test_wait.acquire.cluster.shared::cta.b64` |


### [(0)](#0-mbarrier_test_wait) `mbarrier_test_wait`
{: .no_toc }
```cuda
// mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline bool mbarrier_test_wait(
  uint64_t* addr,
  const uint64_t& state);
```

### [(1)](#1-mbarrier_test_wait) `mbarrier_test_wait`
{: .no_toc }
```cuda
// mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.   PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
```

## mbarrier.test_wait.parity

| C++ | PTX |
| [(0)](#0-mbarrier_test_wait_parity) `cuda::ptx::mbarrier_test_wait_parity`| `mbarrier.test_wait.parity.shared.b64` |
| [(1)](#1-mbarrier_test_wait_parity) `cuda::ptx::mbarrier_test_wait_parity`| `mbarrier.test_wait.parity.acquire.cta.shared::cta.b64` |
| [(1)](#1-mbarrier_test_wait_parity) `cuda::ptx::mbarrier_test_wait_parity`| `mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64` |


### [(0)](#0-mbarrier_test_wait_parity) `mbarrier_test_wait_parity`
{: .no_toc }
```cuda
// mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3.  PTX ISA 71, SM_80
template <typename=void>
__device__ static inline bool mbarrier_test_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);
```

### [(1)](#1-mbarrier_test_wait_parity) `mbarrier_test_wait_parity`
{: .no_toc }
```cuda
// mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
```
