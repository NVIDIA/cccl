# mbarrier.try_wait

- PTX ISA: [`mbarrier.try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## mbarrier.try_wait

| C++ | PTX |
| [(0)](#0-mbarrier_try_wait) `cuda::ptx::mbarrier_try_wait`| `mbarrier.try_wait.shared::cta.b64` |
| [(1)](#1-mbarrier_try_wait) `cuda::ptx::mbarrier_try_wait`| `mbarrier.try_wait.shared::cta.b64` |
| [(2)](#2-mbarrier_try_wait) `cuda::ptx::mbarrier_try_wait`| `mbarrier.try_wait.acquire.cta.shared::cta.b64` |
| [(2)](#2-mbarrier_try_wait) `cuda::ptx::mbarrier_try_wait`| `mbarrier.try_wait.acquire.cluster.shared::cta.b64` |
| [(3)](#3-mbarrier_try_wait) `cuda::ptx::mbarrier_try_wait`| `mbarrier.try_wait.acquire.cta.shared::cta.b64` |
| [(3)](#3-mbarrier_try_wait) `cuda::ptx::mbarrier_try_wait`| `mbarrier.try_wait.acquire.cluster.shared::cta.b64` |


### [(0)](#0-mbarrier_try_wait) `mbarrier_try_wait`
{: .no_toc }
```cuda
// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state;                                      // 5a.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state);
```

### [(1)](#1-mbarrier_try_wait) `mbarrier_try_wait`
{: .no_toc }
```cuda
// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    // 5b.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
```

### [(2)](#2-mbarrier_try_wait) `mbarrier_try_wait`
{: .no_toc }
```cuda
// mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state;                        // 6a.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
```

### [(3)](#3-mbarrier_try_wait) `mbarrier_try_wait`
{: .no_toc }
```cuda
// mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
```

## mbarrier.try_wait.parity

| C++ | PTX |
| [(0)](#0-mbarrier_try_wait_parity) `cuda::ptx::mbarrier_try_wait_parity`| `mbarrier.try_wait.parity.shared::cta.b64` |
| [(1)](#1-mbarrier_try_wait_parity) `cuda::ptx::mbarrier_try_wait_parity`| `mbarrier.try_wait.parity.shared::cta.b64` |
| [(2)](#2-mbarrier_try_wait_parity) `cuda::ptx::mbarrier_try_wait_parity`| `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` |
| [(2)](#2-mbarrier_try_wait_parity) `cuda::ptx::mbarrier_try_wait_parity`| `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64` |
| [(3)](#3-mbarrier_try_wait_parity) `cuda::ptx::mbarrier_try_wait_parity`| `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` |
| [(3)](#3-mbarrier_try_wait_parity) `cuda::ptx::mbarrier_try_wait_parity`| `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64` |


### [(0)](#0-mbarrier_try_wait_parity) `mbarrier_try_wait_parity`
{: .no_toc }
```cuda
// mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity;                                // 7a.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);
```

### [(1)](#1-mbarrier_try_wait_parity) `mbarrier_try_wait_parity`
{: .no_toc }
```cuda
// mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint;               // 7b.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
```

### [(2)](#2-mbarrier_try_wait_parity) `mbarrier_try_wait_parity`
{: .no_toc }
```cuda
// mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity;                  // 8a.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
```

### [(3)](#3-mbarrier_try_wait_parity) `mbarrier_try_wait_parity`
{: .no_toc }
```cuda
// mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 8b.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
```
