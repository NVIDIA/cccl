# fence

- PTX ISA: [`fence`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## fence

| C++ | PTX |
| [(0)](#0-fence) `cuda::ptx::fence`| `fence.sc.cta;` |
| [(0)](#0-fence) `cuda::ptx::fence`| `fence.sc.gpu;` |
| [(0)](#0-fence) `cuda::ptx::fence`| `fence.sc.sys;` |
| [(0)](#0-fence) `cuda::ptx::fence`| `fence.acq_rel.cta;` |
| [(0)](#0-fence) `cuda::ptx::fence`| `fence.acq_rel.gpu;` |
| [(0)](#0-fence) `cuda::ptx::fence`| `fence.acq_rel.sys;` |
| [(1)](#1-fence) `cuda::ptx::fence`| `fence.sc.cluster;` |
| [(1)](#1-fence) `cuda::ptx::fence`| `fence.acq_rel.cluster;` |


### [(0)](#0-fence) `fence`
{: .no_toc }
```cuda
// fence{.sem}.scope; // 1. PTX ISA 60, SM_70
// .sem       = { .sc, .acq_rel }
// .scope     = { .cta, .gpu, .sys }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope);
```

### [(1)](#1-fence) `fence`
{: .no_toc }
```cuda
// fence{.sem}.scope; // 2. PTX ISA 78, SM_90
// .sem       = { .sc, .acq_rel }
// .scope     = { .cluster }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_cluster_t);
```

## fence.mbarrier_init

| C++ | PTX |
| [(0)](#0-fence_mbarrier_init) `cuda::ptx::fence_mbarrier_init`| `fence.mbarrier_init.release.cluster;` |


### [(0)](#0-fence_mbarrier_init) `fence_mbarrier_init`
{: .no_toc }
```cuda
// fence.mbarrier_init.sem.scope; // 3. PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
template <typename=void>
__device__ static inline void fence_mbarrier_init(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t);
```

## fence.proxy.alias

| C++ | PTX |
| [(0)](#0-fence_proxy_alias) `cuda::ptx::fence_proxy_alias`| `fence.proxy.alias;` |


### [(0)](#0-fence_proxy_alias) `fence_proxy_alias`
{: .no_toc }
```cuda
// fence.proxy.alias; // 4. PTX ISA 75, SM_70
template <typename=void>
__device__ static inline void fence_proxy_alias();
```

## fence.proxy.async

| C++ | PTX |
| [(0)](#0-fence_proxy_async) `cuda::ptx::fence_proxy_async`| `fence.proxy.async;` |
| [(1)](#1-fence_proxy_async) `cuda::ptx::fence_proxy_async`| `fence.proxy.async.global;` |
| [(1)](#1-fence_proxy_async) `cuda::ptx::fence_proxy_async`| `fence.proxy.async.shared::cluster;` |
| [(1)](#1-fence_proxy_async) `cuda::ptx::fence_proxy_async`| `fence.proxy.async.shared::cta;` |


### [(0)](#0-fence_proxy_async) `fence_proxy_async`
{: .no_toc }
```cuda
// fence.proxy.async; // 5. PTX ISA 80, SM_90
template <typename=void>
__device__ static inline void fence_proxy_async();
```

### [(1)](#1-fence_proxy_async) `fence_proxy_async`
{: .no_toc }
```cuda
// fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
// .space     = { .global, .shared::cluster, .shared::cta }
template <cuda::ptx::dot_space Space>
__device__ static inline void fence_proxy_async(
  cuda::ptx::space_t<Space> space);
```

## fence.proxy.tensormap

| C++ | PTX |
| [(0)](#0-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.release.cta;` |
| [(0)](#0-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.release.cluster;` |
| [(0)](#0-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.release.gpu;` |
| [(0)](#0-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.release.sys;` |
| [(1)](#1-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.acquire.cta` |
| [(1)](#1-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.acquire.cluster` |
| [(1)](#1-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.acquire.gpu` |
| [(1)](#1-fence_proxy_tensormap_generic) `cuda::ptx::fence_proxy_tensormap_generic`| `fence.proxy.tensormap::generic.acquire.sys` |


### [(0)](#0-fence_proxy_tensormap_generic) `fence_proxy_tensormap_generic`
{: .no_toc }
```cuda
// fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope);
```

### [(1)](#1-fence_proxy_tensormap_generic) `fence_proxy_tensormap_generic`
{: .no_toc }
```cuda
// fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  const void* addr,
  cuda::ptx::n32_t<N32> size);
```
