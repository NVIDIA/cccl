# tensormap.cp_fenceproxy

- PTX ISA: [`tensormap.cp_fenceproxy`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy)

| C++ | PTX |
| [(0)](#0-tensormap_cp_fenceproxy) `cuda::ptx::tensormap_cp_fenceproxy`| `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cta.sync.aligned` |
| [(0)](#0-tensormap_cp_fenceproxy) `cuda::ptx::tensormap_cp_fenceproxy`| `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cluster.sync.aligned` |
| [(0)](#0-tensormap_cp_fenceproxy) `cuda::ptx::tensormap_cp_fenceproxy`| `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned` |
| [(0)](#0-tensormap_cp_fenceproxy) `cuda::ptx::tensormap_cp_fenceproxy`| `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.sys.sync.aligned` |


### [(0)](#0-tensormap_cp_fenceproxy) `tensormap_cp_fenceproxy`
{: .no_toc }
```cuda
// tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.sem.scope.sync.aligned  [dst], [src], size; // PTX ISA 83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void tensormap_cp_fenceproxy(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  void* dst,
  const void* src,
  cuda::ptx::n32_t<N32> size);
```
