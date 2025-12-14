// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_multimem_ld_reduce(void** fn_ptr)
{
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.min.u32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.min.u32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.min.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.min.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.min.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.min.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.min.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.min.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.min.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.min.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.min.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.min.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.min.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.min.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.min.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.min.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.min.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.min.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.min.s32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.min.s32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.min.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.min.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.min.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.min.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.min.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.min.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.min.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.min.s64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.min.s64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.min.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.min.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.min.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.min.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.min.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.min.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.min.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_min_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.max.u32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.max.u32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.max.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.max.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.max.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.max.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.max.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.max.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.max.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.max.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.max.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.max.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.max.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.max.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.max.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.max.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.max.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.max.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.max.s32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.max.s32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.max.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.max.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.max.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.max.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.max.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.max.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.max.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.max.s64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.max.s64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.max.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.max.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.max.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.max.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.max.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.max.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.max.s64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_max_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.add.u32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.add.u32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.add.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.add.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.add.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.add.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.add.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.add.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.add.u32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::uint32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.add.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.add.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::uint64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.add.s32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.add.s32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.add.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.add.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.add.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.add.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.add.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.add.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.add.s32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.add.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.add.u64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.add.u64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.and.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.and.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.and.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.and.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.and.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.and.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.and.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.and.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.and.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_and_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.or.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.or.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.or.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.or.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.or.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.or.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.or.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.or.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.or.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_or_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.xor.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.xor.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int32_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.xor.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.xor.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.xor.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.xor.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.xor.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.xor.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.xor.b32 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int32_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_xor_op_t, const cuda::std::int32_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.and.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.and.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.and.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.and.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.and.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.and.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.and.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.and.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.and.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_and_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.or.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.or.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.or.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.or.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.or.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.or.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.or.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.or.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.or.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_or_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.weak.global.xor.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(cuda::ptx::sem_weak_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.ld_reduce.relaxed.cta.global.xor.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<int64_t (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
            cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.cluster.global.xor.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.gpu.global.xor.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.relaxed.sys.global.xor.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cta.global.xor.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.cluster.global.xor.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.gpu.global.xor.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));
          // multimem.ld_reduce.acquire.sys.global.xor.b64 dest, [addr];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<int64_t (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, cuda::ptx::op_xor_op_t, const cuda::std::int64_t*)>(
                cuda::ptx::multimem_ld_reduce));));
#endif // __cccl_ptx_isa >= 810
}
