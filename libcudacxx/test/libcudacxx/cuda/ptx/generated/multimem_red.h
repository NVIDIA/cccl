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

__global__ void test_multimem_red(void** fn_ptr)
{
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.min.u32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_min_t,
                               cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.min.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.min.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.min.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.min.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.min.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.min.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.min.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.min.u64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_min_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.min.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.min.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.min.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.min.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.min.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.min.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.min.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.min.s32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_min_t,
                               cuda::std::int32_t*,
                               cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.min.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.min.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.min.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.min.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.min.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.min.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.min.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.min.s64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_min_t,
                               cuda::std::int64_t*,
                               cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.min.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.min.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.min.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.min.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.min.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.min.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.min.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_min_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.max.u32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_max_t,
                               cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.max.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.max.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.max.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.max.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.max.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.max.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.max.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.max.u64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_max_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.max.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.max.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.max.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.max.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.max.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.max.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.max.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.max.s32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_max_t,
                               cuda::std::int32_t*,
                               cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.max.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.max.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.max.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.max.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.max.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.max.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.max.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.max.s64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_max_t,
                               cuda::std::int64_t*,
                               cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.max.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.max.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.max.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.max.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.max.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.max.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.max.s64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_max_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.add.u32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_add_t,
                               cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.add.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.add.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.add.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.add.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.add.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.add.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.add.u32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint32_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.add.u64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_add_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::uint64_t*,
                                   cuda::std::uint64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.add.s32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_add_t,
                               cuda::std::int32_t*,
                               cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.add.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.add.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.add.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.add.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.add.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.add.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.add.s32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.add.u64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_add_t,
                               cuda::std::int64_t*,
                               cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.add.u64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_add_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.and.b32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_and_op_t,
                               cuda::std::int32_t*,
                               cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.and.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.and.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.and.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.and.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.and.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.and.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.and.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.or.b32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_or_op_t,
                               cuda::std::int32_t*,
                               cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.or.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.or.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.or.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.or.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.or.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.or.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.or.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.xor.b32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_xor_op_t,
                               cuda::std::int32_t*,
                               cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.xor.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.xor.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.xor.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.xor.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.xor.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.xor.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.xor.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int32_t*,
                                   cuda::std::int32_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.and.b64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_and_op_t,
                               cuda::std::int64_t*,
                               cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.and.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.and.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.and.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.and.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.and.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.and.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.and.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.or.b64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_or_op_t,
                               cuda::std::int64_t*,
                               cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.or.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.or.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.or.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.or.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.or.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.or.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.or.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.red.relaxed.cta.global.xor.b64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::ptx::op_xor_op_t,
                               cuda::std::int64_t*,
                               cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.cluster.global.xor.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.gpu.global.xor.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.relaxed.sys.global.xor.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cta.global.xor.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cta_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.cluster.global.xor.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.gpu.global.xor.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_gpu_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));
          // multimem.red.release.sys.global.xor.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_release_t,
                                   cuda::ptx::scope_sys_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   cuda::std::int64_t)>(cuda::ptx::multimem_red));));
#endif // __cccl_ptx_isa >= 810
}
