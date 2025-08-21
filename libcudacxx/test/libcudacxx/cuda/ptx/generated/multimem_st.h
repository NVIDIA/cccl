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

__global__ void test_multimem_st(void** fn_ptr)
{
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.st.weak.global.b32 [addr], val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_weak_t, cuda::std::int32_t*, cuda::std::int32_t)>(
            cuda::ptx::multimem_st));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.st.relaxed.cta.global.b32 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::std::int32_t*, cuda::std::int32_t)>(
            cuda::ptx::multimem_st));
          // multimem.st.relaxed.cluster.global.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::std::int32_t*, cuda::std::int32_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.relaxed.gpu.global.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::std::int32_t*, cuda::std::int32_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.relaxed.sys.global.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::std::int32_t*, cuda::std::int32_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.cta.global.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::std::int32_t*, cuda::std::int32_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.cluster.global.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::std::int32_t*, cuda::std::int32_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.gpu.global.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t, cuda::std::int32_t*, cuda::std::int32_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.sys.global.b32 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t, cuda::std::int32_t*, cuda::std::int32_t)>(
                cuda::ptx::multimem_st));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.st.weak.global.b64 [addr], val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_weak_t, cuda::std::int64_t*, cuda::std::int64_t)>(
            cuda::ptx::multimem_st));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // multimem.st.relaxed.cta.global.b64 [addr], val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::std::int64_t*, cuda::std::int64_t)>(
            cuda::ptx::multimem_st));
          // multimem.st.relaxed.cluster.global.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::std::int64_t*, cuda::std::int64_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.relaxed.gpu.global.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_gpu_t, cuda::std::int64_t*, cuda::std::int64_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.relaxed.sys.global.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_sys_t, cuda::std::int64_t*, cuda::std::int64_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.cta.global.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::std::int64_t*, cuda::std::int64_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.cluster.global.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::std::int64_t*, cuda::std::int64_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.gpu.global.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t, cuda::std::int64_t*, cuda::std::int64_t)>(
                cuda::ptx::multimem_st));
          // multimem.st.release.sys.global.b64 [addr], val;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t, cuda::std::int64_t*, cuda::std::int64_t)>(
                cuda::ptx::multimem_st));));
#endif // __cccl_ptx_isa >= 810
}
