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

__global__ void test_mbarrier_try_wait(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state; // 5a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::std::uint64_t*, const cuda::std::uint64_t&)>(cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    //
        // 5b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::std::uint64_t*, const cuda::std::uint64_t&, const cuda::std::uint32_t&)>(
            cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.acquire.cta.shared::cta.b64         waitComplete, [addr], state;                        //
        // 6a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::std::uint64_t*, const cuda::std::uint64_t&)>(
            cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.acquire.cluster.shared::cta.b64         waitComplete, [addr], state; // 6a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::std::uint64_t*, const cuda::std::uint64_t&)>(
                cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.acquire.cta.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      //
        // 6b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t,
                               cuda::ptx::scope_cta_t,
                               cuda::std::uint64_t*,
                               const cuda::std::uint64_t&,
                               const cuda::std::uint32_t&)>(cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.acquire.cluster.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;
          // // 6b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::std::uint64_t*,
                                   const cuda::std::uint64_t&,
                                   const cuda::std::uint32_t&)>(cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.relaxed.cta.shared::cta.b64 waitComplete, [addr], state, suspendTimeHint;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::std::uint64_t*,
                               const cuda::std::uint64_t&,
                               const cuda::std::uint32_t&)>(cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.relaxed.cluster.shared::cta.b64 waitComplete, [addr], state, suspendTimeHint;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_relaxed_t,
                                   cuda::ptx::scope_cluster_t,
                                   cuda::std::uint64_t*,
                                   const cuda::std::uint64_t&,
                                   const cuda::std::uint32_t&)>(cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.relaxed.cta.shared::cta.b64 waitComplete, [addr], state;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::std::uint64_t*, const cuda::std::uint64_t&)>(
            cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.relaxed.cluster.shared::cta.b64 waitComplete, [addr], state;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::std::uint64_t*, const cuda::std::uint64_t&)>(
                cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 860
}
