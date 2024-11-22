__global__ void test_mbarrier_try_wait(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state; // 5a.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint64_t&)>(cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    //
        // 5b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(uint64_t*, const uint64_t&, const uint32_t&)>(cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.acquire.cta.shared::cta.b64         waitComplete, [addr], state;                        //
        // 6a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint64_t&)>(
            cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.acquire.cluster.shared::cta.b64         waitComplete, [addr], state; // 6a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint64_t&)>(
                cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.acquire.cta.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      //
        // 6b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint64_t&, const uint32_t&)>(
            cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.acquire.cluster.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;
          // // 6b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint64_t&, const uint32_t&)>(
                cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 800
}
