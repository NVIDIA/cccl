__global__ void test_mbarrier_try_wait_parity(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity; // 7a.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 7b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(uint64_t*, const uint32_t&, const uint32_t&)>(cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  waitComplete, [addr], phaseParity;                  //
        // 8a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_try_wait_parity));
          // mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  waitComplete, [addr], phaseParity; // 8a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint32_t&)>(
                cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; //
        // 8b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint32_t&, const uint32_t&)>(
            cuda::ptx::mbarrier_try_wait_parity));
          // mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  waitComplete, [addr], phaseParity,
          // suspendTimeHint; // 8b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint32_t&, const uint32_t&)>(
                cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 800
}
