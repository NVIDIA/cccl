__global__ void test_mbarrier_test_wait_parity(void** fn_ptr)
{
#if __cccl_ptx_isa >= 710
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity; // 3.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 710

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 waitComplete, [addr], phaseParity; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_test_wait_parity));
          // mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 waitComplete, [addr], phaseParity; // 4.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint32_t&)>(
                cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 800
}
