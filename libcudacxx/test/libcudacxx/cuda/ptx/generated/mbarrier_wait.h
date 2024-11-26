__global__ void test_mbarrier_test_wait(void** fn_ptr)
{
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // mbarrier.test_wait.shared.b64 waitComplete, [addr], state; // 1.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint64_t&)>(cuda::ptx::mbarrier_test_wait));));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.acquire.cta.shared::cta.b64        waitComplete, [addr], state; // 2.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint64_t&)>(
            cuda::ptx::mbarrier_test_wait));
          // mbarrier.test_wait.acquire.cluster.shared::cta.b64        waitComplete, [addr], state; // 2.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint64_t&)>(
                cuda::ptx::mbarrier_test_wait));));
#endif // __cccl_ptx_isa >= 800
}
