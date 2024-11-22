__global__ void test_mbarrier_arrive_expect_tx(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 state, [addr], tx_count; // 8.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive_expect_tx));
          // mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 state, [addr], tx_count; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(cuda::ptx::sem_release_t,
                                       cuda::ptx::scope_cluster_t,
                                       cuda::ptx::space_shared_t,
                                       uint64_t*,
                                       const uint32_t&)>(cuda::ptx::mbarrier_arrive_expect_tx));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64   _, [addr], tx_count; // 9.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive_expect_tx));));
#endif // __cccl_ptx_isa >= 800
}
