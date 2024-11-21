__global__ void test_mbarrier_arrive(void** fn_ptr)
{
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint64_t (*)(uint64_t*)>(cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<uint64_t (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cta.shared::cta.b64                   state,  [addr];           // 3a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t*)>(
            cuda::ptx::mbarrier_arrive));
          // mbarrier.arrive.release.cluster.shared::cta.b64                   state,  [addr];           // 3a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_shared_t, uint64_t*)>(
                cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cta.shared::cta.b64                   state,  [addr], count;    // 3b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive));
          // mbarrier.arrive.release.cluster.shared::cta.b64                   state,  [addr], count;    // 3b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(cuda::ptx::sem_release_t,
                                       cuda::ptx::scope_cluster_t,
                                       cuda::ptx::space_shared_t,
                                       uint64_t*,
                                       const uint32_t&)>(cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [addr];                // 4a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t*)>(
            cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [addr], count;         // 4b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800
}
