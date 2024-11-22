__global__ void test_fence_mbarrier_init(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.mbarrier_init.release.cluster; // 3.
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t)>(
          cuda::ptx::fence_mbarrier_init));));
#endif // __cccl_ptx_isa >= 800
}
