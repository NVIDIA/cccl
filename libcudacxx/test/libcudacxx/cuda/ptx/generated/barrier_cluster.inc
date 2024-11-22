__global__ void test_barrier_cluster(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.arrive;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::barrier_cluster_arrive));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.wait;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::barrier_cluster_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.arrive.release;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_release_t)>(cuda::ptx::barrier_cluster_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.arrive.relaxed;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_relaxed_t)>(cuda::ptx::barrier_cluster_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // barrier.cluster.wait.acquire;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::sem_acquire_t)>(cuda::ptx::barrier_cluster_wait));));
#endif // __cccl_ptx_isa >= 800
}
