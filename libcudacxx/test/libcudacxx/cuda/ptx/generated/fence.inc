__global__ void test_fence(void** fn_ptr)
{
#if __cccl_ptx_isa >= 600
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // fence.sc.cta; // 1.
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.sc.gpu; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.sc.sys; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));
          // fence.acq_rel.cta; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_cta_t)>(cuda::ptx::fence));
          // fence.acq_rel.gpu; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_gpu_t)>(cuda::ptx::fence));
          // fence.acq_rel.sys; // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_sys_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 600

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.sc.cluster; // 2.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_sc_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));
          // fence.acq_rel.cluster; // 2.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acq_rel_t, cuda::ptx::scope_cluster_t)>(cuda::ptx::fence));));
#endif // __cccl_ptx_isa >= 780
}
