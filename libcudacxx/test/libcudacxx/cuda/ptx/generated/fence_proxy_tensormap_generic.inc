__global__ void test_fence_proxy_tensormap_generic(void** fn_ptr)
{
#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.proxy.tensormap::generic.release.cta; // 7.
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t)>(
          cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.release.cluster; // 7.
            * fn_ptr++ =
              reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t)>(
                cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.release.gpu; // 7.
            * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t)>(
              cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.release.sys; // 7.
            * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t)>(
              cuda::ptx::fence_proxy_tensormap_generic));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.proxy.tensormap::generic.acquire.cta [addr], size; // 8.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, const void*, cuda::ptx::n32_t<128>)>(
            cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.acquire.cluster [addr], size; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.acquire.gpu [addr], size; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_gpu_t, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::fence_proxy_tensormap_generic));
          // fence.proxy.tensormap::generic.acquire.sys [addr], size; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_sys_t, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::fence_proxy_tensormap_generic));));
#endif // __cccl_ptx_isa >= 830
}
