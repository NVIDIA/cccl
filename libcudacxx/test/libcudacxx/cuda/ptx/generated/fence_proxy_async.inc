__global__ void test_fence_proxy_async(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // fence.proxy.async; // 5.
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::fence_proxy_async));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // fence.proxy.async.global; // 6.
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t)>(cuda::ptx::fence_proxy_async));
          // fence.proxy.async.shared::cluster; // 6.
            * fn_ptr++ =
              reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_cluster_t)>(cuda::ptx::fence_proxy_async));
          // fence.proxy.async.shared::cta; // 6.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t)>(cuda::ptx::fence_proxy_async));));
#endif // __cccl_ptx_isa >= 800
}
