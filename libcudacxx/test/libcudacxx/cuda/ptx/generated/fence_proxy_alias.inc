__global__ void test_fence_proxy_alias(void** fn_ptr)
{
#if __cccl_ptx_isa >= 750
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // fence.proxy.alias; // 4.
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::fence_proxy_alias));));
#endif // __cccl_ptx_isa >= 750
}
