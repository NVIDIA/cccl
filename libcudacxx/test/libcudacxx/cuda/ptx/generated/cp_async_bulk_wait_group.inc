__global__ void test_cp_async_bulk_wait_group(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // cp.async.bulk.wait_group N;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::n32_t<128>)>(cuda::ptx::cp_async_bulk_wait_group));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // cp.async.bulk.wait_group.read N;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::n32_t<128>)>(cuda::ptx::cp_async_bulk_wait_group_read));));
#endif // __cccl_ptx_isa >= 800
}
