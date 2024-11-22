__global__ void test_cp_async_bulk_commit_group(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.commit_group;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::cp_async_bulk_commit_group));));
#endif // __cccl_ptx_isa >= 800
}
