__global__ void test_getctarank(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // getctarank.shared::cluster.u32 dest, addr;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<uint32_t (*)(cuda::ptx::space_cluster_t, const void*)>(cuda::ptx::getctarank));));
#endif // __cccl_ptx_isa >= 780
}
