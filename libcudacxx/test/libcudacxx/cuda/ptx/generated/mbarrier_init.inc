__global__ void test_mbarrier_init(void** fn_ptr)
{
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // mbarrier.init.shared.b64 [addr], count;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_init));));
#endif // __cccl_ptx_isa >= 700
}
