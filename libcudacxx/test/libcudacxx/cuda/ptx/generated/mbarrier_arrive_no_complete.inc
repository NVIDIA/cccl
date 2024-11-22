__global__ void test_mbarrier_arrive_no_complete(void** fn_ptr)
{
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<uint64_t (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_arrive_no_complete));));
#endif // __cccl_ptx_isa >= 700
}
