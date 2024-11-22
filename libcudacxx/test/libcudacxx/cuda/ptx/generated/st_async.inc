__global__ void test_st_async(void** fn_ptr)
{
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [addr], value, [remote_bar];    // 1.
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(int32_t*, const int32_t&, uint64_t*)>(cuda::ptx::st_async));
          // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64 [addr], value, [remote_bar];    // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(int64_t*, const int64_t&, uint64_t*)>(cuda::ptx::st_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [addr], value, [remote_bar]; // 2.
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(int32_t*, const int32_t(&)[2], uint64_t*)>(cuda::ptx::st_async));
          // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 [addr], value, [remote_bar]; // 2.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(int64_t*, const int64_t(&)[2], uint64_t*)>(cuda::ptx::st_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];
                   // // 3.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(int32_t*, const int32_t(&)[4], uint64_t*)>(cuda::ptx::st_async));));
#endif // __cccl_ptx_isa >= 810
}
