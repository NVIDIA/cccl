__global__ void test_cp_async_bulk_multicast(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const uint32_t&,
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk));));
#endif // __cccl_ptx_isa >= 800
}
