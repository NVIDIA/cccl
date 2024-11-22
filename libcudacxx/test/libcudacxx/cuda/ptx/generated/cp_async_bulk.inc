__global__ void test_cp_async_bulk(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [smem_bar]; //
        // 1a. unicast
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const uint32_t&, uint64_t*)>(
            cuda::ptx::cp_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], [srcMem], size,
        // [rdsmem_bar]; // 2.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_shared_t, void*, const void*, const uint32_t&, uint64_t*)>(
            cuda::ptx::cp_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size; // 3.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, void*, const void*, const uint32_t&)>(
            cuda::ptx::cp_async_bulk));));
#endif // __cccl_ptx_isa >= 800
}
