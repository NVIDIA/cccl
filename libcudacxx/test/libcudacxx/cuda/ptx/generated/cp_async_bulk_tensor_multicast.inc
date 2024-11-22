__global__ void test_cp_async_bulk_tensor_multicast(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[1],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[2],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2c.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[3],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2d.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[4],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask; // 2e.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const int32_t(&)[5],
                               uint64_t*,
                               const uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800
}
