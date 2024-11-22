__global__ void test_cp_async_bulk_tensor(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];// 1a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t(&)[1], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t(&)[1], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];// 1b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t(&)[2], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t(&)[2], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];// 1c.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t(&)[3], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3c.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t(&)[3], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];// 1d.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t(&)[4], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3d.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t(&)[4], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];// 1e.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t(&)[5], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3e.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t(&)[5], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800
}
