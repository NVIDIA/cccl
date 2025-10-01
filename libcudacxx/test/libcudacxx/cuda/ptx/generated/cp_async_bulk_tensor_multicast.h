// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_cp_async_bulk_tensor_multicast(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem],
        // [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));),
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
        // [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_2_t,
                               void*,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800
}
