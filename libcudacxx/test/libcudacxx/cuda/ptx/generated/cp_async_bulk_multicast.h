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

__global__ void test_cp_async_bulk_multicast(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint16_t&)>(cuda::ptx::cp_async_bulk));));

#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b [dstMem], [srcMem],
        // size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::80000000
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_80000000_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::80000000
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_80000000_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::8000
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_8000_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::8000
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_8000_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::80
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_80_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::80
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_80_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::8
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_8_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::8
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_16bytes_8_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_element::ff
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_element_ff_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_element::ff
        // [dstMem], [srcMem], size, [smem_bar], ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::mbarrier_report_valid_per_element_ff_t,
                               void*,
                               const void*,
                               const cuda::std::uint32_t&,
                               cuda::std::uint64_t*,
                               const cuda::std::uint32_t&)>(cuda::ptx::cp_async_bulk_multicast_32b));));

#endif // __cccl_ptx_isa >= 940
}
