__global__ void test_cp_reduce_async_bulk(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.and.b32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_and_op_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.or.b32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_or_op_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.xor.b32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_xor_op_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.u32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_min_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.u32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_max_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.inc.u32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_inc_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.dec.u32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_dec_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.s32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_min_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.s32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_max_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.s32 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               uint64_t*,
                               const uint64_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 [dstMem], [srcMem],
        // size, [rdsmem_bar]; // 2.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_cluster_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               int64_t*,
                               const int64_t*,
                               uint32_t,
                               uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b32  [dstMem], [srcMem], size; // 3.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_and_op_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));
          // cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b64  [dstMem], [srcMem], size; // 3.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_and_op_t,
                                   int64_t*,
                                   const int64_t*,
                                   uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b32  [dstMem], [srcMem], size; // 3.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_or_op_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));
          // cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b64  [dstMem], [srcMem], size; // 3.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_or_op_t,
                                   int64_t*,
                                   const int64_t*,
                                   uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32  [dstMem], [srcMem], size; // 3.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_xor_op_t,
                               int32_t*,
                               const int32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));
          // cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b64  [dstMem], [srcMem], size; // 3.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_xor_op_t,
                                   int64_t*,
                                   const int64_t*,
                                   uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_min_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_max_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.inc.u32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_inc_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.dec.u32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_dec_t,
                               uint32_t*,
                               const uint32_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_min_t, int32_t*, const int32_t*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_max_t, int32_t*, const int32_t*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_add_t, int32_t*, const int32_t*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_min_t,
                               uint64_t*,
                               const uint64_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_max_t,
                               uint64_t*,
                               const uint64_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               uint64_t*,
                               const uint64_t*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_min_t, int64_t*, const int64_t*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_max_t, int64_t*, const int64_t*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_add_t, float*, const float*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_add_t, double*, const double*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64  [dstMem], [srcMem], size; // 6.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, cuda::ptx::op_add_t, int64_t*, const int64_t*, uint32_t)>(
            cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800
}
