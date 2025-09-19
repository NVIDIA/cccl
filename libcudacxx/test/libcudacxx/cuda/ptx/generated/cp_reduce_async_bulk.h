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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint64_t*,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int64_t*,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));
          // cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b64  [dstMem], [srcMem], size; // 3.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_and_op_t,
                                   cuda::std::int64_t*,
                                   const cuda::std::int64_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));
          // cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b64  [dstMem], [srcMem], size; // 3.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_or_op_t,
                                   cuda::std::int64_t*,
                                   const cuda::std::int64_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));
          // cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b64  [dstMem], [srcMem], size; // 3.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_xor_op_t,
                                   cuda::std::int64_t*,
                                   const cuda::std::int64_t*,
                                   cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint32_t*,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_min_t,
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_max_t,
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               cuda::std::int32_t*,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint64_t*,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint64_t*,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
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
                               cuda::std::uint64_t*,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_min_t,
                               cuda::std::int64_t*,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_max_t,
                               cuda::std::int64_t*,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               float*,
                               const float*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               double*,
                               const double*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64  [dstMem], [srcMem], size; // 6.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               cuda::std::int64_t*,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#endif // __cccl_ptx_isa >= 800
}
