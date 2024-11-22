__global__ void test_red_async(void** fn_ptr)
{
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_inc_t, uint32_t*, const uint32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_dec_t, uint32_t*, const uint32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t, uint32_t*, const uint32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t, uint32_t*, const uint32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t, uint32_t*, const uint32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t, int32_t*, const int32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t, int32_t*, const int32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.s32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t, int32_t*, const int32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t, int32_t*, const int32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t, int32_t*, const int32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t, int32_t*, const int32_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [dest], value, [remote_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t, uint64_t*, const uint64_t&, uint64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [dest], value, [remote_bar];
        // // .u64 intentional
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t, int64_t*, const int64_t&, int64_t*)>(cuda::ptx::red_async));));
#endif // __cccl_ptx_isa >= 810
}
