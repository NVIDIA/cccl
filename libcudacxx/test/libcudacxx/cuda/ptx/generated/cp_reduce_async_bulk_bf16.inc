__global__ void test_cp_reduce_async_bulk_bf16(void** fn_ptr)
{
#  if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_min_t,
                               __nv_bfloat16*,
                               const __nv_bfloat16*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#  endif // __cccl_ptx_isa >= 800

#  if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16  [dstMem], [srcMem], size; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_max_t,
                               __nv_bfloat16*,
                               const __nv_bfloat16*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#  endif // __cccl_ptx_isa >= 800

#  if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16  [dstMem], [srcMem], size; // 5.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               __nv_bfloat16*,
                               const __nv_bfloat16*,
                               uint32_t)>(cuda::ptx::cp_reduce_async_bulk));));
#  endif // __cccl_ptx_isa >= 800
}
