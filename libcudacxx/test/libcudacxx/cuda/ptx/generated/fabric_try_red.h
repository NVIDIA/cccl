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

__global__ void test_fabric_try_red(void** fn_ptr)
{
#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.f16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.f16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.f16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.f16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.f16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.f16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.f16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.f16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.bf16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.bf16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.bf16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.bf16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.bf16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.bf16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.bf16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.bf16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.bf16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.bf16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.bf16
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.bf16
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const float*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const float*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f32
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const float*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f32
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const float*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const double*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const double*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f64
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const double*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f64
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const double*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_red_multimem_counted));));
#endif // __cccl_ptx_isa >= 930
}
