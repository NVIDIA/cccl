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

__global__ void test_fabric_try_put(void** fn_ptr)
{
#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const void*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_put));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar], bytemask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const void*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint16_t)>(cuda::ptx::fabric_try_put_cp_mask));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const void*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_put_counted));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const void*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_put_multimem));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128
        // [dstLeId, dstDataOff], [srcMem], size, [smem_bar], bytemask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               const void*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint16_t)>(cuda::ptx::fabric_try_put_multimem_cp_mask));));
#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128
        // [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               const void*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_put_multimem_counted));));
#endif // __cccl_ptx_isa >= 930
}
