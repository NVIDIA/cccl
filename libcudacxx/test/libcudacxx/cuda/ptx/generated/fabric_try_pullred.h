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

__global__ void test_fabric_try_pullred(void** fn_ptr)
{
#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.and.b32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.xor.b32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.or.b32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.and.b64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_and_op_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.xor.b64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_xor_op_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.or.b64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_or_op_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.min.u32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.max.u32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.min.s32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.max.s32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.min.u64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.max.u64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.min.s64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.max.s64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::int64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.min.f16.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.max.f16.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.min.bf16.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_min_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.max.bf16.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_max_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.u32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::uint32_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.u64.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.f16.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.bf16.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.f32.sync
        // [dst], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               float*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.f16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __half*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

#endif // __cccl_ptx_isa >= 930

#if __cccl_ptx_isa >= 930
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121a,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121f,
    (
        // fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.acc::f32.bf16.sync
        // [dst_mem], [srcLeId, srcDataOff], size, [smem_bar], 0xFFFFFFFF;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::op_add_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_sys_t,
                               cuda::ptx::space_shared_t,
                               __nv_bfloat16*,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t*)>(cuda::ptx::fabric_try_pullred_acc_f32));));

#endif // __cccl_ptx_isa >= 930
}
