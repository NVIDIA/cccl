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

__global__ void test_mbarrier_test_wait_parity(void** fn_ptr)
{
#if __cccl_ptx_isa >= 710
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<bool (*)(cuda::std::uint64_t*, const cuda::std::uint32_t&)>(
          cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 710

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 waitComplete, [addr], phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, cuda::std::uint64_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::mbarrier_test_wait_parity));
          // mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 waitComplete, [addr], phaseParity;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, cuda::std::uint64_t*, const cuda::std::uint32_t&)>(
                cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.relaxed.cta.shared::cta.b64 waitComplete, [addr], phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cta_t, cuda::std::uint64_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::mbarrier_test_wait_parity));
          // mbarrier.test_wait.parity.relaxed.cluster.shared::cta.b64 waitComplete, [addr], phaseParity;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_relaxed_t, cuda::ptx::scope_cluster_t, cuda::std::uint64_t*, const cuda::std::uint32_t&)>(
                cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 waitComplete|isReportSeen, [addr],
        // phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_primary_t,
                               cuda::ptx::sem_acquire_t,
                               cuda::ptx::scope_cta_t,
                               bool& isReportSeen,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::primary.acquire.cluster.shared::cta.b64 waitComplete|isReportSeen,
        // [addr], phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_primary_t,
                               cuda::ptx::sem_acquire_t,
                               cuda::ptx::scope_cluster_t,
                               bool& isReportSeen,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::primary.relaxed.cta.shared::cta.b64 waitComplete|isReportSeen, [addr],
        // phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_primary_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               bool& isReportSeen,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::primary.relaxed.cluster.shared::cta.b64 waitComplete|isReportSeen,
        // [addr], phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_primary_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               bool& isReportSeen,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::conditional.acquire.cta.shared::cta.b64 waitComplete, [addr],
        // phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_conditional_t,
                               cuda::ptx::sem_acquire_t,
                               cuda::ptx::scope_cta_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::conditional.acquire.cluster.shared::cta.b64 waitComplete, [addr],
        // phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_conditional_t,
                               cuda::ptx::sem_acquire_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::conditional.relaxed.cta.shared::cta.b64 waitComplete, [addr],
        // phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_conditional_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cta_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.phase_type::conditional.relaxed.cluster.shared::cta.b64 waitComplete, [addr],
        // phaseParity;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::mbarrier_phase_conditional_t,
                               cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 940
}
