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

__global__ void test_mbarrier_arrive_drop(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.release.cta.shared::cta.b64 state, [addr], count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_release_t,
                                              cuda::ptx::scope_cta_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.release.cluster.shared::cta.b64 state, [addr], count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_release_t,
                                              cuda::ptx::scope_cluster_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.release.cluster.shared::cluster.b64 _, [addr], count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_release_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.relaxed.cta.shared::cta.b64 state, [addr], count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_relaxed_t,
                                              cuda::ptx::scope_cta_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.relaxed.cluster.shared::cta.b64 state, [addr], count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_relaxed_t,
                                              cuda::ptx::scope_cluster_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.relaxed.cluster.shared::cluster.b64 _, [addr], count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // mbarrier.arrive_drop.release.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr], count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_release_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_multicast));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // mbarrier.arrive_drop.release.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr], count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_release_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_multicast));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // mbarrier.arrive_drop.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr], count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_multicast));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // mbarrier.arrive_drop.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr], count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_multicast));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // mbarrier.arrive_drop.expect_tx.release.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr],
        // tx_count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_release_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx_multicast));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // mbarrier.arrive_drop.expect_tx.release.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr],
        // tx_count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_release_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx_multicast));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr],
        // tx_count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx_multicast));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cluster.multicast::cluster::32b.b64 _, [addr],
        // tx_count, ctaMask;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx_multicast));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // mbarrier.arrive_drop.noComplete.release.cta.shared::cta.b64 state, [addr], count;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<cuda::std::uint64_t (*)(cuda::std::uint64_t*, cuda::std::uint32_t)>(
            cuda::ptx::mbarrier_arrive_drop_no_complete));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.expect_tx.release.cta.shared::cta.b64 state, [addr], tx_count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_release_t,
                                              cuda::ptx::scope_cta_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.expect_tx.release.cluster.shared::cta.b64 state, [addr], tx_count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_release_t,
                                              cuda::ptx::scope_cluster_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.expect_tx.release.cluster.shared::cluster.b64 _, [addr], tx_count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_release_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.expect_tx.relaxed.cta.shared::cta.b64 state, [addr], tx_count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_relaxed_t,
                                              cuda::ptx::scope_cta_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx));));
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cta.b64 state, [addr], tx_count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint64_t (*)(cuda::ptx::sem_relaxed_t,
                                              cuda::ptx::scope_cluster_t,
                                              cuda::ptx::space_shared_t,
                                              cuda::std::uint64_t*,
                                              cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cluster.b64 _, [addr], tx_count;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::sem_relaxed_t,
                               cuda::ptx::scope_cluster_t,
                               cuda::ptx::space_cluster_t,
                               cuda::std::uint64_t*,
                               cuda::std::uint32_t)>(cuda::ptx::mbarrier_arrive_drop_expect_tx));));
#endif // __cccl_ptx_isa >= 860
}
