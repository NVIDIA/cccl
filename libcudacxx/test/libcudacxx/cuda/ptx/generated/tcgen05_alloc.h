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

__global__ void test_tcgen05_alloc(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t*, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_alloc));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, nCols;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, const cuda::std::uint32_t&)>(
            cuda::ptx::tcgen05_dealloc));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t)>(cuda::ptx::tcgen05_relinquish_alloc_permit));));
#endif // __cccl_ptx_isa >= 860
}
