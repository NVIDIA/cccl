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

__global__ void test_tcgen05_cp(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.128x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.4x256b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.128x128b [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x256b_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_4x256b_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_128x128b_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_2_t, cuda::std::uint32_t, cuda::std::uint64_t)>(
            cuda::ptx::tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860
}
