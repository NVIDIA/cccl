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

__global__ void test_tcgen05_mma_sp(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_fill));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_fill));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_use));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_use));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_collector_a_discard));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], a_desc,
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_collector_a_discard));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc,
        // [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [d_tmem], [a_tmem],
        // b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard));));
#endif // __cccl_ptx_isa >= 880
}
