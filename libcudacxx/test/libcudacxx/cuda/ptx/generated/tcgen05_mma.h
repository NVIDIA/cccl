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

__global__ void test_tcgen05_mma(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d,
        // scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[4],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               const cuda::std::uint32_t (&)[8],
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::ptx::n32_t<0>)>(cuda::ptx::tcgen05_mma_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f16_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::tf32 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_tf32_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.cta_group::2.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_f8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::i8 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_i8_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc,
        // [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc,
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
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_collector_a_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2x_collector_a_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_collector_a_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf8f6f4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc,
        // idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_1_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc,
        // b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::kind_mxf4nvf4_t,
                               cuda::ptx::cta_group_2_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard));));
#endif // __cccl_ptx_isa >= 860
}
