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

__global__ void test_tcgen05_mma_ws(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b0_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b0_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b1_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b1_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b2_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b2_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_fill));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_use));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_lastuse));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
        // zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], a_desc, b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_collector_b3_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d, zero_column_mask_desc;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool,
                               cuda::std::uint64_t)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f16_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_tf32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_f8f6f4_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc,
        // enable_input_d;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::cta_group_1_t,
                               cuda::ptx::kind_i8_t,
                               cuda::std::uint32_t,
                               cuda::std::uint32_t,
                               cuda::std::uint64_t,
                               cuda::std::uint32_t,
                               bool)>(cuda::ptx::tcgen05_mma_ws_tmem_a_collector_b3_discard));));
#endif // __cccl_ptx_isa >= 860
}
