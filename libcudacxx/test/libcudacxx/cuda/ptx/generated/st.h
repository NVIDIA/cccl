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

__global__ void test_st(void** fn_ptr)
{
#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // st.global.b8 [addr], src;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t)>(cuda::ptx::st));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // st.global.b16 [addr], src;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t)>(cuda::ptx::st));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // st.global.b32 [addr], src;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t)>(cuda::ptx::st));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // st.global.b64 [addr], src;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t)>(cuda::ptx::st));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // st.global.b128 [addr], src;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2)>(cuda::ptx::st));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L2::cache_hint.b8 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t, uint64_t)>(cuda::ptx::st_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L2::cache_hint.b16 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t, uint64_t)>(cuda::ptx::st_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L2::cache_hint.b32 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t, uint64_t)>(cuda::ptx::st_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L2::cache_hint.b64 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t, uint64_t)>(cuda::ptx::st_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L2::cache_hint.b128 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2, uint64_t)>(
            cuda::ptx::st_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_normal.b8 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t)>(cuda::ptx::st_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_normal.b16 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t)>(cuda::ptx::st_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_normal.b32 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t)>(cuda::ptx::st_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_normal.b64 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t)>(cuda::ptx::st_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_normal.b128 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2)>(cuda::ptx::st_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_normal.L2::cache_hint.b8 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t, uint64_t)>(
          cuda::ptx::st_L1_evict_normal_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_normal.L2::cache_hint.b16 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t, uint64_t)>(
            cuda::ptx::st_L1_evict_normal_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_normal.L2::cache_hint.b32 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t, uint64_t)>(
            cuda::ptx::st_L1_evict_normal_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_normal.L2::cache_hint.b64 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t, uint64_t)>(
            cuda::ptx::st_L1_evict_normal_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_normal.L2::cache_hint.b128 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2, uint64_t)>(
            cuda::ptx::st_L1_evict_normal_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_unchanged.b8 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t)>(cuda::ptx::st_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_unchanged.b16 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t)>(cuda::ptx::st_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_unchanged.b32 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t)>(cuda::ptx::st_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_unchanged.b64 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t)>(cuda::ptx::st_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_unchanged.b128 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2)>(cuda::ptx::st_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_unchanged.L2::cache_hint.b8 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t, uint64_t)>(
          cuda::ptx::st_L1_evict_unchanged_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_unchanged.L2::cache_hint.b16 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t, uint64_t)>(
            cuda::ptx::st_L1_evict_unchanged_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_unchanged.L2::cache_hint.b32 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t, uint64_t)>(
            cuda::ptx::st_L1_evict_unchanged_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_unchanged.L2::cache_hint.b64 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t, uint64_t)>(
            cuda::ptx::st_L1_evict_unchanged_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_unchanged.L2::cache_hint.b128 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2, uint64_t)>(
            cuda::ptx::st_L1_evict_unchanged_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_first.b8 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t)>(cuda::ptx::st_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_first.b16 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t)>(cuda::ptx::st_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_first.b32 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t)>(cuda::ptx::st_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_first.b64 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t)>(cuda::ptx::st_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_first.b128 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2)>(cuda::ptx::st_L1_evict_first));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_first.L2::cache_hint.b8 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t, uint64_t)>(
          cuda::ptx::st_L1_evict_first_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_first.L2::cache_hint.b16 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t, uint64_t)>(
            cuda::ptx::st_L1_evict_first_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_first.L2::cache_hint.b32 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t, uint64_t)>(
            cuda::ptx::st_L1_evict_first_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_first.L2::cache_hint.b64 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t, uint64_t)>(
            cuda::ptx::st_L1_evict_first_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_first.L2::cache_hint.b128 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2, uint64_t)>(
            cuda::ptx::st_L1_evict_first_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // st.global.L1::evict_last.b8 [addr], src;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t)>(cuda::ptx::st_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_last.b16 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t)>(cuda::ptx::st_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_last.b32 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t)>(cuda::ptx::st_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_last.b64 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t)>(cuda::ptx::st_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::evict_last.b128 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2)>(cuda::ptx::st_L1_evict_last));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_last.L2::cache_hint.b8 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t, uint64_t)>(
          cuda::ptx::st_L1_evict_last_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_last.L2::cache_hint.b16 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t, uint64_t)>(
            cuda::ptx::st_L1_evict_last_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_last.L2::cache_hint.b32 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t, uint64_t)>(
            cuda::ptx::st_L1_evict_last_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_last.L2::cache_hint.b64 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t, uint64_t)>(
            cuda::ptx::st_L1_evict_last_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::evict_last.L2::cache_hint.b128 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2, uint64_t)>(
            cuda::ptx::st_L1_evict_last_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::no_allocate.b8 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t)>(cuda::ptx::st_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::no_allocate.b16 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t)>(cuda::ptx::st_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::no_allocate.b32 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t)>(cuda::ptx::st_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::no_allocate.b64 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t)>(cuda::ptx::st_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // st.global.L1::no_allocate.b128 [addr], src;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2)>(cuda::ptx::st_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::no_allocate.L2::cache_hint.b8 [addr], src, cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int8_t*, int8_t, uint64_t)>(
          cuda::ptx::st_L1_no_allocate_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::no_allocate.L2::cache_hint.b16 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int16_t*, int16_t, uint64_t)>(
            cuda::ptx::st_L1_no_allocate_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::no_allocate.L2::cache_hint.b32 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int32_t*, int32_t, uint64_t)>(
            cuda::ptx::st_L1_no_allocate_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::no_allocate.L2::cache_hint.b64 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, int64_t*, int64_t, uint64_t)>(
            cuda::ptx::st_L1_no_allocate_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // st.global.L1::no_allocate.L2::cache_hint.b128 [addr], src, cache_policy;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, longlong2*, longlong2, uint64_t)>(
            cuda::ptx::st_L1_no_allocate_L2_cache_hint));));
#endif // __cccl_ptx_isa >= 830
}
