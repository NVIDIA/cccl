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

__global__ void test_ld(void** fn_ptr)
{
#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.b16 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // ld.global.b128 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::64B.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::64B.b16 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::64B.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::64B.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::128B.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::128B.b16 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::128B.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.L2::128B.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // ld.global.L2::256B.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // ld.global.L2::256B.b16 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // ld.global.L2::256B.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // ld.global.L2::256B.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_normal.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_normal.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_normal.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_normal.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_normal.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_normal.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_normal.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_normal.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_normal.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_normal.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_normal.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_unchanged.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_unchanged.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_unchanged.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_unchanged.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_unchanged.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_unchanged.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_unchanged.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_unchanged.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_unchanged.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_unchanged.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_unchanged.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_first.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_first.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_first.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_first.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_first.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_first));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_first.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_first.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_first.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_first.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_first.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_first.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_last.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_last.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_last.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_last.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::evict_last.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_last));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::evict_last.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_last.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_last.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_last.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_last.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::evict_last.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::no_allocate.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::no_allocate.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::no_allocate.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::no_allocate.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.L1::no_allocate.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.L1::no_allocate.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::no_allocate.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::no_allocate.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::no_allocate.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::no_allocate.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.L1::no_allocate.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.nc.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.nc.b16 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.nc.b32 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 100
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // ld.global.nc.b64 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc));));
#endif // __cccl_ptx_isa >= 100

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (
        // ld.global.nc.b128 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.nc.L2::64B.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ld.global.nc.L2::128B.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // ld.global.nc.L2::256B.b8 dest, [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_normal.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_normal.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_normal.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_normal.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_normal.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_normal));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_normal.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_normal.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_normal.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_normal.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_normal.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_normal.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_normal_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_unchanged.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_unchanged.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_unchanged.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_unchanged.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_unchanged.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<longlong2 (*)(const longlong2*)>(
                     cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<longlong2 (*)(const longlong2*)>(
                     cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_unchanged.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<longlong2 (*)(const longlong2*)>(
                     cuda::ptx::ld_global_nc_L1_evict_unchanged_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_first.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_first.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_first.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_first.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_first));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_first.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_first));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_first.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_first.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_first.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_first.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_first.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_first.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_first_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_last.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_last.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_last.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_last.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::evict_last.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_last));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::evict_last.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_last.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_last.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_last.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_last.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::evict_last.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_evict_last_L2_256B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::no_allocate.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::no_allocate.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::no_allocate.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::no_allocate.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // ld.global.nc.L1::no_allocate.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_no_allocate));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::64B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::64B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::64B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::64B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::64B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_64B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::128B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::128B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::128B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::128B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_75,
               (
                   // ld.global.nc.L1::no_allocate.L2::128B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_128B));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::no_allocate.L2::256B.b8 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int8_t (*)(const int8_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::no_allocate.L2::256B.b16 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int16_t (*)(const int16_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::no_allocate.L2::256B.b32 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int32_t (*)(const int32_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::no_allocate.L2::256B.b64 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<int64_t (*)(const int64_t*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // ld.global.nc.L1::no_allocate.L2::256B.b128 dest, [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<longlong2 (*)(const longlong2*)>(cuda::ptx::ld_global_nc_L1_no_allocate_L2_256B));));
#endif // __cccl_ptx_isa >= 830
}
