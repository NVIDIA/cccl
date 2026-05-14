// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_MMA_SP_H_
#define _CUDA_PTX_GENERATED_TCGEN05_MMA_SP_H_

/*
// tcgen05.mma.sp.cta_group.kind [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .kind      = { .kind::f16, .kind::f8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_f8f6f4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_f16 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_f16 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::f16 [%0], %1, %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_f8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [%0], %1, %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_f8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [%0], %1, %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.sp.cta_group.kind [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .kind      = { .kind::f16, .kind::f8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_tmem_a(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_f8f6f4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_f16 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], [%1], %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_f16 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::f16 [%0], [%1], %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_f8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [%0], [%1], %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_f8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %5, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [%0], [%1], %2, [%3], %4, PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem],
[scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem],
[scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [%0], %1, %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem],
[scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_tmem_a(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem],
[scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32 [%0], [%1], %2, [%3], %4, [%5], [%6], "
      "PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_fill(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_collector_a_fill(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill [%0], [%1], %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill [%0], [%1], %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_use(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_collector_a_use(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f,
SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use [%0], [%1], %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use [%0], [%1], %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_lastuse(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_collector_a_lastuse(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_discard(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_collector_a_discard(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint64_t __a_desc,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [%0], %1, %2, [%3], %4, [%5], "
      "[%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [%0], %1, %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard(
  ::cuda::ptx::kind_mxf4nvf4_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem],
idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t sp_info_tmem,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_kind _Kind, ::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
  ::cuda::ptx::kind_t<_Kind> __kind,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  ::cuda::std::uint32_t __d_tmem,
  ::cuda::std::uint32_t __a_tmem,
  ::cuda::std::uint64_t __b_desc,
  ::cuda::std::uint32_t __sp_info_tmem,
  ::cuda::std::uint32_t __idesc,
  ::cuda::std::uint32_t __scale_A_tmem,
  ::cuda::std::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf8f6f4 || __kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
  if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
  else if constexpr (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t"
      ".reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %7, 0; \n\t"
      "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard [%0], [%1], %2, [%3], %4, "
      "[%5], [%6], PRED_enable_input_d; \n"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__sp_info_tmem),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<::cuda::std::uint32_t>(__enable_input_d))
      : "memory");
  }
}
#endif // __cccl_ptx_isa >= 880

#endif // _CUDA_PTX_GENERATED_TCGEN05_MMA_SP_H_
