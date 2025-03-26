// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_MMA_H_
#define _CUDA_PTX_GENERATED_TCGEN05_MMA_H_

/*
// tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d;
// PTX ISA 86, SM_100a
// .kind      = { .kind::f16, .kind::tf32 }
// .cta_group = { .cta_group::1 }
template <int N32, cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_1_t,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[4],
  bool enable_input_d,
  cuda::ptx::n32_t<N32> scale_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a__();
template <int _N32, dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma(
  kind_t<_Kind> __kind,
  cta_group_1_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[4],
  bool __enable_input_d,
  n32_t<_N32> __scale_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32, "");
// __cta_group == cta_group_1 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d, %9;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d, %9;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d;
// PTX ISA 86, SM_100a
// .kind      = { .kind::f16, .kind::tf32 }
// .cta_group = { .cta_group::2 }
template <int N32, cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_2_t,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[8],
  bool enable_input_d,
  cuda::ptx::n32_t<N32> scale_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a__();
template <int _N32, dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma(
  kind_t<_Kind> __kind,
  cta_group_2_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[8],
  bool __enable_input_d,
  n32_t<_N32> __scale_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32, "");
// __cta_group == cta_group_2 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, PRED_enable_input_d, "
      "%13;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, PRED_enable_input_d, "
      "%13;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
// .cta_group = { .cta_group::1 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_1_t,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[4],
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma(
  kind_t<_Kind> __kind,
  cta_group_1_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[4],
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
// __cta_group == cta_group_1 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
// .cta_group = { .cta_group::2 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_2_t,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[8],
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma(
  kind_t<_Kind> __kind,
  cta_group_2_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[8],
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
// __cta_group == cta_group_2 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::i8 [%0], %1, %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a
// .kind      = { .kind::f16, .kind::tf32 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  cuda::ptx::n32_t<N32> scale_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a__();
template <int _N32, dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  n32_t<_N32> __scale_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::i8 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d;
// PTX ISA 86, SM_100a
// .kind      = { .kind::f16, .kind::tf32 }
// .cta_group = { .cta_group::1 }
template <int N32, cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_1_t,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[4],
  bool enable_input_d,
  cuda::ptx::n32_t<N32> scale_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a__();
template <int _N32, dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_tmem_a(
  kind_t<_Kind> __kind,
  cta_group_1_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[4],
  bool __enable_input_d,
  n32_t<_N32> __scale_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32, "");
// __cta_group == cta_group_1 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d, %9;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d, %9;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d;
// PTX ISA 86, SM_100a
// .kind      = { .kind::f16, .kind::tf32 }
// .cta_group = { .cta_group::2 }
template <int N32, cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_2_t,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[8],
  bool enable_input_d,
  cuda::ptx::n32_t<N32> scale_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a__();
template <int _N32, dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_tmem_a(
  kind_t<_Kind> __kind,
  cta_group_2_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[8],
  bool __enable_input_d,
  n32_t<_N32> __scale_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32, "");
// __cta_group == cta_group_2 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, PRED_enable_input_d, "
      "%13;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], [%1], %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d, %13;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
// .cta_group = { .cta_group::1 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_1_t,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[4],
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_tmem_a(
  kind_t<_Kind> __kind,
  cta_group_1_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[4],
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
// __cta_group == cta_group_1 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::i8 [%0], [%1], %2, %3, {%4, %5, %6, %7}, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
// .cta_group = { .cta_group::2 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_2_t,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  const uint32_t (&disable_output_lane)[8],
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_tmem_a(
  kind_t<_Kind> __kind,
  cta_group_2_t,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  const _CUDA_VSTD::uint32_t (&__disable_output_lane)[8],
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
// __cta_group == cta_group_2 (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], [%1], %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %12, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::i8 [%0], [%1], %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__disable_output_lane[0]),
        "r"(__disable_output_lane[1]),
        "r"(__disable_output_lane[2]),
        "r"(__disable_output_lane[3]),
        "r"(__disable_output_lane[4]),
        "r"(__disable_output_lane[5]),
        "r"(__disable_output_lane[6]),
        "r"(__disable_output_lane[7]),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a
// .kind      = { .kind::f16, .kind::tf32 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  cuda::ptx::n32_t<N32> scale_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a__();
template <int _N32, dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_tmem_a(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  n32_t<_N32> __scale_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "n"(__scale_input_d.value)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_tmem_a(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f16 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::i8 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::i8 [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem],
enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_block_scale_vec_1x_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem],
enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2x(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_block_scale_vec_2x_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2x(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2x_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem],
enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_block_scale_vec_4x_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem],
enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_tmem_a(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem],
enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2_tmem_a(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem],
enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_tmem_a(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_fill(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_collector_a_fill(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_fill(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2x_collector_a_fill(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_fill(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_collector_a_fill(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_use(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_collector_a_use(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_use(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2x_collector_a_use(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_use(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_collector_a_use(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], [%5], "
      "PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_lastuse(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_collector_a_lastuse(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_lastuse(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2x_collector_a_lastuse(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_lastuse(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_collector_a_lastuse(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_discard(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_collector_a_discard(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_discard(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2x_collector_a_discard(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2x_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_discard(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_collector_a_discard(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf8f6f4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard(
  cuda::ptx::kind_mxf8f6f4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard(
  kind_mxf8f6f4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf8f6f4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard(
  cuda::ptx::kind_t<Kind> kind,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind, dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard(
  kind_t<_Kind> __kind,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  static_assert(__kind == kind_mxf4 || __kind == kind_mxf4nvf4, "");
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_mxf4nvf4 && __cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc, b_desc, idesc,
[scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_101a
// .kind      = { .kind::mxf4nvf4 }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard(
  cuda::ptx::kind_mxf4nvf4_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  uint32_t scale_A_tmem,
  uint32_t scale_B_tmem,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard(
  kind_mxf4nvf4_t,
  cta_group_t<_Cta_Group> __cta_group,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  _CUDA_VSTD::uint32_t __scale_A_tmem,
  _CUDA_VSTD::uint32_t __scale_B_tmem,
  bool __enable_input_d)
{
  // __kind == kind_mxf4nvf4 (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__cta_group == cta_group_1)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__cta_group == cta_group_2)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard [%0], %1, %2, %3, [%4], "
      "[%5], PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(__scale_A_tmem),
        "r"(__scale_B_tmem),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_MMA_H_
