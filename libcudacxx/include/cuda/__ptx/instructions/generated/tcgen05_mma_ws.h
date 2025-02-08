// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_MMA_WS_H_
#define _CUDA_PTX_GENERATED_TCGEN05_MMA_WS_H_

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::use [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b0_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b0_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b0::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b0_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b0_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b0_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::use [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b1_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b1_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b1::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b1_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b1_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b1::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b1_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::use [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b2_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b2_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b2::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b2_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b2_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b2_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::fill [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::fill [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_fill(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_fill(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::fill [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_fill_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::use [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::use [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_use(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_use(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::use [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_use_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::lastuse [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::lastuse [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_lastuse(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_lastuse(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::lastuse [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_lastuse_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::discard [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86,
SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_collector_b3_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_mma_ws_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_collector_b3_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint64_t __a_desc,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "l"(__a_desc),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d,
zero_column_mask_desc; // PTX ISA 86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d,
  uint64_t zero_column_mask_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d,
  _CUDA_VSTD::uint64_t __zero_column_mask_desc)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d, %5;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d)),
        "l"(__zero_column_mask_desc)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.mma.ws.cta_group.kind.collector::b3::discard [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA
86, SM_100a, SM_101a
// .cta_group = { .cta_group::1 }
// .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
template <cuda::ptx::dot_kind Kind>
__device__ static inline void tcgen05_mma_ws_tmem_a_collector_b3_discard(
  cuda::ptx::cta_group_1_t,
  cuda::ptx::kind_t<Kind> kind,
  uint32_t d_tmem,
  uint32_t a_tmem,
  uint64_t b_desc,
  uint32_t idesc,
  bool enable_input_d);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
template <dot_kind _Kind>
_CCCL_DEVICE static inline void tcgen05_mma_ws_tmem_a_collector_b3_discard(
  cta_group_1_t,
  kind_t<_Kind> __kind,
  _CUDA_VSTD::uint32_t __d_tmem,
  _CUDA_VSTD::uint32_t __a_tmem,
  _CUDA_VSTD::uint64_t __b_desc,
  _CUDA_VSTD::uint32_t __idesc,
  bool __enable_input_d)
{
  // __cta_group == cta_group_1 (due to parameter type constraint)
  static_assert(__kind == kind_f16 || __kind == kind_tf32 || __kind == kind_f8f6f4 || __kind == kind_i8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  _CCCL_IF_CONSTEXPR (__kind == kind_f16)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_tf32)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_f8f6f4)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
      "}"
      :
      : "r"(__d_tmem),
        "r"(__a_tmem),
        "l"(__b_desc),
        "r"(__idesc),
        "r"(static_cast<_CUDA_VSTD::uint32_t>(__enable_input_d))
      : "memory");
  }
  else _CCCL_IF_CONSTEXPR (__kind == kind_i8)
  {
    asm volatile(
      "{\n\t .reg .pred PRED_enable_input_d; \n\t"
      "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b3::discard [%0], [%1], %2, %3, PRED_enable_input_d;\n\t"
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
  __cuda_ptx_tcgen05_mma_ws_tmem_a_collector_b3_discard_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_MMA_WS_H_
