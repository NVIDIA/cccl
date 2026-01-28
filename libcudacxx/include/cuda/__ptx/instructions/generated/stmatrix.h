// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_STMATRIX_H_
#define _CUDA_PTX_GENERATED_STMATRIX_H_

/*
// stmatrix.sync.aligned.m8n8.x1.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8(
  B16* gmem_ptr,
  const uint32_t (&input)[1]);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_stmatrix_m8n8_is_not_supported_before_SM_90__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[1])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m8n8_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x2.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8(
  B16* gmem_ptr,
  const uint32_t (&input)[2]);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_stmatrix_m8n8_is_not_supported_before_SM_90__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[2])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0]), "r"(__input[1])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m8n8_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x4.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8(
  B16* gmem_ptr,
  const uint32_t (&input)[4]);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_stmatrix_m8n8_is_not_supported_before_SM_90__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[4])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0]), "r"(__input[1]), "r"(__input[2]), "r"(__input[3])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m8n8_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8_trans(
  B16* gmem_ptr,
  const uint32_t (&input)[1]);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_stmatrix_m8n8_trans_is_not_supported_before_SM_90__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8_trans(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[1])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m8n8_trans_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8_trans(
  B16* gmem_ptr,
  const uint32_t (&input)[2]);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_stmatrix_m8n8_trans_is_not_supported_before_SM_90__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8_trans(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[2])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0]), "r"(__input[1])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m8n8_trans_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [gmem_ptr], input; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8_trans(
  B16* gmem_ptr,
  const uint32_t (&input)[4]);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_stmatrix_m8n8_trans_is_not_supported_before_SM_90__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8_trans(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[4])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0]), "r"(__input[1]), "r"(__input[2]), "r"(__input[3])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m8n8_trans_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void stmatrix_m16n8_trans(
  B8* gmem_ptr,
  const uint32_t (&input)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_stmatrix_m16n8_trans_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m16n8_trans(_B8* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[1])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1200)                              \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1210) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)                                  \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(120) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(121)
  asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m16n8_trans_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void stmatrix_m16n8_trans(
  B8* gmem_ptr,
  const uint32_t (&input)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_stmatrix_m16n8_trans_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m16n8_trans(_B8* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[2])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1200)                              \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1210) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)                                  \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(120) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(121)
  asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0]), "r"(__input[1])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m16n8_trans_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void stmatrix_m16n8_trans(
  B8* gmem_ptr,
  const uint32_t (&input)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_stmatrix_m16n8_trans_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m16n8_trans(_B8* __gmem_ptr, const ::cuda::std::uint32_t (&__input)[4])
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1200)                              \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1210) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)                                  \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(120) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(121)
  asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input[0]), "r"(__input[1]), "r"(__input[2]), "r"(__input[3])
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_stmatrix_m16n8_trans_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_STMATRIX_H_
