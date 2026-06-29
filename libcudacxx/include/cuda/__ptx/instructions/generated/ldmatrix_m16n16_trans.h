// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_LDMATRIX_M16N16_TRANS_H_
#define _CUDA_PTX_GENERATED_LDMATRIX_M16N16_TRANS_H_

/*
// ldmatrix.sync.aligned.m16n16.x1.trans.space.b8 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void ldmatrix_m16n16_trans(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[2],
  const B8* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m16n16_trans_is_only_supported_on_SM_100a_110a__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m16n16_trans(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[2], const _B8* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m16n16_trans_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m16n16.x1.trans.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m16n16_trans_b8x16_b6x16_p32(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[1],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void ldmatrix_m16n16_trans_b8x16_b6x16_p32(
  ::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[1], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 {%0}, [%1];"
      : "=r"(__out[0])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m16n16.x1.trans.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m16n16_trans_b8x16_b4x16_p64(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[1],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void ldmatrix_m16n16_trans_b8x16_b4x16_p64(
  ::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[1], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64 {%0}, [%1];"
      : "=r"(__out[0])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m16n16.x2.trans.space.b8 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void ldmatrix_m16n16_trans(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[4],
  const B8* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m16n16_trans_is_only_supported_on_SM_100a_110a__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m16n16_trans(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[4], const _B8* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m16n16_trans_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m16n16.x2.trans.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m16n16_trans_b8x16_b6x16_p32(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[2],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void ldmatrix_m16n16_trans_b8x16_b6x16_p32(
  ::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[2], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m16n16.x2.trans.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m16n16_trans_b8x16_b4x16_p64(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[2],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void ldmatrix_m16n16_trans_b8x16_b4x16_p64(
  ::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[2], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m16n16_trans_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_LDMATRIX_M16N16_TRANS_H_
