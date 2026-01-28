// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_LDMATRIX_H_
#define _CUDA_PTX_GENERATED_LDMATRIX_H_

/*
// ldmatrix.sync.aligned.m8n8.x1.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
// .space     = { .shared }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void ldmatrix_m8n8(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[1],
  const B16* smem_ptr);
*/
#if __cccl_ptx_isa >= 650
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n8_is_not_supported_before_SM_75__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m8n8(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[1], const _B16* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 750
  asm("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
      : "=r"(__out[0])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n8_is_not_supported_before_SM_75__();
#  endif
}
#endif // __cccl_ptx_isa >= 650

/*
// ldmatrix.sync.aligned.m8n8.x2.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
// .space     = { .shared }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void ldmatrix_m8n8(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[2],
  const B16* smem_ptr);
*/
#if __cccl_ptx_isa >= 650
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n8_is_not_supported_before_SM_75__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m8n8(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[2], const _B16* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 750
  asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n8_is_not_supported_before_SM_75__();
#  endif
}
#endif // __cccl_ptx_isa >= 650

/*
// ldmatrix.sync.aligned.m8n8.x4.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
// .space     = { .shared }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void ldmatrix_m8n8(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[4],
  const B16* smem_ptr);
*/
#if __cccl_ptx_isa >= 650
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n8_is_not_supported_before_SM_75__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m8n8(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[4], const _B16* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 750
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n8_is_not_supported_before_SM_75__();
#  endif
}
#endif // __cccl_ptx_isa >= 650

/*
// ldmatrix.sync.aligned.m8n8.x1.trans.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
// .space     = { .shared }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void ldmatrix_m8n8_trans(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[1],
  const B16* smem_ptr);
*/
#if __cccl_ptx_isa >= 650
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n8_trans_is_not_supported_before_SM_75__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m8n8_trans(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[1], const _B16* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 750
  asm("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
      : "=r"(__out[0])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n8_trans_is_not_supported_before_SM_75__();
#  endif
}
#endif // __cccl_ptx_isa >= 650

/*
// ldmatrix.sync.aligned.m8n8.x2.trans.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
// .space     = { .shared }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void ldmatrix_m8n8_trans(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[2],
  const B16* smem_ptr);
*/
#if __cccl_ptx_isa >= 650
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n8_trans_is_not_supported_before_SM_75__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m8n8_trans(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[2], const _B16* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 750
  asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n8_trans_is_not_supported_before_SM_75__();
#  endif
}
#endif // __cccl_ptx_isa >= 650

/*
// ldmatrix.sync.aligned.m8n8.x4.trans.space.b16 out, [smem_ptr]; // PTX ISA 65, SM_75
// .space     = { .shared }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void ldmatrix_m8n8_trans(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[4],
  const B16* smem_ptr);
*/
#if __cccl_ptx_isa >= 650
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n8_trans_is_not_supported_before_SM_75__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void
ldmatrix_m8n8_trans(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[4], const _B16* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 750
  asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n8_trans_is_not_supported_before_SM_75__();
#  endif
}
#endif // __cccl_ptx_isa >= 650

/*
// ldmatrix.sync.aligned.m8n16.x1.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m8n16_b8x16_b6x16_p32(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[1],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n16_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void
ldmatrix_m8n16_b8x16_b6x16_p32(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[1], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {%0}, [%1];"
      : "=r"(__out[0])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n16_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m8n16.x1.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m8n16_b8x16_b4x16_p64(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[1],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n16_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void
ldmatrix_m8n16_b8x16_b4x16_p64(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[1], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 {%0}, [%1];"
      : "=r"(__out[0])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n16_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m8n16.x2.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m8n16_b8x16_b6x16_p32(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[2],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n16_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void
ldmatrix_m8n16_b8x16_b6x16_p32(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[2], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n16_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m8n16.x2.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m8n16_b8x16_b4x16_p64(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[2],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n16_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void
ldmatrix_m8n16_b8x16_b4x16_p64(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[2], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n16_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m8n16.x4.space.b8x16.b6x16_p32 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m8n16_b8x16_b6x16_p32(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[4],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n16_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void
ldmatrix_m8n16_b8x16_b6x16_p32(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[4], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n16_b8x16_b6x16_p32_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// ldmatrix.sync.aligned.m8n16.x4.space.b8x16.b4x16_p64 out, [smem_ptr]; // PTX ISA 86, SM_100a, SM_110a
// .space     = { .shared }
template <typename = void>
__device__ static inline void ldmatrix_m8n16_b8x16_b4x16_p64(
  cuda::ptx::space_shared_t,
  uint32_t (&out)[4],
  const void* smem_ptr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_ldmatrix_m8n16_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
template <typename = void>
_CCCL_DEVICE static inline void
ldmatrix_m8n16_b8x16_b4x16_p64(::cuda::ptx::space_shared_t, ::cuda::std::uint32_t (&__out)[4], const void* __smem_ptr)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100)
  asm("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__as_ptr_smem(__smem_ptr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ldmatrix_m8n16_b8x16_b4x16_p64_is_only_supported_on_SM_100a_110a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_LDMATRIX_H_
