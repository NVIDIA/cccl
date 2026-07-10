// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_STMATRIX_H_
#define _CUDA_PTX_GENERATED_STMATRIX_H_

/*
// stmatrix.sync.aligned.m8n8.x1.shared.b16 [gmem_ptr], input_var; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8(
  B16* gmem_ptr,
  const uint32_t (&input_var)[1]);
*/
#if __cccl_ptx_isa >= 780
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[1])
{
  asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0])
               : "memory");
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x2.shared.b16 [gmem_ptr], input_var; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8(
  B16* gmem_ptr,
  const uint32_t (&input_var)[2]);
*/
#if __cccl_ptx_isa >= 780
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[2])
{
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0]), "r"(__input_var[1])
               : "memory");
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x4.shared.b16 [gmem_ptr], input_var; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8(
  B16* gmem_ptr,
  const uint32_t (&input_var)[4]);
*/
#if __cccl_ptx_isa >= 780
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[4])
{
  asm volatile(
    "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
    :
    : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0]), "r"(__input_var[1]), "r"(__input_var[2]), "r"(__input_var[3])
    : "memory");
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [gmem_ptr], input_var; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8_trans(
  B16* gmem_ptr,
  const uint32_t (&input_var)[1]);
*/
#if __cccl_ptx_isa >= 780
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8_trans(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[1])
{
  asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0])
               : "memory");
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [gmem_ptr], input_var; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8_trans(
  B16* gmem_ptr,
  const uint32_t (&input_var)[2]);
*/
#if __cccl_ptx_isa >= 780
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8_trans(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[2])
{
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0]), "r"(__input_var[1])
               : "memory");
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [gmem_ptr], input_var; // PTX ISA 78, SM_90
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void stmatrix_m8n8_trans(
  B16* gmem_ptr,
  const uint32_t (&input_var)[4]);
*/
#if __cccl_ptx_isa >= 780
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m8n8_trans(_B16* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[4])
{
  asm volatile(
    "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
    :
    : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0]), "r"(__input_var[1]), "r"(__input_var[2]), "r"(__input_var[3])
    : "memory");
}
#endif // __cccl_ptx_isa >= 780

/*
// stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input_var; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <typename B8,
enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void stmatrix_m16n8_trans(
  B8* gmem_ptr,
  const uint32_t (&input_var)[1]);
*/
#if __cccl_ptx_isa >= 860
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m16n8_trans(_B8* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[1])
{
  asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0])
               : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input_var; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <typename B8,
enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void stmatrix_m16n8_trans(
  B8* gmem_ptr,
  const uint32_t (&input_var)[2]);
*/
#if __cccl_ptx_isa >= 860
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m16n8_trans(_B8* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[2])
{
  asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};"
               :
               : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0]), "r"(__input_var[1])
               : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input_var; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f template <typename B8,
enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void stmatrix_m16n8_trans(
  B8* gmem_ptr,
  const uint32_t (&input_var)[4]);
*/
#if __cccl_ptx_isa >= 860
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void stmatrix_m16n8_trans(_B8* __gmem_ptr, const ::cuda::std::uint32_t (&__input_var)[4])
{
  asm volatile(
    "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};"
    :
    : "r"(__as_ptr_smem(__gmem_ptr)), "r"(__input_var[0]), "r"(__input_var[1]), "r"(__input_var[2]), "r"(__input_var[3])
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_STMATRIX_H_
