// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_SHR_H_
#define _CUDA_PTX_GENERATED_SHR_H_

/*
// shr.b16 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 shr(
  B16 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 shr(_B16 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  static_assert(sizeof(_B16) == 2);
  static_assert(sizeof(_B16) == 2);
  ::cuda::std::uint16_t __dest;
  asm("shr.b16 %0, %1, %2;"
      : "=h"(__dest)
      : "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_B16*>(&__dest);
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.b32 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4 && !(is_integral_v<B32> && is_signed_v<B32>), bool> = true>
__device__ static inline B32 shr(
  B32 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
template <
  typename _B32,
  ::cuda::std::enable_if_t<sizeof(_B32) == 4 && !(::cuda::std::is_integral_v<_B32> && ::cuda::std::is_signed_v<_B32>),
                           bool> = true>
_CCCL_DEVICE static inline _B32 shr(_B32 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("shr.b32 %0, %1, %2;"
      : "=r"(__dest)
      : "r"(*reinterpret_cast<const ::cuda::std::int32_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_B32*>(&__dest);
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.b64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename B64, enable_if_t<sizeof(B64) == 8 && !(is_integral_v<B64> && is_signed_v<B64>), bool> = true>
__device__ static inline B64 shr(
  B64 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
template <
  typename _B64,
  ::cuda::std::enable_if_t<sizeof(_B64) == 8 && !(::cuda::std::is_integral_v<_B64> && ::cuda::std::is_signed_v<_B64>),
                           bool> = true>
_CCCL_DEVICE static inline _B64 shr(_B64 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  ::cuda::std::uint64_t __dest;
  asm("shr.b64 %0, %1, %2;"
      : "=l"(__dest)
      : "l"(*reinterpret_cast<const ::cuda::std::int64_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_B64*>(&__dest);
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.s16 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline int16_t shr(
  int16_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::int16_t shr(::cuda::std::int16_t __a_reg, ::cuda::std::uint32_t __b_reg)
{
  ::cuda::std::int16_t __dest;
  asm("shr.s16 %0, %1, %2;" : "=h"(__dest) : "h"(__a_reg), "r"(__b_reg) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.s32 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename S32, enable_if_t<sizeof(S32) == 4 && is_integral_v<S32> && is_signed_v<S32>, bool> = true>
__device__ static inline S32 shr(
  S32 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
template <typename _S32,
          ::cuda::std::enable_if_t<sizeof(_S32) == 4 && ::cuda::std::is_integral_v<_S32>&& ::cuda::std::is_signed_v<_S32>,
                                   bool> = true>
_CCCL_DEVICE static inline _S32 shr(_S32 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  ::cuda::std::int32_t __dest;
  asm("shr.s32 %0, %1, %2;"
      : "=r"(__dest)
      : "r"(*reinterpret_cast<const ::cuda::std::int32_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_S32*>(&__dest);
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.s64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename S64, enable_if_t<sizeof(S64) == 8 && is_integral_v<S64> && is_signed_v<S64>, bool> = true>
__device__ static inline S64 shr(
  S64 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
template <typename _S64,
          ::cuda::std::enable_if_t<sizeof(_S64) == 8 && ::cuda::std::is_integral_v<_S64>&& ::cuda::std::is_signed_v<_S64>,
                                   bool> = true>
_CCCL_DEVICE static inline _S64 shr(_S64 __a_reg, ::cuda::std::uint32_t __b_reg)
{
  ::cuda::std::int64_t __dest;
  asm("shr.s64 %0, %1, %2;"
      : "=l"(__dest)
      : "l"(*reinterpret_cast<const ::cuda::std::int64_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_S64*>(&__dest);
}
#endif // __cccl_ptx_isa >= 100

#endif // _CUDA_PTX_GENERATED_SHR_H_
