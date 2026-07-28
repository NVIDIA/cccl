// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_BFIND_H_
#define _CUDA_PTX_GENERATED_BFIND_H_

/*
// bfind.u32 dest, a_reg; // PTX ISA 20, SM_50
template <typename U32, enable_if_t<sizeof(U32) == 4 && is_integral_v<U32> && is_unsigned_v<U32>, bool> = true>
__device__ static inline uint32_t bfind(
  U32 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <
  typename _U32,
  ::cuda::std::enable_if_t<sizeof(_U32) == 4 && ::cuda::std::is_integral_v<_U32>&& ::cuda::std::is_unsigned_v<_U32>,
                           bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind(_U32 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.u32 %0, %1;" : "=r"(__dest) : "r"(*reinterpret_cast<const ::cuda::std::uint32_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.u32 dest, a_reg; // PTX ISA 20, SM_50
template <typename U32, enable_if_t<sizeof(U32) == 4 && is_integral_v<U32> && is_unsigned_v<U32>, bool> = true>
__device__ static inline uint32_t bfind_shiftamt(
  U32 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <
  typename _U32,
  ::cuda::std::enable_if_t<sizeof(_U32) == 4 && ::cuda::std::is_integral_v<_U32>&& ::cuda::std::is_unsigned_v<_U32>,
                           bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind_shiftamt(_U32 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.shiftamt.u32 %0, %1;" : "=r"(__dest) : "r"(*reinterpret_cast<const ::cuda::std::uint32_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.u64 dest, a_reg; // PTX ISA 20, SM_50
template <typename U64, enable_if_t<sizeof(U64) == 8 && is_integral_v<U64> && is_unsigned_v<U64>, bool> = true>
__device__ static inline uint32_t bfind(
  U64 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <
  typename _U64,
  ::cuda::std::enable_if_t<sizeof(_U64) == 8 && ::cuda::std::is_integral_v<_U64>&& ::cuda::std::is_unsigned_v<_U64>,
                           bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind(_U64 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.u64 %0, %1;" : "=r"(__dest) : "l"(*reinterpret_cast<const ::cuda::std::uint64_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.u64 dest, a_reg; // PTX ISA 20, SM_50
template <typename U64, enable_if_t<sizeof(U64) == 8 && is_integral_v<U64> && is_unsigned_v<U64>, bool> = true>
__device__ static inline uint32_t bfind_shiftamt(
  U64 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <
  typename _U64,
  ::cuda::std::enable_if_t<sizeof(_U64) == 8 && ::cuda::std::is_integral_v<_U64>&& ::cuda::std::is_unsigned_v<_U64>,
                           bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind_shiftamt(_U64 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.shiftamt.u64 %0, %1;" : "=r"(__dest) : "l"(*reinterpret_cast<const ::cuda::std::uint64_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.s32 dest, a_reg; // PTX ISA 20, SM_50
template <typename S32, enable_if_t<sizeof(S32) == 4 && is_integral_v<S32> && is_signed_v<S32>, bool> = true>
__device__ static inline uint32_t bfind(
  S32 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <typename _S32,
          ::cuda::std::enable_if_t<sizeof(_S32) == 4 && ::cuda::std::is_integral_v<_S32>&& ::cuda::std::is_signed_v<_S32>,
                                   bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind(_S32 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.s32 %0, %1;" : "=r"(__dest) : "r"(*reinterpret_cast<const ::cuda::std::int32_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.s32 dest, a_reg; // PTX ISA 20, SM_50
template <typename S32, enable_if_t<sizeof(S32) == 4 && is_integral_v<S32> && is_signed_v<S32>, bool> = true>
__device__ static inline uint32_t bfind_shiftamt(
  S32 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <typename _S32,
          ::cuda::std::enable_if_t<sizeof(_S32) == 4 && ::cuda::std::is_integral_v<_S32>&& ::cuda::std::is_signed_v<_S32>,
                                   bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind_shiftamt(_S32 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.shiftamt.s32 %0, %1;" : "=r"(__dest) : "r"(*reinterpret_cast<const ::cuda::std::int32_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.s64 dest, a_reg; // PTX ISA 20, SM_50
template <typename S64, enable_if_t<sizeof(S64) == 8 && is_integral_v<S64> && is_signed_v<S64>, bool> = true>
__device__ static inline uint32_t bfind(
  S64 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <typename _S64,
          ::cuda::std::enable_if_t<sizeof(_S64) == 8 && ::cuda::std::is_integral_v<_S64>&& ::cuda::std::is_signed_v<_S64>,
                                   bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind(_S64 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.s64 %0, %1;" : "=r"(__dest) : "l"(*reinterpret_cast<const ::cuda::std::int64_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.s64 dest, a_reg; // PTX ISA 20, SM_50
template <typename S64, enable_if_t<sizeof(S64) == 8 && is_integral_v<S64> && is_signed_v<S64>, bool> = true>
__device__ static inline uint32_t bfind_shiftamt(
  S64 a_reg);
*/
#if __cccl_ptx_isa >= 200
template <typename _S64,
          ::cuda::std::enable_if_t<sizeof(_S64) == 8 && ::cuda::std::is_integral_v<_S64>&& ::cuda::std::is_signed_v<_S64>,
                                   bool> = true>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bfind_shiftamt(_S64 __a_reg)
{
  ::cuda::std::uint32_t __dest;
  asm("bfind.shiftamt.s64 %0, %1;" : "=r"(__dest) : "l"(*reinterpret_cast<const ::cuda::std::int64_t*>(&__a_reg)) :);
  return __dest;
}
#endif // __cccl_ptx_isa >= 200

#endif // _CUDA_PTX_GENERATED_BFIND_H_
