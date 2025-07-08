// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/unsafe_bitcast.cuh>
#include <cub/thread/thread_operators.cuh> // is_cuda_minimum_maximum_v
#include <cub/util_arch.cuh> // CUB_PTX_ARCH
#include <cub/warp/specializations/warp_reduce_config.cuh>
#include <cub/warp/warp_utils.cuh> // logical_warp_id

#include <cuda/bit> // cuda::bitmask
#include <cuda/cmath> // is_power_of_two
#include <cuda/functional>
#include <cuda/std/cstdint>

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif // _CCCL_HAS_NVBF16()

CUB_NAMESPACE_BEGIN
namespace detail
{

// predicate version should always be present to efficiently handle segmented reduction and dynamic valid items.
// non-predicate version should be used only in the other cases and where predicate generates redundant instructions.

//----------------------------------------------------------------------------------------------------------------------
// 16-bit shfl_down_op

#define _CUB_SHFL_DOWN_OP_16BIT(OPERATOR, TYPE, PTX_OP)                                                                \
                                                                                                                       \
  template <typename = void>                                                                                           \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op_pred(                                                 \
    OPERATOR, TYPE value, uint32_t source_offset, uint32_t shfl_c, uint32_t mask)                                      \
  {                                                                                                                    \
    auto tmp = cub::detail::unsafe_bitcast<uint16_t>(value);                                                           \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                           \n\t\t" \
      ".reg .b16  dummy;                                                                                       \n\t\t" \
      ".reg .b16  h1;                                                                                          \n\t\t" \
      ".reg .b32  r0;                                                                                          \n\t\t" \
      "mov.b32 r0, {%0, dummy};                                                                                \n\t\t" \
      "shfl.sync.down.b32 r0|p, r0, %1, %2, %3;                                                                \n\t\t" \
      "mov.b32 {h1, dummy}, r0;                                                                                \n\t\t" \
      "@p " #PTX_OP " %0, h1, %0;                                                                              \n\t\t" \
      "}"                                                                                                              \
      : "+h"(tmp)                                                                                                      \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                     \
  }

//----------------------------------------------------------------------------------------------------------------------
// 32-bit shfl_down_op

#define _CUB_SHFL_DOWN_OP_32BIT(OPERATOR, TYPE, PTX_OP)                                                                \
                                                                                                                       \
  template <typename = void>                                                                                           \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op_pred(                                                 \
    OPERATOR, TYPE value, uint32_t source_offset, uint32_t shfl_c, uint32_t mask)                                      \
  {                                                                                                                    \
    auto tmp = cub::detail::unsafe_bitcast<uint32_t>(value);                                                           \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                           \n\t\t" \
      ".reg .b32  r0;                                                                                          \n\t\t" \
      "shfl.sync.down.b32 r0|p, %0, %1, %2, %3;                                                                \n\t\t" \
      "@p " #PTX_OP " %0, r0, %0;                                                                              \n\t\t" \
      "}"                                                                                                              \
      : "+r"(tmp)                                                                                                      \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                     \
  }

//----------------------------------------------------------------------------------------------------------------------
// 64-bit shfl_down_op

#define _CUB_SHFL_DOWN_OP_64BIT(OPERATOR, TYPE, PTX_OP)                                                                \
                                                                                                                       \
  template <typename = void>                                                                                           \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op_pred(                                                 \
    OPERATOR, TYPE value, uint32_t source_offset, uint32_t shfl_c, uint32_t mask)                                      \
  {                                                                                                                    \
    auto tmp = cub::detail::unsafe_bitcast<uint64_t>(value);                                                           \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                           \n\t\t" \
      ".reg .u32 lo;                                                                                           \n\t\t" \
      ".reg .u32 hi;                                                                                           \n\t\t" \
      ".reg .b64 r1;                                                                                           \n\t\t" \
      "mov.b64 {lo, hi}, %0;                                                                                   \n\t\t" \
      "shfl.sync.down.b32 lo,   lo, %1, %2, %3;                                                                \n\t\t" \
      "shfl.sync.down.b32 hi|p, hi, %1, %2, %3;                                                                \n\t\t" \
      "mov.b64 r1, {lo, hi};                                                                                   \n\t\t" \
      "@p " #PTX_OP " %0, r1, %0;                                                                              \n\t\t" \
      "}"                                                                                                              \
      : "+l"(tmp)                                                                                                      \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                     \
  }

//----------------------------------------------------------------------------------------------------------------------
// cuda::std::plus Instantiations

_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, uint32_t, add.u32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, int, add.s32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, float, add.f32)
_CUB_SHFL_DOWN_OP_64BIT(_CUDA_VSTD::plus<>, double, add.f64)

#if __cccl_ptx_isa >= 860 && (__CUDA_ARCH_HAS_FEATURE__(SM100_ALL) || CUB_PTX_ARCH >= 1000)
_CUB_SHFL_DOWN_OP_64BIT(_CUDA_VSTD::plus<>, ::float2, add.f32x2)
#endif // __cccl_ptx_isa >= 860 && (__CUDA_ARCH_HAS_FEATURE__(SM100_ALL) || CUB_PTX_ARCH >= 1000)

#if __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, ::short2, add.s16x2)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, ::ushort2, add.u16x2)
#endif // __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900

#if _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 530
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<>, ::__half, add.f16)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, ::__half2, add.f16x2)
#endif // _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 530

#if _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<>, ::__nv_bfloat16, add.bf16)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, ::__nv_bfloat162, add.bf16x2)
#endif // _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900

//----------------------------------------------------------------------------------------------------------------------
// cuda::maximum/minimum Instantiations

_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, int, max.s32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, uint32_t, max.u32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, float, max.f32)

_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, int, min.s32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, uint32_t, min.u32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, float, min.f32)

_CUB_SHFL_DOWN_OP_64BIT(::cuda::maximum<>, double, max.f64)
_CUB_SHFL_DOWN_OP_64BIT(::cuda::minimum<>, double, min.f64)

#if __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, ::short2, max.s16x2)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, ::ushort2, max.u16x2)

_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, ::short2, min.s16x2)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, ::ushort2, min.u16x2)
#endif // __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900

#if _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 800
_CUB_SHFL_DOWN_OP_16BIT(::cuda::minimum<>, ::__half, min.f16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, ::__half2, min.f16x2)

_CUB_SHFL_DOWN_OP_16BIT(::cuda::maximum<>, ::__half, max.f16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, ::__half2, max.f16x2)
#endif // _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 800

#if _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 800
_CUB_SHFL_DOWN_OP_16BIT(::cuda::minimum<>, ::__nv_bfloat16, min.bf16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, ::__nv_bfloat162, min.bf16x2)

_CUB_SHFL_DOWN_OP_16BIT(::cuda::maximum<>, ::__nv_bfloat16, max.bf16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, ::__nv_bfloat162, max.bf16x2)
#endif // _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 800

//----------------------------------------------------------------------------------------------------------------------
// cuda::std::bit_and/bit_or/bit_xor Instantiations

_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_and<>, uint32_t, and.b32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_or<>, uint32_t, or.b32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_xor<>, uint32_t, xor.b32)

#undef _CUB_SHFL_DOWN_OP_16BIT
#undef _CUB_SHFL_DOWN_OP_32BIT
#undef _CUB_SHFL_DOWN_OP_64BIT

//----------------------------------------------------------------------------------------------------------------------
// Generation of Shuffle / Ballot Mask

template <int LogicalWarpSize, size_t ValidItems, bool IsSegmented>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t reduce_shuffle_bound_mask(
  [[maybe_unused]] logical_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items,
  is_segmented_t<IsSegmented> is_segmented)
{
  if constexpr (is_segmented)
  {
    return valid_items.extent(0); // segmented limit
  }
  else // valid_items is dynamic
  {
    return cub::detail::logical_warp_base_id(logical_size) + valid_items.extent(0) - 1;
  }
}

//----------------------------------------------------------------------------------------------------------------------
// Generation of Shuffle/Reduce Member Mask

template <ReduceLogicalMode LogicalMode, int LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE constexpr uint32_t
reduce_lane_mask([[maybe_unused]] reduce_logical_mode_t<LogicalMode> logical_mode, logical_warp_size_t<LogicalWarpSize>)
{
  return (logical_mode == multiple_reductions) ? 0xFFFFFFFF : (0xFFFFFFFF >> (warp_threads - LogicalWarpSize));
}

} // namespace detail
CUB_NAMESPACE_END
