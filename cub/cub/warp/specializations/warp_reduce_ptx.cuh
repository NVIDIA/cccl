/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 *       following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/

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
#include <cuda/cmath> // has_single_bit
#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <vector_types.h>

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif // _CCCL_HAS_NVBF16()

CUB_NAMESPACE_BEGIN
namespace detail
{

//----------------------------------------------------------------------------------------------------------------------
// 16-bit shfl_down_op

#define _CUB_SHFL_DOWN_OP_16BIT(OPERATOR, TYPE, PTX_OP)                                                               \
                                                                                                                      \
  template <typename = void>                                                                                          \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op(                                                     \
    OPERATOR, TYPE value, uint32_t source_offset, uint32_t shfl_c, uint32_t mask)                                     \
  {                                                                                                                   \
    auto tmp = cub::detail::unsafe_bitcast<uint16_t>(value);                                                          \
    asm volatile(                                                                                                     \
      "{                                                                                                      \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t" \
      ".reg .b16  dummy;                                                                                      \n\t\t" \
      ".reg .b16  h1;                                                                                         \n\t\t" \
      ".reg .b32  r0;                                                                                         \n\t\t" \
      "mov.b32 r0, {%0, dummy};                                                                               \n\t\t" \
      "shfl.sync.down.b32 r0|p, r0, %1, %2, %3;                                                               \n\t\t" \
      "mov.b32 {h1, dummy}, r0;                                                                               \n\t\t" \
      "" #PTX_OP " %0, h1, %0;                                                                                \n\t\t" \
      "}"                                                                                                             \
      : "+h"(tmp)                                                                                                     \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                  \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                    \
  }

//----------------------------------------------------------------------------------------------------------------------
// 32-bit shfl_down_op

#define _CUB_SHFL_DOWN_OP_32BIT(OPERATOR, TYPE, PTX_OP)                                                               \
                                                                                                                      \
  template <typename = void>                                                                                          \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op(                                                     \
    OPERATOR, TYPE value, uint32_t source_offset, uint32_t shfl_c, uint32_t mask)                                     \
  {                                                                                                                   \
    auto tmp = cub::detail::unsafe_bitcast<uint32_t>(value);                                                          \
    asm volatile(                                                                                                     \
      "{                                                                                                      \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t" \
      ".reg .b32  r0;                                                                                         \n\t\t" \
      "shfl.sync.down.b32 r0|p, %0, %1, %2, %3;                                                               \n\t\t" \
      "" #PTX_OP " %0, r0, %0;                                                                                \n\t\t" \
      "}"                                                                                                             \
      : "+r"(tmp)                                                                                                     \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                  \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                    \
  }

//----------------------------------------------------------------------------------------------------------------------
// 64-bit shfl_down_op

#define _CUB_SHFL_DOWN_OP_64BIT(OPERATOR, TYPE, PTX_OP)                                                               \
                                                                                                                      \
  template <typename = void>                                                                                          \
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE TYPE shfl_down_op(                                                     \
    OPERATOR, TYPE value, uint32_t source_offset, uint32_t shfl_c, uint32_t mask)                                     \
  {                                                                                                                   \
    auto tmp = cub::detail::unsafe_bitcast<uint64_t>(value);                                                          \
    asm volatile(                                                                                                     \
      "{                                                                                                      \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t" \
      ".reg .u32 lo;                                                                                          \n\t\t" \
      ".reg .u32 hi;                                                                                          \n\t\t" \
      ".reg .b64 r1;                                                                                          \n\t\t" \
      "mov.b64 {lo, hi}, %0;                                                                                  \n\t\t" \
      "shfl.sync.down.b32 lo,   lo, %1, %2, %3;                                                               \n\t\t" \
      "shfl.sync.down.b32 hi|p, hi, %1, %2, %3;                                                               \n\t\t" \
      "mov.b64 r1, {lo, hi};                                                                                  \n\t\t" \
      "" #PTX_OP " %0, r1, %0;                                                                                \n\t\t" \
      "}"                                                                                                             \
      : "+l"(tmp)                                                                                                     \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                  \
    return cub::detail::unsafe_bitcast<TYPE>(tmp);                                                                    \
  }

//----------------------------------------------------------------------------------------------------------------------
// cuda::std::plus Instantiations

_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, uint32_t, @p add.u32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, int, @p add.s32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, float, @p add.f32)
_CUB_SHFL_DOWN_OP_64BIT(_CUDA_VSTD::plus<>, double, add.f64)

#if _CCCL_HAS_NVFP16() && __cccl_ptx_isa >= 860 && (__CUDA_ARCH_HAS_FEATURE__(SM100_ALL) || CUB_PTX_ARCH >= 1000)
_CUB_SHFL_DOWN_OP_64BIT(_CUDA_VSTD::plus<>, float2, add.f32x2)
#endif // _CCCL_HAS_NVFP16() && __cccl_ptx_isa >= 860 && (__CUDA_ARCH_HAS_FEATURE__(SM100_ALL) || CUB_PTX_ARCH >= 1000)

#if __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, short2, add.s16x2)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, ushort2, add.u16x2)
#endif // __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900

#if _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 530
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<>, __half, add.f16)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, __half2, add.f16x2)
#endif // _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 530

#if _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900
_CUB_SHFL_DOWN_OP_16BIT(_CUDA_VSTD::plus<>, __nv_bfloat16, add.bf16)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::plus<>, __nv_bfloat162, add.bf16x2)
#endif // _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900

//----------------------------------------------------------------------------------------------------------------------
// cuda::maximum/minimum Instantiations

_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, int, @p max.s32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, uint32_t, @p max.u32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, float, @p max.f32)
_CUB_SHFL_DOWN_OP_64BIT(::cuda::maximum<>, double, max.f64)

_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, int, @p min.s32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, uint32_t, @p min.u32)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, float, @p min.f32)
_CUB_SHFL_DOWN_OP_64BIT(::cuda::minimum<>, double, min.f64)

#if __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, short2, max.s16x2)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, ushort2, max.u16x2)

_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, short2, min.s16x2)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, ushort2, min.u16x2)
#endif // __cccl_ptx_isa >= 800 && CUB_PTX_ARCH >= 900

#if _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 800
_CUB_SHFL_DOWN_OP_16BIT(::cuda::minimum<>, __half, min.f16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, __half2, min.f16x2)

_CUB_SHFL_DOWN_OP_16BIT(::cuda::maximum<>, __half, max.f16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, __half2, max.f16x2)
#endif // _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 800

#if _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 800
_CUB_SHFL_DOWN_OP_16BIT(::cuda::minimum<>, __nv_bfloat16, min.bf16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::minimum<>, __nv_bfloat162, @p min.bf16x2)

_CUB_SHFL_DOWN_OP_16BIT(::cuda::maximum<>, __nv_bfloat16, max.bf16)
_CUB_SHFL_DOWN_OP_32BIT(::cuda::maximum<>, __nv_bfloat162, @p max.bf16x2)
#endif // _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 800

//----------------------------------------------------------------------------------------------------------------------
// cuda::std::bit_and/bit_or/bit_xor Instantiations

_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_and<>, uint32_t, @p and.b32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_or<>, uint32_t, @p or.b32)
_CUB_SHFL_DOWN_OP_32BIT(_CUDA_VSTD::bit_xor<>, uint32_t, @p xor.b32)

#undef _CUB_SHFL_DOWN_OP_16BIT
#undef _CUB_SHFL_DOWN_OP_32BIT
#undef _CUB_SHFL_DOWN_OP_64BIT

//----------------------------------------------------------------------------------------------------------------------
// SM100 Min/Max Reduction

extern "C" _CCCL_DEVICE float redux_min_max_sync_is_not_supported_before_sm100a();

#if __cccl_ptx_isa >= 860

#  define _CUB_REDUX_FLOAT_OP(OPERATOR, PTX_OP)                                                               \
                                                                                                              \
    template <typename = void>                                                                                \
    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE float redux_sm100a_ptx(OPERATOR, float value, uint32_t mask) \
    {                                                                                                         \
      float result;                                                                                           \
      asm volatile("{"                                                                                        \
                   "redux.sync." #PTX_OP ".f32 %0, %1, %2;"                                                   \
                   "}"                                                                                        \
                   : "=f"(result)                                                                             \
                   : "f"(value), "r"(mask));                                                                  \
      return result;                                                                                          \
    }

_CUB_REDUX_FLOAT_OP(::cuda::minimum<>, min)
_CUB_REDUX_FLOAT_OP(::cuda::maximum<>, max)

#endif // __cccl_ptx_isa >= 860

//----------------------------------------------------------------------------------------------------------------------
// Generation of Shuffle / Ballot Mask

template <int LogicalWarpSize, size_t ValidItems, bool IsSegmented>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t reduce_shuffle_bound_mask(
  [[maybe_unused]] uint32_t step,
  [[maybe_unused]] logical_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items,
  is_segmented_t<IsSegmented> is_segmented = {})
{
  if constexpr (is_segmented)
  {
    return valid_items.extent(0); // segmented limit
  }
  else if constexpr (valid_items.rank_dynamic() == 0 && ::cuda::is_power_of_two(ValidItems))
  {
    const auto clamp   = 1u << step;
    const auto segmask = 0b11110u << (step + 8);
    return clamp | segmask;
  }
  else // valid_items is dynamic
  {
    return cub::detail::logical_warp_id(logical_size) * LogicalWarpSize + valid_items.extent(0) - 1;
  }
}

//----------------------------------------------------------------------------------------------------------------------
// Generation of Shuffle/Reduce Member Mask

template <ReduceLogicalMode LogicalMode, int LogicalWarpSize, size_t ValidItems, bool IsSegmented = false>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t reduce_lane_mask(
  reduce_logical_mode_t<LogicalMode> logical_mode,
  logical_warp_size_t<LogicalWarpSize>,
  valid_items_t<ValidItems> valid_items,
  is_segmented_t<IsSegmented> is_segmented = {})
{
  if constexpr (!is_segmented && valid_items.rank_dynamic() == 1)
  {
    return __activemask(); // equivalent to (0xFFFFFFFF >> (warp_threads - valid_items))
  }
  else
  {
    return (logical_mode == single_reduction) ? (0xFFFFFFFF >> (warp_threads - LogicalWarpSize)) : 0xFFFFFFFF;
  }
}

template <typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t redux_lane_mask(Config config)
{
  auto [logical_mode, _, logical_size, valid_items, is_segmented, first_pos] = config;
  constexpr bool is_single_reduction                                         = logical_mode == single_reduction;
  [[maybe_unused]] auto shift = is_single_reduction ? 0 : cub::detail::logical_warp_base_id(logical_size);
  if constexpr (is_segmented)
  {
    return ::cuda::bitmask(first_pos, valid_items.extent(0) - first_pos + 1);
  }
  else if constexpr (valid_items.rank_dynamic() == 1)
  {
    auto base_mask = ::cuda::bitmask(0, valid_items.extent(0));
    return base_mask << shift;
  }
  else
  {
    constexpr auto base_mask = ::cuda::bitmask(0, logical_size); // must be constexpr
    return base_mask << shift;
  }
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Redux SM100

template <typename T, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T redux_sm100a(ReductionOp, T value, Config config)
{
  using namespace _CUDA_VSTD;
#if __cccl_ptx_isa >= 860
  static_assert(is_same_v<T, float> && is_cuda_minimum_maximum_v<ReductionOp, T>);
  const auto mask = cub::detail::redux_lane_mask(config);
  NV_IF_TARGET(NV_PROVIDES_SM_100,
               (return cub::detail::redux_sm100a_ptx(ReductionOp1{}, value, mask);),
               (return cub::detail::redux_min_max_sync_is_not_supported_before_sm100a();))
#else
  static_assert(__always_false_v<T>, "redux.sync.min/max.f32  requires PTX ISA >= 860");
#endif // __cccl_ptx_isa >= 860
}

} // namespace detail
CUB_NAMESPACE_END
