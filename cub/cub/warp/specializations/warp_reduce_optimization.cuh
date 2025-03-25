/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
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

/**
 * @file
 * cub::WarpReduceSmem provides smem-based variants of parallel reduction of items partitioned
 * across a CUDA thread warp.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_type.cuh>

#include <cuda/functional>
#include <cuda/ptx>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

#include "cuda/std/__concepts/concept_macros.h"

CUB_NAMESPACE_BEGIN
namespace detail
{

enum class WarpReduceMode
{
  SingleLogicalWarp,
  MultipleLogicalWarps
};

template <WarpReduceMode Mode>
using warp_reduce_mode_t = _CUDA_VSTD::integral_constant<WarpReduceMode, Mode>;

inline constexpr auto single_logical_warp    = warp_reduce_mode_t<WarpReduceMode::SingleLogicalWarp>{};
inline constexpr auto multiple_logical_warps = warp_reduce_mode_t<WarpReduceMode::MultipleLogicalWarps>{};

//----------------------------------------------------------------------------------------------------------------------

enum class WarpReduceResult
{
  AllLanes,
  SingleLane
};

template <WarpReduceResult Kind>
using warp_reduce_result_t = _CUDA_VSTD::integral_constant<WarpReduceResult, Kind>;

inline constexpr auto all_lanes_result   = warp_reduce_result_t<WarpReduceResult::AllLanes>{};
inline constexpr auto single_lane_result = warp_reduce_result_t<WarpReduceResult::SingleLane>{};

//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Reduction
 *
 * @tparam ALL_LANES_VALID
 *   Whether all lanes in each warp are contributing a valid fold of items
 *
 * @param[in] input
 *   Calling thread's input
 *
 * @param[in] valid_items
 *   Total number of valid items across the logical warp
 *
 * @param[in] reduction_op
 *   Reduction operator
 */

_CCCL_DEVICE _CCCL_FORCEINLINE static unsigned reduce_down_mask(int step)
{
  const auto clamp   = unsigned{LogicalWarpSize} - 1;
  const auto segmask = (unsigned{warp_threads} - LogicalWarpSize) << 8;
  return clamp | segmask;
}

template <typename Input, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE static Input reduce_sm30(Input input, ReductionOp)
{
  using namespace internal;
  unsigned member_mask = ::__activemask();
  if constexpr (_CUDA_VSTD::is_same_v<Input, bool>)
  {
    if constexpr (is_cuda_std_plus_v<ReductionOp, Input>)
    {
      return _CUDA_VSTD::popcount(::__ballot_sync(member_mask, input));
    }
    else if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
    {
      return ::__all_sync(member_mask, input);
    }
    else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
    {
      return ::__any_sync(member_mask, input);
    }
    else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
    {
      return _CUDA_VSTD::popcount(::__ballot_sync(member_mask, input)) % 2u;
    }
  }
}

template <typename Input, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE static Input reduce_sm80(Input input, ReductionOp)
{
  using namespace internal;
  unsigned member_mask = ::__activemask();
  static_assert(_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) <= sizeof(uint32_t));
  static_assert(is_cuda_std_bitwise_v<ReductionOp, Input> || is_cuda_std_plus_v<ReductionOp, Input>
                || is_cuda_std_min_max_v<ReductionOp, Input>);
  if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_and_sync(member_mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_or_sync(member_mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_xor_sync(member_mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input>)
  {
    return ::__reduce_add_sync(member_mask, input);
  }
  else if constexpr (is_cuda_minimum_v<ReductionOp, Input>)
  {
    return ::__reduce_min_sync(member_mask, input);
  }
  else if constexpr (is_cuda_maximum_v<ReductionOp, Input>)
  {
    return ::__reduce_max_sync(member_mask, input);
  }
}

template <typename Input>
_CCCL_DEVICE _CCCL_FORCEINLINE static auto split(Input input)
{
  constexpr auto half_bits = _CUDA_VSTD::numeric_limits<Input>::digits / 2u;
  using unsigned_t         = _CUDA_VSTD::make_unsigned_t<Input>;
  using half_size_t        = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
  using output_t           = _CUDA_VSTD::__make_nbit_int_t<half_bits, _CUDA_VSTD::is_signed_v<Input>>;
  auto input1              = static_cast<unsigned_t>(input);
  auto high                = static_cast<half_size_t>(input1 >> half_bits);
  auto low                 = static_cast<half_size_t>(input1);
  return _CUDA_VSTD::array<output_t, 2>{static_cast<output_t>(high), static_cast<output_t>(low)};
}

template <typename Input>
_CCCL_DEVICE _CCCL_FORCEINLINE static auto merge(Input inputA, Input inputB)
{
  static_assert(_CUDA_VSTD::is_integral_v<Input>);
  constexpr auto digits = _CUDA_VSTD::numeric_limits<Input>::digits;
  using unsigned_t      = _CUDA_VSTD::__make_nbit_uint_t<digits * 2>;
  using output_t        = _CUDA_VSTD::__make_nbit_int_t<digits * 2, _CUDA_VSTD::is_signed_v<Input>>;
  return static_cast<output_t>(static_cast<unsigned_t>(inputA) << digits | inputB);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp)
_CCCL_REQUIRES(cub::internal::is_cuda_std_bitwise_v<ReductionOp, Input>)
_CCCL_DEVICE _CCCL_FORCEINLINE static Input reduce_recursive(Input input, ReductionOp reduction_op)
{
  using namespace internal;
  static_assert(is_cuda_std_bitwise_v<ReductionOp, Input> && _CUDA_VSTD::is_integral_v<Input>);
  auto [high, low]    = split(input);
  auto high_reduction = reduce(high, reduction_op);
  auto low_reduction  = reduce(low, reduction_op);
  return merge(high_reduction, low_reduction);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp)
_CCCL_REQUIRES(cub::internal::is_cuda_std_min_max_v<ReductionOp, Input>)
_CCCL_DEVICE _CCCL_FORCEINLINE static Input reduce_recursive(Input input, ReductionOp reduction_op)
{
  using detail::merge;
  using detail::split;
  using internal::identity_v;
  constexpr auto half_bits = _CUDA_VSTD::numeric_limits<Input>::digits / 2u;
  auto [high, low]         = split(input);
  auto high_result         = reduce(high, reduction_op);
  if (high_result == 0) // shortcut: input is in range [0, 2^N/2)
  {
    return reduce(low, reduction_op);
  }
  if (_CUDA_VSTD::is_unsigned_v<Input> || high_result > 0) // >= 2^N/2 -> perform the computation as unsigned
  {
    using half_size_unsigned_t = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
    constexpr auto identity    = identity_v<ReductionOp, half_size_unsigned_t>;
    auto low_unsigned          = static_cast<half_size_unsigned_t>(low);
    auto low_result            = reduce(high_result == high ? low_unsigned : identity, reduction_op);
    return static_cast<Input>(merge(high_result, low_result));
  }
  // signed type and < 0
  using half_size_signed_t = _CUDA_VSTD::__make_nbit_int_t<half_bits, true>;
  constexpr auto identity  = identity_v<ReductionOp, half_size_signed_t>;
  auto low_result          = reduce(high_result == high ? low : identity, reduction_op);
  return merge(high_result, low_result);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp)
_CCCL_REQUIRES(cub::internal::is_cuda_std_plus_v<ReductionOp, Input>)
_CCCL_DEVICE _CCCL_FORCEINLINE Input reduce_recursive(Input input, ReductionOp reduction_op)
{
  using detail::merge;
  using detail::split;
  using internal::identity_v;
  using unsigned_t      = _CUDA_VSTD::make_unsigned_t<Input>;
  constexpr auto digits = _CUDA_VSTD::numeric_limits<Input>::digits;
  auto [high, low]      = split(input);
  auto high_reduction   = reduce(high, reduction_op);
  auto low_digits       = static_cast<unsigned_t>(low) >> (digits - 5);
  auto carry_out        = reduce(static_cast<uint32_t>(low_digits), reduction_op);
  auto low_reduction    = reduce(low, reduction_op);
  auto result_high      = high_reduction + carry_out;
  return merge(result_high, low_reduction);
}

template <typename T>
inline constexpr bool is_complex_v = _CUDA_VSTD::__is_complex<T>::value;

template <typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult ResultMode>
_CCCL_DEVICE _CCCL_FORCEINLINE Input Reduce(
  Input input,
  ReductionOp reduction_op,
  warp_reduce_mode_t<Mode> warp_mode,
  warp_reduce_result_t<ResultMode> result_mode)
{
  using namespace internal;
  constexpr bool is_natively_supported_type =
    _CUDA_VSTD::is_integral_v<Input> //
    || _CUDA_VSTD::is_floating_point_v<Input> //
    || _CUDA_VSTD::is_same_v<Input, _CUDA_VSTD::complex<float>>
    || _CUDA_VSTD::is_same_v<Input, _CUDA_VSTD::complex<double>>;
  constexpr bool is_small_integer = _CUDA_VSTD::is_integral_v<Input> && sizeof(Input) <= sizeof(uint32_t);
  if constexpr (is_natively_supported_type)
  {
    if constexpr (is_cuda_std_min_max_v<ReductionOp, Input> && _CUDA_VSTD::is_floating_point_v<Input>)
    {
      constexpr auto digits = _CUDA_VSTD::numeric_limits<Input>::digits;
      using signed_t        = _CUDA_VSTD::__make_nbit_int_t<digits, true>;
      auto result           = reduce(_CUDA_VSTD::bit_cast<signed_t>(input), reduction_op, warp_mode, result_mode);
      return _CUDA_VSTD::bit_cast<Input>(result);
    }
    else if constexpr (is_complex_v<Input> && !is_cuda_std_min_max_v<ReductionOp, Input>)
    {
      auto real = reduce(input.real(), reduction_op, warp_mode, result_mode);
      auto img  = reduce(input.img(), reduction_op, warp_mode, result_mode);
      return Input{real, img};
    }
    else if constexpr (warp_mode == single_logical_warp && is_small_integer)
    {
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80, //
                        (return reduce_sm80(input, reduction_op);),
                        (return reduce_sm30(input, reduction_op, warp_mode);));
    }
    else if constexpr (is_small_integer || _CUDA_VSTD::is_floating_point_v<Input>)
    {
      return reduce_sm30(input, reduction_op, warp_mode);
    }
    else if constexpr (_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) > sizeof(uint32_t))
    {
      return reduce_recursive(input, reduction_op, warp_mode, result_mode);
    }
    else
    {
      static_assert(_CUDA_VSTD::__always_false_v<Input>, "Invalid input type/reduction operator combination");
      _CCCL_UNREACHABLE();
    }
  }
  else // generic implementation
  {
    constexpr auto Log2Size     = ::cuda::ilog2(LogicalWarpSize);
    constexpr auto LogicalWidth = _CUDA_VSTD::integral_constant<int, LogicalWarpSize>{};
    int pred;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int K = 0; K < Log2Size; K++)
    {
      auto res = _CUDA_VDEV::warp_shuffle_down(input, 1u << K, LogicalWidth);
      if (res.pred)
      {
        input = reduction_op(input, res.data);
      }
    }
    if constexpr (result_mode == all_lanes_result)
    {
      return _CUDA_VDEV::warp_shuffle_idx(input, 0, LogicalWidth);
    }
  }
}

} // namespace detail

// #####################################################################################################################
// # SHUFFLE DOWN + OP (float, int)
// #####################################################################################################################

#define _CUB_SHFL_OP_32BIT(TYPE, DIRECTION, OP, PTX_TYPE, PTX_REG_TYPE)                                                \
                                                                                                                       \
  template <typename = void>                                                                                           \
  _CCCL_DEVICE void shfl_##DIRECTION##_##OP(                                                                           \
    TYPE& value, int& pred, unsigned source_offset, unsigned shfl_c, unsigned mask)                                    \
  {                                                                                                                    \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t"  \
      ".reg .b32  r0;                                                                                         \n\t\t"  \
      "shfl.sync." #DIRECTION ".b32 r0|p, %0, %2, %3, %4;                                                     \n\t\t"  \
      "@p " #OP "." #PTX_TYPE " %0, r0, %0;                                                                   \n\t\t"  \
      "selp.s32 %1, 1, 0, p;                                                                                    \n\t"  \
      "}"                                                                                                              \
      : "+" #PTX_REG_TYPE(value), "=r"(pred)                                                                           \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
  }

_CUB_SHFL_OP_32BIT(float, down, add, f32, f) // shfl_down_add(float)
_CUB_SHFL_OP_32BIT(int, down, add, s32, r) // shfl_down_add(int)
_CUB_SHFL_OP_32BIT(int, down, max, s32, r) // shfl_down_max(int)
_CUB_SHFL_OP_32BIT(int, down, min, s32, r) // shfl_down_min(int)
_CUB_SHFL_OP_32BIT(unsigned, down, add, u32, r) // shfl_down_add(unsigned)
_CUB_SHFL_OP_32BIT(unsigned, down, max, u32, r) // shfl_down_max(unsigned)
_CUB_SHFL_OP_32BIT(unsigned, down, min, u32, r) // shfl_down_min(unsigned)
_CUB_SHFL_OP_32BIT(unsigned, down, and, u32, r) // shfl_down_and(unsigned)
_CUB_SHFL_OP_32BIT(unsigned, down, or, u32, r) // shfl_down_or(unsigned)
_CUB_SHFL_OP_32BIT(unsigned, down, xor, u32, r) // shfl_down_xor(unsigned)
#undef _CUB_SHFL_OP_32BIT

// #####################################################################################################################
// # SHUFFLE DOWN + OP (double)
// #####################################################################################################################

#define _CUB_SHFL_OP_64BIT(TYPE, DIRECTION, OP, PTX_TYPE, PTX_REG_TYPE)                                                \
                                                                                                                       \
  template <typename = void>                                                                                           \
  _CCCL_DEVICE void shfl_##DIRECTION##_##OP(                                                                           \
    TYPE& value, int& pred, unsigned source_offset, unsigned shfl_c, unsigned mask)                                    \
  {                                                                                                                    \
    asm volatile(                                                                                                      \
      "{                                                                                                       \n\t\t" \
      ".reg .pred p;                                                                                          \n\t\t"  \
      ".reg .u32 lo;                                                                                          \n\t\t"  \
      ".reg .u32 hi;                                                                                          \n\t\t"  \
      ".reg ." #PTX_TYPE " r1;                                                                                \n\t\t"  \
      "mov.b64 {lo, hi}, %0;                                                                                  \n\t\t"  \
      "shfl.sync." #DIRECTION ".b32 lo,   lo, %2, %3, %4;                                                     \n\t\t"  \
      "shfl.sync." #DIRECTION ".b32 hi|p, hi, %2, %3, %4;                                                     \n\t\t"  \
      "@p mov.b64 r1, {lo, hi};                                                                               \n\t\t"  \
      "@p " #OP "." #PTX_TYPE " %0, r1, %0;                                                                   \n\t\t"  \
      "selp.s32 %1, 1, 0, p;                                                                                    \n\t"  \
      "}"                                                                                                              \
      : "+" #PTX_REG_TYPE(value), "=r"(pred)                                                                           \
      : "r"(source_offset), "r"(shfl_c), "r"(mask));                                                                   \
  }

_CUB_SHFL_OP_64BIT(double, down, add, f64, d) // shfl_down_add (double)
#undef SHFL_EXCH_OP_64

CUB_NAMESPACE_END
