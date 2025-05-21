/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
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

#include <cub/detail/integer_utils.cuh>
#include <cub/detail/type_traits.cuh> // is_any_short2_v
#include <cub/detail/unsafe_bitcast.cuh>
#include <cub/thread/thread_operators.cuh> // is_cuda_minimum_maximum_v
#include <cub/warp/specializations/warp_reduce_config.cuh> // WarpReduceConfig
#include <cub/warp/specializations/warp_reduce_ptx_redux.cuh> // shfl_down_op
#include <cub/warp/specializations/warp_reduce_ptx_shuffle.cuh> // shfl_down_op
#include <cub/warp/warp_utils.cuh> // logical_lane_id

#include <cuda/cmath> // ilog2
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/complex> // __is_complex
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN
namespace detail
{

/***********************************************************************************************************************
 * WarpReduce Base Step
 **********************************************************************************************************************/

template <typename T, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T warp_reduce_redux_op(T input, ReductionOp, Config config)
{
  using namespace _CUDA_VSTD;
  static_assert(is_integral_v<T> && sizeof(T) <= sizeof(uint32_t));
  const auto mask = cub::detail::redux_lane_mask(config);
  using cast_t    = _If<is_signed_v<T>, int, uint32_t>;
  if constexpr (is_cuda_std_bit_and_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_and_sync(mask, input));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_or_sync(mask, input));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_xor_sync(mask, input));
  }
  else if constexpr (is_cuda_std_plus_v<ReductionOp, T>)
  {
    return __reduce_add_sync(mask, static_cast<cast_t>(input));
  }
  else if constexpr (is_cuda_minimum_v<ReductionOp, T>)
  {
    return __reduce_min_sync(mask, static_cast<cast_t>(input));
  }
  else if constexpr (is_cuda_maximum_v<ReductionOp, T>)
  {
    return __reduce_max_sync(mask, static_cast<cast_t>(input));
  }
  else
  {
    static_assert(__always_false_v<T>, "invalid input type/reduction operator combination");
  }
  _CCCL_UNREACHABLE();
}

template <bool UsePtx = true, typename T, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T warp_reduce_shuffle(T input, ReductionOp reduction_op, Config config)
{
  auto [logical_mode, result_mode, logical_size, valid_items, is_segmented, _] = config;
  constexpr auto is_power_of_two                     = ::cuda::is_power_of_two(logical_size());
  constexpr auto log2_size                           = ::cuda::ceil_ilog2(logical_size());
  constexpr auto logical_size1                       = is_segmented ? warp_threads : logical_size;
  [[maybe_unused]] constexpr auto logical_size_round = ::cuda::next_power_of_two(logical_size1);
  constexpr bool is_vector_type                      = _CUDA_VSTD::is_same_v<float2, T> && is_any_short2_v<T>;
  const auto mask = cub::detail::reduce_lane_mask(logical_mode, logical_size, valid_items, is_segmented);

  using cast_t = normalize_integer_t<T>; // promote (u)int8, (u)int16, (u)long (windows) to (u)int32
  auto input1  = static_cast<cast_t>(input);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int K = 0; K < log2_size; K++)
  {
    if constexpr (is_power_of_two && valid_items.rank_dynamic() == 0 && !is_vector_type)
    {
      auto result = _CUDA_DEVICE::warp_shuffle_down(input1, 1u << K, mask, logical_size);
      input1      = reduction_op(input1, result.data); // do not use shuffle predicate
    }
    else if constexpr (UsePtx && is_shfl_down_op_pred_v<ReductionOp, T>)
    {
      auto shuffle_mask = cub::detail::reduce_shuffle_bound_mask(K, logical_size, valid_items, is_segmented);
      input1            = cub::detail::shfl_down_op_pred(ReductionOp{}, input1, 1u << K, shuffle_mask, mask);
    }
    else
    {
      auto limit     = valid_items.extent(0) - !is_segmented;
      auto lane_dest = cub::detail::logical_lane_id<logical_size1>() + (1u << K);
      auto dest      = ::min(lane_dest, limit);
      auto result    = _CUDA_DEVICE::warp_shuffle_idx<logical_size_round>(input1, dest, mask);
      if (lane_dest <= limit)
      {
        input1 = reduction_op(input1, result.data);
      }
    }
  }
  if constexpr (result_mode == all_lanes_result)
  {
    input1 = _CUDA_DEVICE::warp_shuffle_idx<logical_size_round>(input1, config.first_pos, mask);
  }
  return input1;
}

/***********************************************************************************************************************
 * WarpReduce Recursive Implementation
 **********************************************************************************************************************/

// forward declaration
template <typename T, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T warp_reduce_dispatch(T, ReductionOp, Config);

_CCCL_TEMPLATE(typename T, typename ReductionOp, typename Config)
_CCCL_REQUIRES(is_cuda_std_bitwise_v<ReductionOp, T>)
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static T
warp_reduce_recursive(T input, ReductionOp reduction_op, Config warp_config)
{
  using cub::detail::merge_integers;
  using cub::detail::split_integers;
  using cub::detail::warp_reduce_dispatch;
  auto [high, low]    = split_integers(input);
  auto high_reduction = warp_reduce_dispatch(high, reduction_op, warp_config);
  auto low_reduction  = warp_reduce_dispatch(low, reduction_op, warp_config);
  return merge_integers(high_reduction, low_reduction);
}

_CCCL_TEMPLATE(typename T, typename ReductionOp, typename Config)
_CCCL_REQUIRES(is_cuda_minimum_maximum_v<ReductionOp, T>)
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static T
warp_reduce_recursive(T input, ReductionOp reduction_op, Config warp_config)
{
  using namespace _CUDA_VSTD;
  using cub::detail::merge_integers;
  using cub::detail::split_integers;
  using cub::detail::warp_reduce_dispatch;
  constexpr auto half_bits   = __num_bits_v<T> / 2;
  using half_size_unsigned_t = __make_nbit_uint_t<half_bits>;
  auto [high, low]           = split_integers(input);
  auto high_result           = warp_reduce_dispatch(high, reduction_op, warp_config);
  if (high_result == 0) // shortcut: input is in range [0, 2^N/2) -> perform the computation as unsigned
  {
    return warp_reduce_dispatch(static_cast<half_size_unsigned_t>(low), reduction_op, warp_config);
  }
  if (is_unsigned_v<T> || high_result > 0) // -> perform the computation as unsigned
  {
    constexpr auto identity = identity_v<ReductionOp, half_size_unsigned_t>;
    auto low_unsigned       = static_cast<half_size_unsigned_t>(low);
    auto low_selected       = high_result == high ? low_unsigned : identity;
    auto low_result         = warp_reduce_dispatch(low_selected, reduction_op, warp_config);
    return static_cast<T>(merge_integers(static_cast<half_size_unsigned_t>(high_result), low_result));
  }
  // signed type and input < 0
  using half_size_signed_t = __make_nbit_int_t<half_bits, true>;
  constexpr auto identity  = identity_v<ReductionOp, half_size_signed_t>;
  auto low_signed          = static_cast<half_size_signed_t>(low);
  auto low_selected        = high_result == high ? low_signed : identity;
  auto low_result          = warp_reduce_dispatch(low_selected, reduction_op, warp_config);
  return merge_integers(static_cast<half_size_signed_t>(high_result), low_result);
}

_CCCL_TEMPLATE(typename T, typename ReductionOp, typename Config)
_CCCL_REQUIRES(is_cuda_std_plus_v<ReductionOp, T>)
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T
warp_reduce_recursive(T input, ReductionOp reduction_op, Config warp_config)
{
  using namespace _CUDA_VSTD;
  using detail::merge_integers;
  using detail::split_integers;
  using detail::warp_reduce_dispatch;
  using unsigned_t         = make_unsigned_t<T>;
  constexpr auto half_bits = __num_bits_v<T> / 2;
  auto [high, low]         = split_integers(static_cast<unsigned_t>(input));
  auto high_reduction      = warp_reduce_dispatch(high, reduction_op, warp_config);
  auto low_reduction       = warp_reduce_dispatch(low, reduction_op, warp_config);
  auto low_top_digits      = low >> 5; // low 27-bit, carry out
  auto carry_out           = warp_reduce_dispatch(low_top_digits, reduction_op, warp_config);
  auto result_high         = high_reduction + (carry_out >> (half_bits - 5));
  return merge_integers(result_high, low_reduction);
}

/***********************************************************************************************************************
 * WarpReduce Dispatch
 **********************************************************************************************************************/

template <typename T, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T warp_reduce_dispatch(T input, ReductionOp reduction_op, Config config)
{
  using cub::detail::comparable_int_to_floating_point;
  using cub::detail::floating_point_to_comparable_int;
  using cub::detail::logical_lane_id;
  using cub::detail::unsafe_bitcast;
  using cub::detail::warp_reduce_dispatch;
  using cub::detail::warp_reduce_recursive;
  using cub::detail::warp_reduce_redux_op;
  using cub::detail::warp_reduce_shuffle;
  using namespace _CUDA_VSTD;
  cub::detail::check_warp_reduce_config(config);
  [[maybe_unused]] constexpr bool is_specialized_operator =
    is_cuda_minimum_maximum_v<ReductionOp, T> || is_cuda_std_plus_v<ReductionOp, T>
    || is_cuda_std_bitwise_v<ReductionOp, T>;
  [[maybe_unused]] constexpr bool is_small_integer = is_integral_v<T> && sizeof(T) <= sizeof(uint32_t);
  constexpr auto logical_warp_size                 = config.logical_size;
  auto valid_items                                 = config.valid_items;
  // generalize_operator() is fundamental to avoid slow fallback with the recursive implementation,
  // matching PTX shuffle_op, and integer promotion calls
  auto reduction_op1 = generalize_operator<T>(reduction_op);
  // early exit for threads outside the range with dynamic number of valid items
  if (!config.is_segmented && valid_items.rank_dynamic() == 1
      && logical_lane_id(logical_warp_size) >= valid_items.extent(0))
  {
    return T{};
  }
  // [Min/Max]:
  if constexpr (is_cuda_minimum_maximum_v<ReductionOp, T>)
  {
    if constexpr (is_same_v<T, float> && __cccl_ptx_isa >= 860)
    {
      NV_IF_TARGET(NV_HAS_FEATURE_SM_100a, (return cub::detail::redux_sm100a(reduction_op1, input, config);))
    }
    else if constexpr (is_any_short2_v<T> && __cccl_ptx_isa >= 800)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90, (return warp_reduce_shuffle(input, reduction_op1, config);))
    }
    else if constexpr (is_any_half_v<T> || is_any_bfloat16_v<T>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80, (return warp_reduce_shuffle(input, reduction_op1, config);))
    }
    if constexpr (::cuda::is_floating_point_v<T>)
    {
      auto input_int  = floating_point_to_comparable_int(reduction_op1, input);
      auto result_int = warp_reduce_dispatch(input_int, reduction_op1, config);
      auto result_rev = comparable_int_to_floating_point(result_int); // reverse
      return unsafe_bitcast<T>(result_rev);
    }
  }
  //--------------------------------------------------------------------------------------------------------------------
  // [Plus]
  else if constexpr (is_cuda_std_plus_v<ReductionOp, T>)
  {
    if constexpr (__is_complex_v<T>) // any complex
    {
#if _CCCL_HAS_NVFP16()
      if constexpr (is_half_v<typename T::value_type>)
      {
        auto half2_value = unsafe_bitcast<__half2>(input);
        auto ret         = warp_reduce_dispatch(half2_value, reduction_op1, config);
        return unsafe_bitcast<T>(ret);
      }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
      if constexpr (is_bfloat16_v<typename T::value_type>)
      {
        auto bfloat2_value = unsafe_bitcast<__nv_bfloat162>(input);
        auto ret           = warp_reduce_dispatch(bfloat2_value, reduction_op1, config);
        return unsafe_bitcast<T>(ret);
      }
#endif // _CCCL_HAS_NVBF16()
      if constexpr (is_same_v<typename T::value_type, float>)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_100,
                     (auto float2_value = unsafe_bitcast<float2>(input);
                      auto ret          = warp_reduce_dispatch(float2_value, reduction_op1, config);
                      return unsafe_bitcast<T>(ret);))
      }
      else
      { // double
        auto real = warp_reduce_dispatch(input.real(), reduction_op1, config);
        auto imag = warp_reduce_dispatch(input.imag(), reduction_op1, config);
        return T{real, imag};
      }
    }
    else if constexpr (is_any_half_v<T>) //  __half, __half2
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53, (return warp_reduce_shuffle(input, reduction_op1, config);));
    } // __nv_bfloat16, __nv_bfloat162, short2, ushort2
    else if constexpr (is_any_bfloat16_v<T> || (is_any_short2_v<T> && __cccl_ptx_isa >= 800))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90, (return warp_reduce_shuffle(input, reduction_op1, config);))
    }
    else if constexpr (is_same_v<T, float2> && __cccl_ptx_isa >= 860)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_100, (return warp_reduce_shuffle(input, reduction_op1, config);))
    }
    else if constexpr (is_floating_point_v<T>) // float, double
    {
      return warp_reduce_shuffle(input, reduction_op1, config);
    }
  }
  //--------------------------------------------------------------------------------------------------------------------
  // [Logical And/Or]
  else if constexpr (is_cuda_std_logical_and_v<ReductionOp, T>)
  {
    return __all_sync(cub::detail::redux_lane_mask(config), input);
  }
  else if constexpr (is_cuda_std_logical_or_v<ReductionOp, T>)
  {
    return __any_sync(cub::detail::redux_lane_mask(config), input);
  }
  // TODO: [Comparison]: equal_to
  // else if constexpr (is_cuda_std_equal_to_v<ReductionOp, T>)
  //--------------------------------------------------------------------------------------------------------------------
  // [Plus/Min/Max/Bitwise]: all integers (int8, uint8, int16, uint16, int32, uint32)
  if constexpr (is_specialized_operator && is_small_integer)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return warp_reduce_redux_op(input, reduction_op1, config);));
  }
  else if constexpr (is_specialized_operator && is_integral_v<T>) // large integers (int64, uint64, int128, uint128)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return warp_reduce_recursive(input, reduction_op1, config);));
  }
  //--------------------------------------------------------------------------------------------------------------------
  // else generic implementation
  return warp_reduce_shuffle<false>(input, reduction_op1, config);
}

template <bool IsHeadSegment,
          typename FlagT,
          typename T,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          int LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T warp_segmented_reduce_dispatch(
  T input,
  FlagT flag,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  logical_warp_size_t<LogicalWarpSize> logical_size)
{
  auto logical_base_warp_id = cub::detail::logical_warp_base_id(logical_size);
  auto member_mask          = cub::detail::reduce_lane_mask(logical_mode, logical_size, valid_items_t<0>{});
  auto warp_flags           = __ballot_sync(member_mask, flag);
  warp_flags >>= IsHeadSegment; // Convert to tail-segmented
  warp_flags |= (1u << (logical_base_warp_id + LogicalWarpSize - 1)); // Mask in the last lane of each logical warp
  auto warp_flags_last  = warp_flags & _CUDA_VPTX::get_sreg_lanemask_ge(); // Clean the bits below the current thread
  auto warp_flags_first = warp_flags & _CUDA_VPTX::get_sreg_lanemask_lt(); // Clean the bits after the current thread
  auto last_lane        = _CUDA_VSTD::countr_zero(warp_flags_last); // Find the next set flag
  auto first_lane       = warp_threads - _CUDA_VSTD::countl_zero(warp_flags_first); // Find the previous flag
  auto valid_items      = valid_items_t<>{last_lane};
  WarpReduceConfig config{
    logical_mode, first_lane_result, logical_size, valid_items, is_segmented_t<true>{}, first_lane};
  return cub::detail::warp_reduce_dispatch(input, reduction_op, config);
}

} // namespace detail
CUB_NAMESPACE_END
