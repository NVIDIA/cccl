/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cub/warp/specializations/warp_reduce_ptx.cuh> // shfl_down_op
#include <cub/warp/warp_utils.cuh> // logical_lane_id

#include <cuda/cmath> // ilog2
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/bit> // has_single_bit
#include <cuda/std/complex> // __is_complex
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN
namespace internal
{

/***********************************************************************************************************************
 * WarpReduce Base Step
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input
warp_reduce_redux_op(Input input, ReductionOp reduction_op, Config config)
{
  using namespace _CUDA_VSTD;
  const auto mask = cub::internal::redux_lane_mask(config);
  if constexpr (is_integral_v<Input>)
  {
    using cast_t = _If<is_signed_v<Input>, int, uint32_t>;
    if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
    {
      return static_cast<Input>(__reduce_and_sync(mask, input));
    }
    else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
    {
      return static_cast<Input>(__reduce_or_sync(mask, input));
    }
    else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
    {
      return static_cast<Input>(__reduce_xor_sync(mask, input));
    }
    else if constexpr (is_cuda_std_plus_v<ReductionOp, Input>)
    {
      return __reduce_add_sync(mask, static_cast<cast_t>(input));
    }
    else if constexpr (is_cuda_minimum_v<ReductionOp, Input>)
    {
      return __reduce_min_sync(mask, static_cast<cast_t>(input));
    }
    else if constexpr (is_cuda_maximum_v<ReductionOp, Input>)
    {
      return __reduce_max_sync(mask, static_cast<cast_t>(input));
    }
  }
  else if constexpr (is_same_v<Input, float> && is_cuda_minimum_maximum_v<ReductionOp, Input>)
  {
    return cub::internal::redux_sm100a(reduction_op, input, mask);
  }
  else
  {
    static_assert(__always_false_v<Input>, "invalid input type/reduction operator combination");
  }
}

template <typename Input, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_shuffle_op(Input input, ReductionOp, Config config)
{
  using namespace _CUDA_VSTD;
  using Reduction1 = generalize_operator_t<ReductionOp, Input>;
  auto [logical_mode, result_mode, logical_size, valid_items, is_segmented, _] = config;
  constexpr auto log2_size                                                     = ::cuda::ilog2(logical_size * 2 - 1);
  const auto mask = cub::internal::reduce_lane_mask(logical_mode, logical_size, valid_items, is_segmented);
  using cast_t    = normalize_integer_t<Input>; // promote (u)int8, (u)int16, (u)long (windows) to (u)int32
  auto input1     = static_cast<cast_t>(input);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int K = 0; K < log2_size; K++)
  {
    auto shuffle_mask = cub::internal::reduce_shuffle_bound_mask(K, logical_size, valid_items, is_segmented);
    input1            = cub::internal::shfl_down_op(Reduction1{}, input1, 1u << K, shuffle_mask, mask);
  }
  if constexpr (result_mode == all_lanes_result)
  {
    constexpr auto logical_size_round = _CUDA_VSTD::bit_ceil(uint32_t{logical_size});
    input1                            = _CUDA_VDEV::warp_shuffle_idx<logical_size_round>(input1, 0, mask);
  }
  return input1;
}

template <typename Input, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input
warp_reduce_generic(Input input, ReductionOp reduction_op, Config config)
{
  auto [logical_mode, result_mode, logical_size, valid_items, is_segmented, _] = config;
  constexpr auto is_power_of_two     = _CUDA_VSTD::has_single_bit(uint32_t{logical_size});
  constexpr auto log2_size           = ::cuda::ilog2(logical_size * 2 - 1);
  constexpr uint32_t logical_size1   = is_segmented ? detail::warp_threads : logical_size;
  constexpr auto logical_size1_round = _CUDA_VSTD::bit_ceil(uint32_t{logical_size1});
  const auto mask = cub::internal::reduce_lane_mask(logical_mode, logical_size, valid_items, is_segmented);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int K = 0; K < log2_size; K++)
  {
    _CUDA_VDEV::warp_shuffle_result<Input> res;
    if constexpr (is_power_of_two && valid_items.rank_dynamic() == 0)
    {
      res = _CUDA_VDEV::warp_shuffle_down(input, 1u << K, mask, logical_size);
    }
    else
    {
      constexpr auto logical_size1_round = _CUDA_VSTD::bit_ceil(uint32_t{logical_size1});
      auto limit                         = valid_items.extent(0) - !is_segmented;
      auto lane_dest                     = cub::internal::logical_lane_id<logical_size1>() + (1u << K);
      auto dest                          = ::min(lane_dest, limit);
      res                                = _CUDA_VDEV::warp_shuffle_idx<logical_size1_round>(input, dest, mask);
      res.pred                           = lane_dest <= limit;
    }
    if (res.pred)
    {
      input = reduction_op(input, res.data);
    }
  }
  if constexpr (result_mode == all_lanes_result)
  {
    input = _CUDA_VDEV::warp_shuffle_idx<logical_size1_round>(input, 0, mask);
  }
  return input;
}

/***********************************************************************************************************************
 * WarpReduce Recursive Step
 **********************************************************************************************************************/

// forward declaration
template <typename Input, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_dispatch(Input, ReductionOp, Config);

_CCCL_TEMPLATE(typename Input, typename ReductionOp, typename Config)
_CCCL_REQUIRES(is_cuda_std_bitwise_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static Input
warp_reduce_recursive(Input input, ReductionOp reduction_op, Config warp_config)
{
  using cub::internal::merge_integers;
  using cub::internal::split_integers;
  using cub::internal::warp_reduce_dispatch;
  auto [high, low]    = split_integers(input);
  auto high_reduction = warp_reduce_dispatch(high, reduction_op, warp_config);
  auto low_reduction  = warp_reduce_dispatch(low, reduction_op, warp_config);
  return merge_integers(high_reduction, low_reduction);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp, typename Config)
_CCCL_REQUIRES(is_cuda_minimum_maximum_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static Input
warp_reduce_recursive(Input input, ReductionOp reduction_op, Config warp_config)
{
  using namespace _CUDA_VSTD;
  using cub::internal::merge_integers;
  using cub::internal::split_integers;
  using cub::internal::warp_reduce_dispatch;
  constexpr auto half_bits = __num_bits_v<Input> / 2;
  auto [high, low]         = split_integers(input);
  auto high_result         = warp_reduce_dispatch(high, reduction_op, warp_config);
  if (high_result == 0) // shortcut: input is in range [0, 2^N/2)
  {
    return warp_reduce_dispatch(low, reduction_op, warp_config);
  }
  if (is_unsigned_v<Input> || high_result > 0) // >= 2^N/2 -> perform the computation as uint32_t
  {
    using half_size_unsigned_t = __make_nbit_uint_t<half_bits>;
    constexpr auto identity    = identity_v<ReductionOp, half_size_unsigned_t>;
    auto low_unsigned          = static_cast<half_size_unsigned_t>(low);
    auto low_selected          = high_result == high ? low_unsigned : identity;
    auto low_result            = warp_reduce_dispatch(low_selected, reduction_op, warp_config);
    return static_cast<Input>(merge_integers(static_cast<half_size_unsigned_t>(high_result), low_result));
  }
  // signed type and < 0
  using half_size_signed_t = __make_nbit_int_t<half_bits, true>;
  constexpr auto identity  = identity_v<ReductionOp, half_size_signed_t>;
  auto low_selected        = high_result == high ? static_cast<half_size_signed_t>(low) : identity;
  auto low_result          = warp_reduce_dispatch(low_selected, reduction_op, warp_config);
  return merge_integers(static_cast<half_size_signed_t>(high_result), low_result);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp, typename Config)
_CCCL_REQUIRES(cub::internal::is_cuda_std_plus_v<ReductionOp, Input>)
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input
warp_reduce_recursive(Input input, ReductionOp reduction_op, Config warp_config)
{
  using namespace _CUDA_VSTD;
  using internal::merge_integers;
  using internal::split_integers;
  using internal::warp_reduce_dispatch;
  using unsigned_t         = make_unsigned_t<Input>;
  constexpr auto half_bits = __num_bits_v<Input> / 2;
  auto [high, low]         = split_integers(static_cast<unsigned_t>(input));
  auto high_reduction      = warp_reduce_dispatch(high, reduction_op, warp_config);
  auto low_digits          = static_cast<uint32_t>(low >> (half_bits - 5)); // carry out bits
  auto carry_out           = warp_reduce_dispatch(low_digits, reduction_op, warp_config);
  auto low_reduction       = warp_reduce_dispatch(low, reduction_op, warp_config);
  auto result_high         = high_reduction + (carry_out >> 5);
  return merge_integers(result_high, low_reduction);
}

/***********************************************************************************************************************
 * WarpReduce Dispatch
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input
warp_reduce_dispatch(Input input, ReductionOp reduction_op, Config config)
{
  using cub::internal::comparable_int_to_floating_point;
  using cub::internal::floating_point_to_comparable_int;
  using cub::internal::logical_lane_id;
  using cub::internal::unsafe_bitcast;
  using cub::internal::warp_reduce_dispatch;
  using cub::internal::warp_reduce_generic;
  using cub::internal::warp_reduce_recursive;
  using cub::internal::warp_reduce_redux_op;
  using cub::internal::warp_reduce_shuffle_op;
  using namespace _CUDA_VSTD;
  check_warp_reduce_config(config);
  constexpr bool is_specialized_operator =
    is_cuda_minimum_maximum_v<ReductionOp, Input> || is_cuda_std_plus_v<ReductionOp, Input>
    || is_cuda_std_bitwise_v<ReductionOp, Input>;
  constexpr bool is_small_integer  = is_integral_v<Input> && sizeof(Input) <= sizeof(uint32_t);
  constexpr auto logical_warp_size = config.logical_size;
  auto valid_items                 = config.valid_items;
  // early exit for threads outside the range with dynamic number of valid items
  if (!config.is_segmented && valid_items.rank_dynamic() == 1
      && logical_lane_id(logical_warp_size) >= valid_items.extent(0))
  {
    return input;
  }
  // [Min/Max]: float, double
  if constexpr (is_cuda_minimum_maximum_v<ReductionOp, Input> && is_floating_point_v<Input>)
  {
    if constexpr (is_same_v<Input, float> && __cccl_ptx_isa >= 860)
    {
      NV_IF_TARGET(NV_HAS_FEATURE_SM_100a, (return warp_reduce_redux_op(input, reduction_op, config);));
    }
    auto input_int  = floating_point_to_comparable_int(reduction_op, input);
    auto result_int = warp_reduce_dispatch(input_int, reduction_op, config);
    auto result_rev = comparable_int_to_floating_point(result_int); // reverse
    return unsafe_bitcast<Input>(result_rev);
  }
  // [Min/Max]: __half, __half2, __nv_bfloat16, __nv_bfloat162
  else if constexpr (is_cuda_minimum_maximum_v<ReductionOp, Input>
                     && (is_any_half_v<Input> || is_any_bfloat16_v<Input>) )
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return warp_reduce_shuffle_op(input, reduction_op, config);))
    _CCCL_UNREACHABLE(); // "__half is not supported before SM80"
  }
  // [Min/Max]: short2, ushort2
  else if constexpr (is_cuda_minimum_maximum_v<ReductionOp, Input> && is_any_short2_v<Input> && __cccl_ptx_isa >= 800)
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      (return warp_reduce_shuffle_op(input, reduction_op, config);),
                      (return warp_reduce_generic(input, reduction_op, config);))
  }
  // [Plus]: any complex
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && __is_complex_v<Input>)
  {
#if _CCCL_HAS_NVFP16()
    if constexpr (is_half_v<typename Input::value_type>)
    {
      auto half2_value = unsafe_bitcast<__half2>(input);
      auto ret         = warp_reduce_dispatch(half2_value, reduction_op, config);
      return unsafe_bitcast<Input>(ret);
    }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
    if constexpr (is_bfloat16_v<typename Input::value_type>)
    {
      auto bfloat2_value = unsafe_bitcast<__nv_bfloat162>(input);
      auto ret           = warp_reduce_dispatch(bfloat2_value, reduction_op, config);
      return unsafe_bitcast<Input>(ret);
    }
#endif // _CCCL_HAS_NVBF16()
    if constexpr (is_same_v<typename Input::value_type, float>)
    {
      auto float2_value = unsafe_bitcast<float2>(input);
      auto ret          = warp_reduce_dispatch(float2_value, reduction_op, config);
      return unsafe_bitcast<Input>(ret);
    }
    auto real = warp_reduce_dispatch(input.real(), reduction_op, config);
    auto img  = warp_reduce_dispatch(input.imag(), reduction_op, config);
    return Input{real, img};
  }
  // [Plus]: __half, __half2
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && is_any_half_v<Input>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_53, (return warp_reduce_shuffle_op(input, reduction_op, config);));
    // TODO: _CCCL_ASSERT(false, "half is not supported before SM53");
    _CCCL_UNREACHABLE();
  }
  // [Plus]: __nv_bfloat16, __nv_bfloat162
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && is_any_bfloat16_v<Input>)
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      (return warp_reduce_shuffle_op(input, reduction_op, config);),
                      (return warp_reduce_generic(input, reduction_op, config);));
  }
  // [Plus]: float2
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && is_same_v<Input, float2> && __cccl_ptx_isa >= 860)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_100,
      (return warp_reduce_shuffle_op(input, reduction_op, config);),
      (auto x = warp_reduce_shuffle_op(input.x, reduction_op, config);
       auto y = warp_reduce_shuffle_op(input.y, reduction_op, config);
       return float2{x, y};))
  }
  // [Plus]: short2, ushort2
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && is_any_short2_v<Input> && __cccl_ptx_isa >= 800)
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      (return warp_reduce_shuffle_op(input, reduction_op, config);),
                      (return warp_reduce_generic(input, reduction_op, config);));
  }
  // [Plus/Min/Max]: small integrals (int8, uint8, int16, uint16, int32, uint32)
  // [Plus]:         float, double
  else if constexpr (is_specialized_operator && (is_small_integer || is_floating_point_v<Input>) )
  {
    static_assert(!is_cuda_std_bitwise_v<ReductionOp, Input> || is_unsigned_v<Input>,
                  "Bitwise reduction operations are only supported for unsigned integral types.");
    if constexpr (is_integral_v<Input>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80, (return warp_reduce_redux_op(input, reduction_op, config);));
    }
    return warp_reduce_shuffle_op(input, reduction_op, config);
  }
  else
  {
    // [Plus/Min/Max]: large integrals (int64, uint64, int128, uint128)
    if constexpr (is_specialized_operator && is_integral_v<Input>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80, (return warp_reduce_recursive(input, reduction_op, config);));
    }
    return warp_reduce_generic(input, reduction_op, config);
  }
}

template <bool IsHeadSegment,
          typename FlagT,
          typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          int LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_segmented_reduce_dispatch(
  Input input,
  FlagT flag,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  logical_warp_size_t<LogicalWarpSize> logical_size)
{
  auto logical_base_warp_id = cub::internal::logical_warp_base_id(logical_size);
  auto member_mask          = cub::internal::reduce_lane_mask(logical_mode, logical_size, valid_items_t<0>{});
  auto warp_flags           = __ballot_sync(member_mask, flag);
  warp_flags >>= IsHeadSegment; // Convert to tail-segmented
  warp_flags |= (1u << (logical_base_warp_id + LogicalWarpSize - 1)); // Mask in the last lane of each logical warp
  auto warp_flags_last  = warp_flags & _CUDA_VPTX::get_sreg_lanemask_ge(); // Clean the bits below the current thread
  auto warp_flags_first = warp_flags & _CUDA_VPTX::get_sreg_lanemask_lt(); // Clean the bits after the current thread
  auto last_lane        = _CUDA_VSTD::countr_zero(warp_flags_last); // Find the next set flag
  auto first_lane       = detail::warp_threads - _CUDA_VSTD::countl_zero(warp_flags_first); // Find the previous flag
  auto valid_items      = valid_items_t<>{last_lane};
  WarpReduceConfig config{
    logical_mode, first_lane_result, logical_size, valid_items, is_segmented_t<true>{}, first_lane};
  return cub::internal::warp_reduce_dispatch(input, reduction_op, config);
}

} // namespace internal
CUB_NAMESPACE_END
