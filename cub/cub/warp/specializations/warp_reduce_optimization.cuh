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
#include <cub/thread/thread_operators.cuh>
#include <cub/warp/specializations/warp_reduce_config.cuh>
#include <cub/warp/specializations/warp_reduce_ptx.cuh>
#include <cub/warp/specializations/warp_utils.cuh>

#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/ptx>
#include <cuda/std/__complex/is_complex.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/bit>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN

namespace detail
{

/***********************************************************************************************************************
 * WarpReduce Base Step
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp, int LogicalWarpSize, size_t ValidItems>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_redux_op(
  Input input,
  ReductionOp reduction_op,
  logial_warp_size_t<LogicalWarpSize> logical_size,

  valid_items_t<ValidItems> valid_items)
{
  using namespace _CUDA_VSTD;
  using namespace internal;
  static_assert(sizeof(Input) <= sizeof(uint32_t));
  const auto mask = cub::detail::reduce_member_mask(single_reduction, logical_size, valid_items);
  if constexpr (is_integral_v<Input>)
  {
    using cast_t = _If<is_signed_v<Input>, int, uint32_t>;
    if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
    {
      return static_cast<Input>(::__reduce_and_sync(mask, input));
    }
    else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
    {
      return static_cast<Input>(::__reduce_or_sync(mask, input));
    }
    else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
    {
      return static_cast<Input>(::__reduce_xor_sync(mask, input));
    }
    else if constexpr (is_cuda_std_plus_v<ReductionOp, Input>)
    {
      return ::__reduce_add_sync(mask, static_cast<cast_t>(input));
    }
    else if constexpr (is_cuda_minimum_v<ReductionOp, Input>)
    {
      return ::__reduce_min_sync(mask, static_cast<cast_t>(input));
    }
    else if constexpr (is_cuda_maximum_v<ReductionOp, Input>)
    {
      return ::__reduce_max_sync(mask, static_cast<cast_t>(input));
    }
  }
  else if constexpr (is_same_v<Input, float> && is_cuda_std_min_max_v<ReductionOp, Input>)
  {
    return cub::detail::reduce_sm100a_sync(reduction_op, input, mask);
  }
  else
  {
    static_assert(__always_false_v<Input>, "invalid input type/reduction operator combination");
    _CCCL_UNREACHABLE();
  }
}

template <typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          WarpReduceResultMode Kind,
          int LogicalWarpSize,
          size_t ValidItems>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_shuffle_op(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<Kind> result_mode,
  logial_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items)
{
  using namespace _CUDA_VSTD;
  using namespace internal;
  static_assert(is_integral_v<Input> || is_arithmetic_cuda_floating_point_v<Input>,
                "invalid input type/reduction operator combination");
  constexpr auto log2_size = ::cuda::ilog2(LogicalWarpSize * 2 - 1);
  const auto mask          = cub::detail::reduce_member_mask(logical_mode, logical_size, valid_items);
  using cast_t             = _If<is_integral_v<Input> && sizeof(Input) < sizeof(int), int, Input>;
  auto input1              = static_cast<cast_t>(input);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int K = 0; K < log2_size; K++)
  {
    auto shuffle_mask = cub::detail::reduce_shuffle_mask(K, logical_size, valid_items);
    input1            = cub::detail::shfl_down_op(reduction_op, input1, 1u << K, shuffle_mask, mask);
  }
  if constexpr (result_mode == all_lanes_result)
  {
    constexpr auto logical_width =
      _CUDA_VSTD::has_single_bit(uint32_t{LogicalWarpSize}) ? LogicalWarpSize : warp_threads;
    input1 = _CUDA_VDEV::warp_shuffle_idx<logical_width>(input1, 0, mask);
  }

  return input1;
}

template <typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          WarpReduceResultMode Kind,
          int LogicalWarpSize,
          size_t ValidItems>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_generic(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<Kind> result_mode,
  logial_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items)
{
  constexpr auto logical_size_round = _CUDA_VSTD::bit_ceil(uint32_t{LogicalWarpSize});
  constexpr auto log2_size          = ::cuda::ilog2(LogicalWarpSize * 2 - 1);
  const auto mask                   = cub::detail::reduce_member_mask(logical_mode, logical_size, valid_items);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int K = 0; K < log2_size; K++)
  {
    _CUDA_VDEV::WarpShuffleResult<Input> res;
    if constexpr (valid_items.rank_dynamic() == 0 && _CUDA_VSTD::has_single_bit(uint32_t{LogicalWarpSize}))
    {
      res = _CUDA_VDEV::warp_shuffle_down(input, 1u << K, mask, logical_size);
    }
    else
    {
      auto lane_id = logical_lane_id(LogicalWarpSize);
      auto dest    = ::min(lane_id + (1u << K), valid_items.extent(0) - 1);
      res          = _CUDA_VDEV::warp_shuffle_idx<logical_size_round>(input, dest, mask);
      res.pred     = lane_id + (1u << K) < valid_items.extent(0);
    }
    if (res.pred)
    {
      input = reduction_op(input, res.data);
    }
  }
  if constexpr (result_mode == all_lanes_result)
  {
    input = _CUDA_VDEV::warp_shuffle_idx<logical_size_round>(input, 0, mask);
  }
  return input;
}

/***********************************************************************************************************************
 * WarpReduce Recursive Step
 **********************************************************************************************************************/

// forward declaration
template <typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          WarpReduceResultMode ResultMode,
          int LogicalWarpSize,
          size_t ValidItems = LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_dispatch(
  Input,
  ReductionOp,
  reduce_logical_mode_t<LogicalMode>,
  reduce_result_mode_t<ResultMode>,
  logial_warp_size_t<LogicalWarpSize>,
  valid_items_t<ValidItems> = {});

_CCCL_TEMPLATE(typename Input,
               typename ReductionOp,
               ReduceLogicalMode LogicalMode,
               WarpReduceResultMode ResultMode,
               int LogicalWarpSize,
               size_t ValidItems)
_CCCL_REQUIRES(cub::internal::is_cuda_std_bitwise_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<ResultMode> result_mode,
  logial_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items)
{
  using detail::merge_integers;
  using detail::split_integers;
  using detail::warp_reduce_dispatch;
  auto [high, low]    = split_integers(input);
  auto high_reduction = warp_reduce_dispatch(high, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  auto low_reduction  = warp_reduce_dispatch(low, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  return merge_integers(high_reduction, low_reduction);
}

_CCCL_TEMPLATE(typename Input,
               typename ReductionOp,
               ReduceLogicalMode LogicalMode,
               WarpReduceResultMode ResultMode,
               int LogicalWarpSize,
               size_t ValidItems)
_CCCL_REQUIRES(cub::internal::is_cuda_std_min_max_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<ResultMode> result_mode,
  logial_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items)
{
  using namespace _CUDA_VSTD;
  using detail::merge_integers;
  using detail::split_integers;
  using detail::warp_reduce_dispatch;
  using internal::identity_v;
  constexpr auto half_bits = __num_bits_v<Input> / 2;
  auto [high, low]         = split_integers(input);
  auto high_result = warp_reduce_dispatch(high, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  if (high_result == 0) // shortcut: input is in range [0, 2^N/2)
  {
    return warp_reduce_dispatch(low, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  }
  if (is_unsigned_v<Input> || high_result > 0) // >= 2^N/2 -> perform the computation as uint32_t
  {
    using half_size_unsigned_t = __make_nbit_uint_t<half_bits>;
    constexpr auto identity    = identity_v<ReductionOp, half_size_unsigned_t>;
    auto low_unsigned          = static_cast<half_size_unsigned_t>(low);
    auto low_selected          = high_result == high ? low_unsigned : identity;
    auto low_result =
      warp_reduce_dispatch(low_selected, reduction_op, logical_mode, result_mode, logical_size, valid_items);
    return static_cast<Input>(merge_integers(static_cast<half_size_unsigned_t>(high_result), low_result));
  }
  // signed type and < 0
  using half_size_signed_t = __make_nbit_int_t<half_bits, true>;
  constexpr auto identity  = identity_v<ReductionOp, half_size_signed_t>;
  auto low_selected        = high_result == high ? static_cast<half_size_signed_t>(low) : identity;
  auto low_result =
    warp_reduce_dispatch(low_selected, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  return merge_integers(static_cast<half_size_signed_t>(high_result), low_result);
}

_CCCL_TEMPLATE(typename Input,
               typename ReductionOp,
               ReduceLogicalMode LogicalMode,
               WarpReduceResultMode ResultMode,
               int LogicalWarpSize,
               size_t ValidItems)
_CCCL_REQUIRES(cub::internal::is_cuda_std_plus_v<ReductionOp, Input>)
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<ResultMode> result_mode,
  logial_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items)
{
  using namespace _CUDA_VSTD;
  using detail::merge_integers;
  using detail::split_integers;
  using detail::warp_reduce_dispatch;
  using unsigned_t         = make_unsigned_t<Input>;
  constexpr auto half_bits = __num_bits_v<Input> / 2;
  auto [high, low]         = split_integers(static_cast<unsigned_t>(input));
  auto high_reduction = warp_reduce_dispatch(high, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  auto low_digits     = static_cast<uint32_t>(low >> (half_bits - 5));
  auto carry_out = warp_reduce_dispatch(low_digits, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  auto low_reduction = warp_reduce_dispatch(low, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  auto result_high   = high_reduction + (carry_out >> 5);
  return merge_integers(result_high, low_reduction);
}

/***********************************************************************************************************************
 * WarpReduce Dispatch
 **********************************************************************************************************************/

template <typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          WarpReduceResultMode ResultMode,
          int LogicalWarpSize,
          size_t ValidItems>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_dispatch(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<ResultMode> result_mode,
  logial_warp_size_t<LogicalWarpSize> logical_size,
  valid_items_t<ValidItems> valid_items)
{
  using cub::detail::comparable_int_to_floating_point;
  using cub::detail::floating_point_to_comparable_int;
  using cub::detail::unsafe_bitcast;
  using cub::detail::warp_reduce_dispatch;
  using cub::detail::warp_reduce_generic;
  using cub::detail::warp_reduce_recursive;
  using cub::detail::warp_reduce_redux_op;
  using cub::detail::warp_reduce_shuffle_op;
  using namespace cub::internal;
  using namespace _CUDA_VSTD;
  // early exit for threads outside the range with dynamic number of valid items
  if (valid_items.rank_dynamic() == 1 && logical_lane_id(LogicalWarpSize) >= valid_items.extent(0))
  {
    return Input{};
  }
  // Min/Max: __half, __nv_bfloat16, float, double
  if constexpr (is_cuda_std_min_max_v<ReductionOp, Input> && is_arithmetic_cuda_floating_point_v<Input>)
  {
    if constexpr (is_same_v<Input, float> && logical_mode == single_reduction)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_100, (return warp_reduce_redux_op(input, reduction_op, logical_size, valid_items);));
    }
    auto input_int = floating_point_to_comparable_int(reduction_op, input);
    auto result_int =
      warp_reduce_dispatch(input_int, reduction_op, logical_mode, result_mode, logical_size, valid_items);
    auto result_rev = comparable_int_to_floating_point(result_int); // reverse
    return unsafe_bitcast<Input>(result_rev);
  }
  // Min/Max: __half2, __nv_bfloat162
  else if constexpr (is_cuda_std_min_max_v<ReductionOp, Input> && (is_half_X2_v<Input> || is_bfloat16_X2_v<Input>) )
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_80,
      (return warp_reduce_shuffle_op(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);),
      (return warp_reduce_generic(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);))
  }
  // Plus, any complex
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && __is_complex_v<Input>)
  {
    if constexpr (is_half_v<typename Input::value_type>)
    {
      auto half2_value = unsafe_bitcast<__half2>(input);
      auto ret = warp_reduce_dispatch(half2_value, reduction_op, logical_mode, result_mode, logical_size, valid_items);
      return unsafe_bitcast<Input>(ret);
    }
    if constexpr (is_bfloat16_v<typename Input::value_type>)
    {
      auto bfloat2_value = unsafe_bitcast<__nv_bfloat162>(input);
      auto ret =
        warp_reduce_dispatch(bfloat2_value, reduction_op, logical_mode, result_mode, logical_size, valid_items);
      return unsafe_bitcast<Input>(ret);
    }
    if constexpr (is_same_v<typename Input::value_type, float>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_100,
                   (auto float2_value = _CUDA_VSTD::bit_cast<float2>(input); //
                    return warp_reduce_shuffle_op(
                      float2_value, reduction_op, logical_mode, result_mode, logical_size, valid_items);))
    }
    auto real = warp_reduce_dispatch(input.real(), reduction_op, logical_mode, result_mode, logical_size, valid_items);
    auto img  = warp_reduce_dispatch(input.imag(), reduction_op, logical_mode, result_mode, logical_size, valid_items);
    return Input{real, img};
  }
  // Plus, bfloat16, bfloat16 X2
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && is_any_bfloat16_v<Input>)
  {
    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (return warp_reduce_shuffle_op(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);),
      (return warp_reduce_generic(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);));
  }
  // Plus, float2
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input> && (is_same_v<Input, float2>) )
  {
    NV_IF_TARGET(
      NV_PROVIDES_SM_100,
      (return warp_reduce_shuffle_op(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);),
      (return warp_reduce_generic(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);));
  }
  // any std operator, large integrals
  else if constexpr (is_cuda_std_operator_v<ReductionOp, Input> && is_integral_v<Input>
                     && sizeof(Input) > sizeof(uint32_t))
  {
    return warp_reduce_recursive(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  }
  // any std operator, small integrals and supported floating point
  else if constexpr (is_cuda_std_operator_v<ReductionOp, Input>
                     && (is_integral_v<Input> || is_arithmetic_cuda_floating_point_v<Input>) )
  {
    static_assert(!is_cuda_std_bitwise_v<ReductionOp, Input> || is_unsigned_v<Input>,
                  "Bitwise reduction operations are only supported for unsigned integral types.");
    if constexpr (is_integral_v<Input> && logical_mode == single_reduction)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80, (return warp_reduce_redux_op(input, reduction_op, logical_size, valid_items);));
    }
    return warp_reduce_shuffle_op(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  }
  else
  {
    return warp_reduce_generic(input, reduction_op, logical_mode, result_mode, logical_size, valid_items);
  }
}

} // namespace detail

CUB_NAMESPACE_END
