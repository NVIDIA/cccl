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

#include <cub/warp/specializations/shfl_down_op.cuh>

#include <cuda/functional>
#include <cuda/ptx>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN

/***********************************************************************************************************************
 * WarpReduce Configuration Enums
 **********************************************************************************************************************/

namespace detail
{

enum class WarpReduceMode
{
  SingleLogicalWarp,
  MultipleLogicalWarps
};

enum class WarpReduceResult
{
  AllLanes,
  SingleLane
};

template <WarpReduceMode Mode>
using warp_reduce_mode_t = _CUDA_VSTD::integral_constant<WarpReduceMode, Mode>;

template <WarpReduceResult Kind>
using warp_reduce_result_t = _CUDA_VSTD::integral_constant<WarpReduceResult, Kind>;

} // namespace detail

inline constexpr auto single_logical_warp = detail::warp_reduce_mode_t<detail::WarpReduceMode::SingleLogicalWarp>{};
inline constexpr auto multiple_logical_warps =
  detail::warp_reduce_mode_t<detail::WarpReduceMode::MultipleLogicalWarps>{};

inline constexpr auto all_lanes_result   = detail::warp_reduce_result_t<detail::WarpReduceResult::AllLanes>{};
inline constexpr auto single_lane_result = detail::warp_reduce_result_t<detail::WarpReduceResult::SingleLane>{};

/***********************************************************************************************************************
 * WarpReduce Base Step
 **********************************************************************************************************************/

namespace detail
{

_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE unsigned warp_reduce_mask(unsigned step)
{
  const auto clamp   = 0b11110u << step;
  const auto segmask = 0b11110u << (step + 8);
  return clamp | segmask;
}

template <unsigned LogicalWarpSize, WarpReduceMode Mode>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE constexpr unsigned member_mask(warp_reduce_mode_t<Mode> mode)
{
  return (mode == single_logical_warp) ? (0xFFFFFFFF >> (warp_threads - LogicalWarpSize)) : 0xFFFFFFFF;
}

template <int LogicalWarpSize, typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult Kind>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_sm30(
  Input input, ReductionOp reduction_op, warp_reduce_mode_t<Mode> mode, warp_reduce_result_t<Kind> result_mode)
{
  using namespace internal;
  constexpr auto mask = cub::detail::member_mask<LogicalWarpSize>(mode);
  if constexpr (_CUDA_VSTD::is_same_v<Input, bool>)
  {
    if constexpr (is_cuda_std_plus_v<ReductionOp, Input>)
    {
      return _CUDA_VSTD::popcount(::__ballot_sync(mask, input));
    }
    else if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
    {
      return ::__all_sync(mask, input);
    }
    else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
    {
      return ::__any_sync(mask, input);
    }
    else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
    {
      return _CUDA_VSTD::popcount(::__ballot_sync(mask, input)) % 2u;
    }
    else
    {
      static_assert(_CUDA_VSTD::__always_false_v<Input>, "invalid reduction operator with bool input type");
      _CCCL_UNREACHABLE();
    }
  }
  else if constexpr (_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) < sizeof(uint32_t))
  {
    return warp_reduce_sm30(static_cast<int>(input), reduction_op, mode, result_mode);
  }
  else if constexpr (_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) == sizeof(uint32_t)
                     || _CUDA_VSTD::is_floating_point_v<Input> || _CUDA_VSTD::__is_extended_floating_point_v<Input>)
  {
    constexpr auto Log2Size     = ::cuda::ilog2(LogicalWarpSize);
    constexpr auto LogicalWidth = _CUDA_VSTD::integral_constant<int, LogicalWarpSize>{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int K = 0; K < Log2Size; K++)
    {
      shfl_down_add(input, 1u << K, cub::detail::warp_reduce_mask(K), mask);
    }
    if constexpr (result_mode == all_lanes_result)
    {
      input = _CUDA_VDEV::warp_shuffle_idx(input, 0, mask, LogicalWidth);
    }
    return input;
  }
  else
  {
    static_assert(_CUDA_VSTD::__always_false_v<Input>, "invalid input type/reduction operator combination");
    _CCCL_UNREACHABLE();
  }
}

template <unsigned LogicalWarpSize, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_sm80(Input input, ReductionOp)
{
  using namespace internal;
  static_assert(_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) <= sizeof(uint32_t));
  using cast_t        = _CUDA_VSTD::_If<_CUDA_VSTD::is_signed_v<Input>, int32_t, uint32_t>;
  constexpr auto mask = cub::detail::member_mask<LogicalWarpSize>(single_logical_warp);
  if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_and_sync(mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_or_sync(mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_xor_sync(mask, static_cast<uint32_t>(input)));
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
  else
  {
    static_assert(_CUDA_VSTD::__always_false_v<Input>, "invalid input type/reduction operator combination");
    _CCCL_UNREACHABLE();
  }
}

template <int LogicalWarpSize, typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult Kind>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE Input
warp_reduce_generic(Input input, ReductionOp, warp_reduce_mode_t<Mode> mode, warp_reduce_result_t<Kind> result_mode)
{
  constexpr auto Log2Size     = ::cuda::ilog2(LogicalWarpSize);
  constexpr auto LogicalWidth = _CUDA_VSTD::integral_constant<int, LogicalWarpSize>{};
  constexpr auto mask         = cub::detail::member_mask<LogicalWarpSize>(mode);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int K = 0; K < Log2Size; K++)
  {
    auto res = _CUDA_VDEV::warp_shuffle_down(input, 1u << K, mask, LogicalWidth);
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

/***********************************************************************************************************************
 * WarpReduce Recursive Step
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult ResultMode>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE
Input warp_reduce_dispatch(Input, ReductionOp, warp_reduce_mode_t<Mode>, warp_reduce_result_t<ResultMode>);

template <typename Input>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto split(Input input)
{
  static_assert(_CUDA_VSTD::is_integral_v<Input>);
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
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto merge(Input inputA, Input inputB)
{
  static_assert(_CUDA_VSTD::is_integral_v<Input>);
  constexpr auto digits = _CUDA_VSTD::numeric_limits<Input>::digits;
  using unsigned_t      = _CUDA_VSTD::__make_nbit_uint_t<digits * 2>;
  using output_t        = _CUDA_VSTD::__make_nbit_int_t<digits * 2, _CUDA_VSTD::is_signed_v<Input>>;
  return static_cast<output_t>(static_cast<unsigned_t>(inputA) << digits | inputB);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult ResultMode)
_CCCL_REQUIRES(cub::internal::is_cuda_std_bitwise_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE static Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  warp_reduce_mode_t<Mode> warp_mode,
  warp_reduce_result_t<ResultMode> result_mode)
{
  using detail::merge;
  using detail::split;
  using detail::warp_reduce_dispatch;
  auto [high, low]    = split(input);
  auto high_reduction = warp_reduce_dispatch(high, reduction_op, warp_mode, result_mode);
  auto low_reduction  = warp_reduce_dispatch(low, reduction_op, warp_mode, result_mode);
  return merge(high_reduction, low_reduction);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult ResultMode)
_CCCL_REQUIRES(cub::internal::is_cuda_std_min_max_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE static Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  warp_reduce_mode_t<Mode> warp_mode,
  warp_reduce_result_t<ResultMode> result_mode)
{
  using detail::merge;
  using detail::split;
  using detail::warp_reduce_dispatch;
  using internal::identity_v;
  constexpr auto half_bits = _CUDA_VSTD::numeric_limits<Input>::digits / 2u;
  auto [high, low]         = split(input);
  auto high_result         = warp_reduce_dispatch(high, reduction_op, warp_mode, result_mode);
  if (high_result == 0) // shortcut: input is in range [0, 2^N/2)
  {
    return warp_reduce_dispatch(low, reduction_op);
  }
  if (_CUDA_VSTD::is_unsigned_v<Input> || high_result > 0) // >= 2^N/2 -> perform the computation as unsigned
  {
    using half_size_unsigned_t = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
    constexpr auto identity    = identity_v<ReductionOp, half_size_unsigned_t>;
    auto low_unsigned          = static_cast<half_size_unsigned_t>(low);
    auto low_result =
      warp_reduce_dispatch(high_result == high ? low_unsigned : identity, reduction_op, warp_mode, result_mode);
    return static_cast<Input>(merge(high_result, low_result));
  }
  // signed type and < 0
  using half_size_signed_t = _CUDA_VSTD::__make_nbit_int_t<half_bits, true>;
  constexpr auto identity  = identity_v<ReductionOp, half_size_signed_t>;
  auto low_result = warp_reduce_dispatch(high_result == high ? low : identity, reduction_op, warp_mode, result_mode);
  return merge(high_result, low_result);
}

_CCCL_TEMPLATE(typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult ResultMode)
_CCCL_REQUIRES(cub::internal::is_cuda_std_plus_v<ReductionOp, Input>)
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  warp_reduce_mode_t<Mode> warp_mode,
  warp_reduce_result_t<ResultMode> result_mode)
{
  using detail::merge;
  using detail::split;
  using detail::warp_reduce_dispatch;
  using unsigned_t      = _CUDA_VSTD::make_unsigned_t<Input>;
  constexpr auto digits = _CUDA_VSTD::numeric_limits<Input>::digits;
  auto [high, low]      = split(input);
  auto high_reduction   = warp_reduce_dispatch(high, reduction_op, warp_mode, result_mode);
  auto low_digits       = static_cast<unsigned_t>(low) >> (digits - 5);
  auto carry_out        = warp_reduce_dispatch(static_cast<uint32_t>(low_digits), reduction_op, warp_mode, result_mode);
  auto low_reduction    = warp_reduce_dispatch(low, reduction_op, warp_mode, result_mode);
  auto result_high      = high_reduction + carry_out;
  return merge(result_high, low_reduction);
}

/***********************************************************************************************************************
 * WarpReduce Dispatch
 **********************************************************************************************************************/

template <typename T>
inline constexpr bool is_complex_v = _CUDA_VSTD::__is_complex<T>::value;

template <typename Input, typename ReductionOp, WarpReduceMode Mode, WarpReduceResult ResultMode>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_dispatch(
  Input input,
  ReductionOp reduction_op,
  warp_reduce_mode_t<Mode> warp_mode,
  warp_reduce_result_t<ResultMode> result_mode)
{
  using internal::is_cuda_std_min_max_v;
  using internal::is_cuda_std_plus_v;
  constexpr bool is_small_integer = _CUDA_VSTD::is_integral_v<Input> && sizeof(Input) <= sizeof(uint32_t);
  constexpr bool is_any_floating_point =
    _CUDA_VSTD::is_floating_point_v<Input> || _CUDA_VSTD::__is_extended_floating_point_v<Input>;
  constexpr bool is_supported_floating_point =
    _CUDA_VSTD::is_floating_point_v<Input> || is_one_of_v<Input, __half, __half2, __nv_bfloat16, __nv_bfloat162>;
  //
  if constexpr (is_any_floating_point && is_cuda_std_min_max_v<ReductionOp, Input>)
  {
    constexpr auto digits = _CUDA_VSTD::numeric_limits<Input>::digits;
    using signed_t        = _CUDA_VSTD::__make_nbit_int_t<digits, true>;
    auto result = warp_reduce_dispatch(_CUDA_VSTD::bit_cast<signed_t>(input), reduction_op, warp_mode, result_mode);
    return _CUDA_VSTD::bit_cast<Input>(result);
  }
  else if constexpr (is_complex_v<Input> && is_cuda_std_plus_v<ReductionOp, Input>)
  {
    if constexpr (_CUDA_VSTD::is_same_v<typename Input::value_type, __half>)
    {
      auto half2_value = unsafe_bitcast<__half2>(input);
      auto ret         = warp_reduce_dispatch(half2_value, reduction_op, warp_mode, result_mode);
      return unsafe_bitcast<Input>(ret);
    }
    else if constexpr (_CUDA_VSTD::is_same_v<typename Input::value_type, __nv_bfloat16>)
    {
      auto bfloat2_value = unsafe_bitcast<__nv_bfloat162>(input);
      auto ret           = warp_reduce_dispatch(bfloat2_value, reduction_op, warp_mode, result_mode);
      return unsafe_bitcast<Input>(ret);
    }
    else
    {
      auto real = warp_reduce_dispatch(input.real(), reduction_op, warp_mode, result_mode);
      auto img  = warp_reduce_dispatch(input.img(), reduction_op, warp_mode, result_mode);
      return Input{real, img};
    }
  }
  else if constexpr (is_small_integer && warp_mode == single_logical_warp)
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80, //
                      (return warp_reduce_sm80(input, reduction_op);),
                      (return warp_reduce_sm30(input, reduction_op, single_logical_warp, result_mode);));
  }
  else if constexpr (is_small_integer || is_supported_floating_point)
  {
    return warp_reduce_sm30(input, reduction_op, warp_mode);
  }
  else if constexpr (_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) > sizeof(uint32_t))
  {
    return warp_reduce_recursive(input, reduction_op, warp_mode, result_mode);
  }
  else // generic implementation
  {
    return warp_reduce_generic(input, reduction_op, warp_mode, result_mode);
  }
}

} // namespace detail

CUB_NAMESPACE_END
