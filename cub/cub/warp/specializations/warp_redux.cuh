// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * Helpers for warp-level REDUX reductions.
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

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/std/__floating_point/cast.h> // IWYU pragma: keep
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
//----------------------------------------------------------------------------------------------------------------------
// Redux Traits

template <typename Op, typename T, typename ReduceOp = ::cuda::std::remove_cvref_t<Op>>
inline constexpr bool is_warp_redux_op_supported_sm80 =
  ::cuda::std::is_integral_v<T> && sizeof(T) <= sizeof(unsigned)
  && (is_cuda_minimum_maximum_v<ReduceOp, T> || is_cuda_std_plus_v<ReduceOp, T> || is_cuda_std_bitwise_v<ReduceOp, T>);

template <typename Op, typename T, typename ReduceOp = ::cuda::std::remove_cvref_t<Op>>
inline constexpr bool is_warp_redux_op_supported_sm100af =
  __cccl_ptx_isa >= 860 && (::cuda::std::is_same_v<T, float> || is_half_v<T> || is_bfloat16_v<T>)
  && is_cuda_minimum_maximum_v<ReduceOp, T>;

template <typename Op, typename T>
inline constexpr bool is_warp_redux_op_supported =
  is_warp_redux_op_supported_sm80<Op, T> || is_warp_redux_op_supported_sm100af<Op, T>;

//----------------------------------------------------------------------------------------------------------------------
// SM80 Redux

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
warp_redux_sm80(const T input, const ::cuda::std::uint32_t mask, ReductionOp)
{
  static_assert(is_warp_redux_op_supported_sm80<ReductionOp, T>, "Reduction operator not supported");
  _CCCL_ASSERT(mask != 0, "Mask must not be 0");

  using promotion_t = ::cuda::std::conditional_t<::cuda::std::is_signed_v<T>, int, unsigned>;
  const auto value  = static_cast<promotion_t>(input);
  if constexpr (is_cuda_maximum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_max_sync(mask, value));
  }
  else if constexpr (is_cuda_minimum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_min_sync(mask, value));
  }
  else if constexpr (is_cuda_std_plus_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_add_sync(mask, value));
  }
  else if constexpr (is_cuda_std_bit_and_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_and_sync(mask, value));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_or_sync(mask, value));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_xor_sync(mask, value));
  }
  else
  {
    _CCCL_UNREACHABLE();
    return T{};
  }
}

//----------------------------------------------------------------------------------------------------------------------
// SM100af Redux

#if __cccl_ptx_isa >= 860

#  define _CUB_REDUX_FLOAT_OP(_CCCL_PTX_OP)                                                    \
    [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE float redux_sm100af_##_CCCL_PTX_OP##_ptx( \
      const float value, ::cuda::std::uint32_t mask)                                           \
    {                                                                                          \
      float result;                                                                            \
      asm volatile("{"                                                                         \
                   "redux.sync." #_CCCL_PTX_OP ".f32 %0, %1, %2;"                              \
                   "}"                                                                         \
                   : "=f"(result)                                                              \
                   : "f"(value), "r"(mask));                                                   \
      return result;                                                                           \
    }

_CUB_REDUX_FLOAT_OP(min)
_CUB_REDUX_FLOAT_OP(max)
// TODO(fbusato): min_abs, max_abs are also available but we need to introduce the corresposing operators
// _CUB_REDUX_FLOAT_OP(min_abs)
// _CUB_REDUX_FLOAT_OP(max_abs)

#  undef _CUB_REDUX_FLOAT_OP

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
warp_redux_sm100af(const T input, const ::cuda::std::uint32_t mask, ReductionOp)
{
  static_assert(is_warp_redux_op_supported_sm100af<ReductionOp, T>, "Reduction operator not supported");
  _CCCL_ASSERT(mask != 0, "Mask must not be 0");

  const float value = ::cuda::std::__fp_cast<float>(input);
  float result;
  if constexpr (is_cuda_minimum_v<ReductionOp, T>)
  {
    result = cub::detail::redux_sm100af_min_ptx(value, mask);
  }
  else
  {
    result = cub::detail::redux_sm100af_max_ptx(value, mask);
  }
  return ::cuda::std::__fp_cast<T>(result);
}

#endif // __cccl_ptx_isa >= 860

//----------------------------------------------------------------------------------------------------------------------
// Redux Dispatch

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE constexpr ::cuda::std::optional<T>
warp_redux(const T input, const ::cuda::std::uint32_t mask, ReductionOp reduction_op)
{
  static_assert(is_warp_redux_op_supported<ReductionOp, T>, "Reduction operator not supported");
  if constexpr (is_warp_redux_op_supported_sm80<ReductionOp, T>)
  { // NOLINT(bugprone-branch-clone)
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return cub::detail::warp_redux_sm80(input, mask, reduction_op);))
  }
  else if constexpr (is_warp_redux_op_supported_sm100af<ReductionOp, T>)
  {
    // Before PTX ISA 8.8, float reductions are only supported on sm100a.
#if __cccl_ptx_isa >= 880
    NV_IF_TARGET(NV_HAS_FEATURE_SM_100f, (return cub::detail::warp_redux_sm100af(input, mask, reduction_op);))
#else // ^^^ __cccl_ptx_isa >= 880 ^^^ / vvv __cccl_ptx_isa < 880 vvv
    NV_IF_TARGET(NV_HAS_FEATURE_SM_100a, (return cub::detail::warp_redux_sm100af(input, mask, reduction_op);))
#endif // ^^^ __cccl_ptx_isa < 880 ^^^
  }
  return ::cuda::std::nullopt;
}
} // namespace detail

CUB_NAMESPACE_END
