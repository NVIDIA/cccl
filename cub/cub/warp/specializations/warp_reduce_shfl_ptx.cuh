// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/integer_utility.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/__functional/operator_properties.h>
#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN
namespace detail
{
template <typename ReductionOp, typename T>
inline constexpr bool is_shfl_down_op_supported_v =
  (::cuda::__is_cuda_std_plus_v<ReductionOp, T>
   && (::cuda::std::is_same_v<T, unsigned> //
       || ::cuda::std::is_same_v<T, float> //
       || ::cuda::std::is_same_v<T, unsigned long long> //
       || ::cuda::std::is_same_v<T, long long> //
       || ::cuda::std::is_same_v<T, double>) )
  || ((::cuda::__is_cuda_minimum_v<ReductionOp, T> || ::cuda::__is_cuda_maximum_v<ReductionOp, T>)
      && ((::cuda::std::__cccl_is_integer_v<T> && sizeof(T) <= 8)
          || ::cuda::std::is_same_v<T, float> || ::cuda::std::is_same_v<T, double>) );

// Use the predicate set by SHFL to guard against invalid peers.
#define _CUB_SHFL_DOWN_OP_32BIT(OP, TYPE, PTX_TYPE, CONSTRAINT)                            \
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TYPE shfl_down_##OP(                    \
    const TYPE input, const int offset, const int shfl_c, const unsigned member_mask)      \
  {                                                                                        \
    TYPE output;                                                                           \
    asm volatile(                                                                          \
      "{"                                                                                  \
      "  .reg ." PTX_TYPE " r0;"                                                           \
      "  .reg .pred p;"                                                                    \
      "  shfl.sync.down.b32 r0|p, %1, %2, %3, %5;"                                         \
      "  @p " #OP "." PTX_TYPE " r0, r0, %4;"                                              \
      "  mov." PTX_TYPE " %0, r0;"                                                         \
      "}"                                                                                  \
      : "=" CONSTRAINT(output)                                                             \
      : CONSTRAINT(input), "r"(offset), "r"(shfl_c), CONSTRAINT(input), "r"(member_mask)); \
    return output;                                                                         \
  }

_CUB_SHFL_DOWN_OP_32BIT(add, unsigned, "u32", "r")
_CUB_SHFL_DOWN_OP_32BIT(add, float, "f32", "f")

_CUB_SHFL_DOWN_OP_32BIT(min, int, "s32", "r")
_CUB_SHFL_DOWN_OP_32BIT(min, unsigned, "u32", "r")
_CUB_SHFL_DOWN_OP_32BIT(min, float, "f32", "f")
_CUB_SHFL_DOWN_OP_32BIT(max, int, "s32", "r")
_CUB_SHFL_DOWN_OP_32BIT(max, unsigned, "u32", "r")
_CUB_SHFL_DOWN_OP_32BIT(max, float, "f32", "f")

#undef _CUB_SHFL_DOWN_OP_32BIT

#define _CUB_SHFL_DOWN_OP_64BIT(OP, TYPE, PTX_TYPE, CONSTRAINT)                       \
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TYPE shfl_down_##OP(               \
    const TYPE input, const int offset, const int shfl_c, const unsigned member_mask) \
  {                                                                                   \
    TYPE output;                                                                      \
    asm volatile(                                                                     \
      "{"                                                                             \
      "  .reg .u32 lo;"                                                               \
      "  .reg .u32 hi;"                                                               \
      "  .reg .pred p;"                                                               \
      "  .reg ." PTX_TYPE " r0;"                                                      \
      "  mov.b64 %0, %1;"                                                             \
      "  mov.b64 {lo, hi}, %1;"                                                       \
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"                                    \
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"                                    \
      "  mov.b64 r0, {lo, hi};"                                                       \
      "  @p " #OP "." PTX_TYPE " %0, %0, r0;"                                         \
      "}"                                                                             \
      : "=" CONSTRAINT(output)                                                        \
      : CONSTRAINT(input), "r"(offset), "r"(shfl_c), "r"(member_mask));               \
    return output;                                                                    \
  }

_CUB_SHFL_DOWN_OP_64BIT(add, unsigned long long, "u64", "l")
_CUB_SHFL_DOWN_OP_64BIT(add, long long, "s64", "l")
_CUB_SHFL_DOWN_OP_64BIT(add, double, "f64", "d")

_CUB_SHFL_DOWN_OP_64BIT(min, ::cuda::std::int64_t, "s64", "l")
_CUB_SHFL_DOWN_OP_64BIT(min, ::cuda::std::uint64_t, "u64", "l")
_CUB_SHFL_DOWN_OP_64BIT(min, double, "f64", "d")
_CUB_SHFL_DOWN_OP_64BIT(max, ::cuda::std::int64_t, "s64", "l")
_CUB_SHFL_DOWN_OP_64BIT(max, ::cuda::std::uint64_t, "u64", "l")
_CUB_SHFL_DOWN_OP_64BIT(max, double, "f64", "d")

#undef _CUB_SHFL_DOWN_OP_64BIT

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T shfl_down_op(
  const T input, const ReductionOp /*reduction_op*/, const int offset, const int shfl_c, const unsigned member_mask)
{
  static_assert(is_shfl_down_op_supported_v<ReductionOp, T>, "Unsupported shuffle reduction operator");
  if constexpr (::cuda::__is_cuda_std_plus_v<ReductionOp, T>)
  {
    return _CUB::detail::shfl_down_add(input, offset, shfl_c, member_mask);
  }
  else
  {
    if constexpr (::cuda::std::is_same_v<T, double>)
    {
      _CCCL_DISPATCH_SLOW_FP64_TARGET(
        using op_t _CCCL_NODEBUG = _CUB::detail::generalize_operator_t<ReductionOp>;
        const auto value         = _CUB::detail::floating_point_min_max_conversion(input, op_t{});
        const auto result        = _CUB::detail::shfl_down_op(value, op_t{}, offset, shfl_c, member_mask);
        return _CUB::detail::comparable_int_to_floating_point<T>(result);)
    }
    constexpr bool is_signed      = ::cuda::std::is_signed_v<T>;
    using integer_t _CCCL_NODEBUG = ::cuda::std::__make_nbit_int_t<(sizeof(T) <= 4 ? 32 : 64), is_signed>;
    using shuffle_t _CCCL_NODEBUG = ::cuda::std::conditional_t<::cuda::std::__cccl_is_integer_v<T>, integer_t, T>;
    const auto value              = static_cast<shuffle_t>(input);
    if constexpr (::cuda::__is_cuda_minimum_v<ReductionOp, T>)
    {
      return static_cast<T>(_CUB::detail::shfl_down_min(value, offset, shfl_c, member_mask));
    }
    else
    {
      return static_cast<T>(_CUB::detail::shfl_down_max(value, offset, shfl_c, member_mask));
    }
  }
}
} // namespace detail
CUB_NAMESPACE_END
