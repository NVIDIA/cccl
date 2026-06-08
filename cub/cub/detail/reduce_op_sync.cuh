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

#include <cub/thread/thread_operators.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T reduce_op_sync(T input, const ::cuda::std::uint32_t mask, ReductionOp)
{
  static_assert(::cuda::std::is_integral_v<T>, "T must be an integral type");
  static_assert(sizeof(T) <= sizeof(unsigned), "T must be less than or equal to unsigned");
  using promoted_t = ::cuda::std::conditional_t<::cuda::std::is_unsigned_v<T>, unsigned, int>;
  if constexpr (is_cuda_maximum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_max_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_minimum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_min_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_plus_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_add_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_and_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_and_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_or_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_xor_sync(mask, static_cast<promoted_t>(input)));
  }
  else
  {
    _CCCL_UNREACHABLE();
    return T{};
  }
}
} // namespace detail

CUB_NAMESPACE_END
