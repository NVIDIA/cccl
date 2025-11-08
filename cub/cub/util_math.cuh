// SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * Define helper math functions.
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

#include <cuda/cmath>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
using is_integral_or_enum =
  ::cuda::std::integral_constant<bool, ::cuda::std::is_integral_v<T> || ::cuda::std::is_enum_v<T>>;

/**
 * Computes lhs + rhs, but bounds the result to the maximum number representable by the given type, if the addition
 * would overflow. Note, lhs must be non-negative.
 *
 * Effectively performs `min((lhs + rhs), ::cuda::std::numeric_limits<OffsetT>::max())`, but is robust against the case
 * where `(lhs + rhs)` would overflow.
 */
template <typename OffsetT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE OffsetT safe_add_bound_to_max(OffsetT lhs, OffsetT rhs)
{
  static_assert(::cuda::std::is_integral_v<OffsetT>, "OffsetT must be an integral type");
  static_assert(sizeof(OffsetT) >= 4, "OffsetT must be at least 32 bits in size");
  auto const capped_operand_rhs = (::cuda::std::min) (rhs, ::cuda::std::numeric_limits<OffsetT>::max() - lhs);
  return lhs + capped_operand_rhs;
}
} // namespace detail

constexpr _CCCL_HOST_DEVICE int Nominal4BItemsToItemsCombined(int nominal_4b_items_per_thread, int combined_bytes)
{
  return (::cuda::std::min) (nominal_4b_items_per_thread,
                             (::cuda::std::max) (1, nominal_4b_items_per_thread * 8 / combined_bytes));
}

template <typename T>
constexpr _CCCL_HOST_DEVICE int Nominal4BItemsToItems(int nominal_4b_items_per_thread)
{
  return (::cuda::std::min) (nominal_4b_items_per_thread,
                             (::cuda::std::max) (1, nominal_4b_items_per_thread * 4 / static_cast<int>(sizeof(T))));
}

template <typename ItemT>
constexpr _CCCL_HOST_DEVICE int Nominal8BItemsToItems(int nominal_8b_items_per_thread)
{
  return sizeof(ItemT) <= 8u
         ? nominal_8b_items_per_thread
         : (::cuda::std::min) (nominal_8b_items_per_thread,
                               (::cuda::std::max) (1,
                                                   ((nominal_8b_items_per_thread * 8) + static_cast<int>(sizeof(ItemT))
                                                    - 1)
                                                     / static_cast<int>(sizeof(ItemT))));
}

/**
 * \brief Computes the midpoint of the integers
 *
 * Extra operation is performed in order to prevent overflow.
 *
 * \return Half the sum of \p begin and \p end
 */
template <typename T>
constexpr _CCCL_HOST_DEVICE T MidPoint(T begin, T end)
{
  return begin + (end - begin) / 2;
}

CUB_NAMESPACE_END
