// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * Thread utilities for sequential search
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

#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

#include <nv/target>

CUB_NAMESPACE_BEGIN

/**
 * Computes the begin offsets into A and B for the specific diagonal
 *
 * Deprecated [Since 3.0]
 */
template <typename AIteratorT, typename BIteratorT, typename OffsetT, typename CoordinateT>
CCCL_DEPRECATED _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void MergePathSearch(
  OffsetT diagonal, AIteratorT a, BIteratorT b, OffsetT a_len, OffsetT b_len, CoordinateT& path_coordinate)
{
  /// The value type of the input iterator
  using T = cub::detail::it_value_t<AIteratorT>;

  OffsetT split_min = ::cuda::std::max(diagonal - b_len, 0);
  OffsetT split_max = ::cuda::std::min(diagonal, a_len);

  while (split_min < split_max)
  {
    OffsetT split_pivot = (split_min + split_max) >> 1;
    if (a[split_pivot] <= b[diagonal - split_pivot - 1])
    {
      // Move candidate split range up A, down B
      split_min = split_pivot + 1;
    }
    else
    {
      // Move candidate split range up B, down A
      split_max = split_pivot;
    }
  }

  path_coordinate.x = ::cuda::std::min(split_min, a_len);
  path_coordinate.y = diagonal - split_min;
}

/**
 * @brief Returns the offset of the first value within @p input which does not compare
 *        less than @p val
 *
 * @param[in] input
 *   Input sequence
 *
 * @param[in] num_items
 *   Input sequence length
 *
 * @param[in] val
 *   Search key
 */
// TODO(bgruber): deprecate once ::cuda::std::lower_bound is made public
template <typename InputIteratorT, typename OffsetT, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT LowerBound(InputIteratorT input, OffsetT num_items, T val)
{
  OffsetT retval = 0;
  while (num_items > 0)
  {
    OffsetT half = num_items >> 1;
    if (input[retval + half] < val)
    {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    }
    else
    {
      num_items = half;
    }
  }

  return retval;
}

/**
 * @brief Returns the offset of the first value within @p input which compares
 *        greater than @p val
 *
 * @param[in] input
 *   Input sequence
 *
 * @param[in] num_items
 *   Input sequence length
 *
 * @param[in] val
 *   Search key
 */
// TODO(bgruber): deprecate once ::cuda::std::upper_bound is made public
template <typename InputIteratorT, typename OffsetT, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT UpperBound(InputIteratorT input, OffsetT num_items, T val)
{
  OffsetT retval = 0;
  while (num_items > 0)
  {
    OffsetT half = num_items >> 1;
    if (val < input[retval + half])
    {
      num_items = half;
    }
    else
    {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    }
  }

  return retval;
}

#if _CCCL_HAS_NVFP16()
/**
 * @param[in] input
 *   Input sequence
 *
 * @param[in] num_items
 *   Input sequence length
 *
 * @param[in] val
 *   Search key
 */
template <typename InputIteratorT, typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT UpperBound(InputIteratorT input, OffsetT num_items, __half val)
{
  OffsetT retval = 0;
  while (num_items > 0)
  {
    OffsetT half = num_items >> 1;

    bool lt;
    NV_IF_TARGET(NV_PROVIDES_SM_53,
                 (lt = __hlt(val, input[retval + half]);),
                 (lt = __half2float(val) < __half2float(input[retval + half]);));

    if (lt)
    {
      num_items = half;
    }
    else
    {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    }
  }

  return retval;
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
/**
 * @param[in] input
 *   Input sequence
 *
 * @param[in] num_items
 *   Input sequence length
 *
 * @param[in] val
 *   Search key
 */
template <typename InputIteratorT, typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT UpperBound(InputIteratorT input, OffsetT num_items, __nv_bfloat16 val)
{
  OffsetT retval = 0;
  while (num_items > 0)
  {
    OffsetT half = num_items >> 1;

    bool lt;
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (lt = __hlt(val, input[retval + half]);),
                 (lt = __bfloat162float(val) < __bfloat162float(input[retval + half]);));

    if (lt)
    {
      num_items = half;
    }
    else
    {
      retval    = retval + (half + 1);
      num_items = num_items - (half + 1);
    }
  }

  return retval;
}
#endif // _CCCL_HAS_NVBF16()

CUB_NAMESPACE_END
