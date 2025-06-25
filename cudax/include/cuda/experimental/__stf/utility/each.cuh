//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 * @brief Widely used artifacts used by most of the library.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/type_traits>
#include <cuda/experimental/__stf/utility/mv.cuh>

namespace cuda::experimental::stf
{

/**
 * @brief   Create an iterable range from 'from' to 'to'
 *
 * @tparam  T   The type of the start range value
 * @tparam  U   The type of the end range value
 * @param   from    The start value of the range
 * @param   to      The end value of the range
 *
 * @return  A range of values from 'from' to 'to'
 *
 * @note    The range includes 'from' and excludes 'to'. The actual type iterated is determined as the type of the
 * expression `true ? from : to`. This ensures expected behavior for iteration with different `from` and `to` types.
 */
template <typename T, typename U>
_CCCL_HOST_DEVICE auto each(T from, U to)
{
  using common = ::cuda::std::remove_reference_t<decltype(true ? from : to)>;

  class iterator
  {
    common value;

  public:
    _CCCL_HOST_DEVICE iterator(common value)
        : value(mv(value))
    {}

    _CCCL_HOST_DEVICE common operator*() const
    {
      return value;
    }

    _CCCL_HOST_DEVICE iterator& operator++()
    {
      if constexpr (::cuda::std::is_enum_v<common>)
      {
        value = static_cast<T>(static_cast<::cuda::std::underlying_type_t<T>>(value) + 1);
      }
      else
      {
        ++value;
      }
      return *this;
    }

    _CCCL_HOST_DEVICE bool operator!=(const iterator& other) const
    {
      return value != other.value;
    }
  };

  class each_t
  {
    common begin_, end_;

  public:
    _CCCL_HOST_DEVICE each_t(T begin, U end)
        : begin_(mv(begin))
        , end_(mv(end))
    {}
    _CCCL_HOST_DEVICE iterator begin() const
    {
      return iterator(begin_);
    }
    _CCCL_HOST_DEVICE iterator end() const
    {
      return iterator(end_);
    }
  };

  return each_t{mv(from), mv(to)};
}

/**
 * @brief   Create an iterable range from `T(0)` to `to`
 *
 * @tparam  T   The type of the end range value
 * @param   to   The end value of the range
 *
 * @return  A range of values from `T(0)` to `to`
 *
 * @note    The range includes 0 and excludes `to`
 */
template <typename T>
auto each(T to)
{
  static_assert(!::cuda::std::is_pointer_v<T>, "Use the two arguments version of each() with pointers.");
  if constexpr (::cuda::std::is_signed_v<T>)
  {
    _CCCL_ASSERT(to >= 0, "Attempt to iterate from 0 to a negative value.");
  }
  return each(T(0), mv(to));
}

} // namespace cuda::experimental::stf
