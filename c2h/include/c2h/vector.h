// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <thrust/detail/vector_base.h>

#include <cuda/std/type_traits>

#include <memory>

#include <catch2/catch_tostring.hpp>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <c2h/checked_allocator.cuh>
#else
#  include <thrust/device_vector.h>
#  include <thrust/host_vector.h>
#endif

namespace c2h
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T>
using host_vector = THRUST_NS_QUALIFIER::detail::vector_base<T, c2h::checked_host_allocator<T>>;

template <typename T>
using device_vector = THRUST_NS_QUALIFIER::detail::vector_base<T, c2h::checked_cuda_allocator<T>>;
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using THRUST_NS_QUALIFIER::device_vector;
using THRUST_NS_QUALIFIER::host_vector;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
} // namespace c2h

// We specialize how Catch2 prints ([signed|unsigned]) char vectors for better readability. Let's print them as numbers
// instead of characters.
template <typename T, typename A>
struct Catch::StringMaker<THRUST_NS_QUALIFIER::detail::vector_base<T, A>,
                          ::cuda::std::enable_if_t<sizeof(T) == 1 && ::cuda::std::is_fundamental_v<T>>>
{
  // Copied from `rangeToString` in catch_tostring.hpp
  static auto convert(const THRUST_NS_QUALIFIER::detail::vector_base<T, A>& v) -> std::string
  {
    auto first = v.begin();
    auto last  = v.end();

    ReusableStringStream rss;
    rss << "{ ";
    if (first != last)
    {
      rss << Detail::stringify(static_cast<unsigned>(static_cast<T>(*first)));
      for (++first; first != last; ++first)
      {
        rss << ", " << Detail::stringify(static_cast<unsigned>(static_cast<T>(*first)));
      }
    }
    rss << " }";
    return rss.str();
  }
};

// due to an nvcc bug, the above specialization of StringMaker is ambiguous with one inside Catch2, so let's disable
// Catch2 range formatting for vector_base with sizeof(T) == 1 entirely
template <typename T, typename A>
struct Catch::is_range<THRUST_NS_QUALIFIER::detail::vector_base<T, A>>
{
  static constexpr bool value = !(sizeof(T) == 1 && ::cuda::std::is_fundamental_v<T>);
};
