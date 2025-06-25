//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Utilities related to trait classes
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

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

namespace cuda::experimental::stf
{

/**
 * @brief Converts each element in `t` to a new value by calling `f`, then returns a tuple collecting the values thus
 * obtained.
 *
 * @tparam Tuple Type of the tuple to convert
 * @tparam Fun Type of mapping function to apply
 * @param t Object to convert, must support `std::apply`
 * @param f function to convert each element of the tuple, must take a single parameter
 * @return constexpr auto The tuple resulting from the mapping
 *
 * @paragraph example Example
 * @snippet unittest.h tuple2tuple
 */
template <typename Tuple, typename Fun>
constexpr auto tuple2tuple(const Tuple& t, Fun&& f)
{
  return ::cuda::std::apply(
    [&](auto&&... x) {
      return ::cuda::std::tuple(f(::std::forward<decltype(x)>(x))...);
    },
    t);
}



/**
 * @brief Converts an array-like object (such as an `std::array`) to an `std::tuple`.
 *
 * This function template takes an array-like object and returns a tuple containing the same elements.
 * If the input array has a size of zero, an empty tuple is returned.
 *
 * @tparam Array Type of the array-like object. Must have `std::tuple_size_v<Array>` specialization.
 * @param array The array-like object to be converted to a tuple.
 * @return A tuple containing the elements of the input array.
 *
 * Example usage:
 * @code
 * std::array<int, 3> arr = {1, 2, 3};
 * auto t = to_tuple(arr); // t is a std::tuple<int, int, int>
 * @endcode
 */
template <typename Array>
auto to_tuple(Array&& array)
{
  return tuple2tuple(::std::forward<Array>(array), [](auto&& e) {
    return ::std::forward<decltype(e)>(e);
  });
}

/**
 * @brief Array-like tuple with a single element type repeated `n` times.
 *
 * The `array_tuple` template generates a `std::tuple` with a single type `T` repeated `n` times.
 * This can be used to create a tuple with consistent types and a fixed size.
 *
 * @tparam T The type of the elements that the tuple will contain.
 * @tparam n The number of elements that the tuple will contain.
 *
 * ### Example
 *
 * ```cpp
 * using my_tuple = array_tuple<int, 5>; // Results in std::tuple<int, int, int, int, int>
 * ```
 *
 * @note The specialization `array_tuple<T, 0>` will result in an empty tuple (`std::tuple<>`).
 */
template <typename T, size_t n>
using array_tuple = decltype(to_tuple(::cuda::std::array<T, n>{}));

#ifndef __CUDACC_RTC__
// Mini-unittest
static_assert(::cuda::std::is_same_v<array_tuple<size_t, 3>, ::cuda::std::tuple<size_t, size_t, size_t>>);
#endif // __CUDACC_RTC__

} // end namespace cuda::experimental::stf

