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
 *
 * @brief This defines a void data interface useful to implement STF
 * dependencies without actual data (e.g. to enforce task dependencies)
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

#include <cuda/experimental/__stf/utility/hash.cuh>

namespace cuda::experimental::stf
{

template <typename T>
class shape_of;

class void_interface
{};

/**
 * @brief defines the shape of a void interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends shape_of
 */
template <>
class shape_of<void_interface>
{
public:
  shape_of()                = default;
  shape_of(const shape_of&) = default;
  shape_of(const void_interface&)
      : shape_of<void_interface>()
  {}

  /// Mandatory method : defined the total number of elements in the shape
  size_t size() const
  {
    return 0;
  }
};

/**
 * @brief A hash of the matrix
 */
template <>
struct hash<void_interface>
{
  ::std::size_t operator()(void_interface const&) const noexcept
  {
    return 42;
  }
};

namespace reserved
{

template <typename... Ts>
struct remove_void_interface
{
  using type = ::std::tuple<>;
};

template <typename T, typename... Ts>
struct remove_void_interface<T, Ts...>
{
private:
  using tail = typename remove_void_interface<Ts...>::type;

  // If T is void_interface, skip it, otherwise prepend it to tail
  using filtered =
    std::conditional_t<::std::is_same_v<T, void_interface>,
                       tail,
                       decltype(::std::tuple_cat(::std::declval<::std::tuple<T>>(), ::std::declval<tail>()))>;

public:
  using type = filtered;
};

template <typename... Ts>
using remove_void_interface_t = typename remove_void_interface<Ts...>::type;

template <typename T>
struct remove_void_interface_from_tuple
{
  // By default, if T is not a std::tuple, do nothing special
  using type = T;
};

template <typename... Ts>
struct remove_void_interface_from_tuple<::std::tuple<Ts...>>
{
  using type = remove_void_interface_t<Ts...>;
};

template <typename T>
using remove_void_interface_from_tuple_t = typename remove_void_interface_from_tuple<T>::type;

/**
 * @brief Check if a function can be invoked while eliding arguments with a void_interface type.
 */
template <typename Fun, typename... Data>
struct is_invocable_with_filtered
{
private:
  template <typename F, typename... Args>
  static auto test(int) -> ::std::bool_constant<::std::is_invocable_v<F, Args...>>
  {
    return {};
  }

  template <typename F>
  static auto test(...) -> ::std::false_type
  {
    return {};
  }

  template <::std::size_t... Idx>
  static auto check(::std::index_sequence<Idx...>)
  {
    using filtered = remove_void_interface_t<Data...>;
    return test<Fun, ::std::tuple_element_t<Idx, filtered>...>(0);
  }

public:
  static constexpr bool value =
    decltype(check(::std::make_index_sequence<::std::tuple_size_v<remove_void_interface_t<Data...>>>{}))::value;
};

/**
 * @brief Check if a function can be invoked using std::apply while eliding tuple arguments with a void_interface type.
 */
template <typename F, typename Tuple>
struct is_tuple_invocable_with_filtered : is_tuple_invocable<F, remove_void_interface_from_tuple_t<Tuple>>
{};

/**
 * @brief Strip tuple entries with a "void_interface" type
 */
template <typename... Ts>
auto remove_void_interface_types(const ::std::tuple<Ts...>& tpl)
{
  return ::std::apply(
    [](auto&&... args) {
      auto filter_one = [](auto&& arg) {
        using T = ::std::decay_t<decltype(arg)>;
        if constexpr (::std::is_same_v<T, void_interface>)
        {
          return ::std::tuple<>{};
        }
        else
        {
          return ::std::tuple<T>(::std::forward<decltype(arg)>(arg));
        }
      };
      return ::std::tuple_cat(filter_one(::std::forward<decltype(args)>(args))...);
    },
    tpl);
}

} // end namespace reserved

} // end namespace cuda::experimental::stf
