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

#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>
#include <cuda/experimental/__utility/optionally_static.cuh>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cuda::experimental::stf
{
// Hack setenv on Windows
#if _CCCL_COMPILER(MSVC)
/**
 * @brief Sets an environment variable, mimicking the behavior of `std::setenv` on Windows.
 *
 * This function attempts to set the value of the environment variable `name` to `value`.
 * If `overwrite` is 0 and the variable already exists, the function does nothing.
 *
 * @param name The name of the environment variable.
 * @param value The value to assign to the environment variable.
 * @param overwrite If non-zero, the function will overwrite the existing value of the variable.
 * @return 0 on success, or -1 on failure (invalid input or memory allocation failure).
 * @note This function is designed for MSVC, which lacks a standard `setenv` function.
 */
inline int setenv(const char* name, const char* value, int overwrite)
{
  if (!name || !value || !name[0])
  {
    // Invalid input: name or value is null, or name is an empty string
    return -1;
  }

  // Check if the variable already exists and if overwrite is allowed
  if (!overwrite && ::std::getenv(name) != nullptr)
  {
    return 0; // Variable exists, and we're not allowed to overwrite it
  }

  // Construct the string in the form "NAME=VALUE"
  auto env_var = ::std::string(name) + "=" + value;

  // Use _putenv to set the environment variable in MSVC
  if (_putenv(env_var.c_str()) != 0)
  {
    return -1; // _putenv failed
  }

  return 0; // Success
}
#endif

#ifndef _CCCL_DOXYGEN_INVOKED // FIXME Doxygen is lost with decltype(auto)
/**
 * @brief Custom move function that performs checks on the argument type.
 *
 * @tparam T Type of the object being moved. The type should satisfy certain conditions for the move to be performed.
 * @param obj The object to be moved.
 * @return The moved object, ready to be passed to another owner.
 *
 * @pre The argument `obj` must be an lvalue, i.e., the function will fail to compile for rvalues.
 * @pre The argument `obj` must not be `const`, i.e., the function will fail to compile for `const` lvalues.
 */
template <typename T>
_CCCL_HOST_DEVICE constexpr decltype(auto) mv(T&& obj)
{
  static_assert(::std::is_lvalue_reference_v<T>, "Useless move from rvalue.");
  static_assert(!::std::is_const_v<::std::remove_reference_t<T>>, "Misleading move from const lvalue.");
  return ::std::move(obj);
}
#endif // _CCCL_DOXYGEN_INVOKED

/**
 * @brief Creates a `std::shared_ptr` managing a copy of the given object.
 *
 * This function takes an object of any type and returns a `std::shared_ptr`
 * that manages a copy of that object. If the object is an lvalue reference,
 * it will be copied into the `shared_ptr`. If the object is an rvalue reference,
 * it will be moved into the `shared_ptr`.
 *
 * The type managed by the `shared_ptr` has all references and `const`/`volatile`
 * qualifiers removed from the original type.
 *
 * @tparam T The type of the object, deduced automatically. May be an lvalue or rvalue reference.
 * @param obj The object to copy into the instance managed by the `shared_ptr`.
 * @return A `std::shared_ptr` managing a new copy of the object.
 *
 * @note This function simplifies the creation of `std::shared_ptr`s by handling
 * the type deduction and appropriate forwarding of the object. It's particularly
 * useful when you want to create a `shared_ptr` from temporary objects or when
 * the object's type includes references or cv-qualifiers.
 *
 * @code
 * int value = 42;
 * auto sp1 = to_shared(value);            // New shared_ptr<int>
 * assert(*sp1 == 42);                     // sp1 points to an int valued at 42
 * @endcode
 */
template <typename T>
auto to_shared(T&& obj)
{
  return ::std::make_shared<::std::remove_cv_t<::std::remove_reference_t<T>>>(::std::forward<T>(obj));
}

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
  using common = ::std::remove_reference_t<decltype(true ? from : to)>;

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
      if constexpr (::std::is_enum_v<common>)
      {
        value = static_cast<T>(static_cast<::std::underlying_type_t<T>>(value) + 1);
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
  static_assert(!::std::is_pointer_v<T>, "Use the two arguments version of each() with pointers.");
  if constexpr (::std::is_signed_v<T>)
  {
    _CCCL_ASSERT(to >= 0, "Attempt to iterate from 0 to a negative value.");
  }
  return each(T(0), mv(to));
}

/**
 * @brief Applies a callable object `f` to each integral constant within a given range `[0, n)`.
 *
 * This function template takes a callable object `f` and applies it to each integral constant
 * in the range `[0, n)`. The callable object is expected to take a single argument of type
 * `std::integral_constant<size_t, i>` (or `size_t`), where `i` is the current index.
 *
 * The important element is that the lambda can use its integral argument during compilation, e.g.
 * to fetch a tuple element with `std::get<i>(t)`.
 *
 * @tparam n The number of times the callable object `f` should be applied.
 * @tparam F Type of the callable object.
 * @tparam i... (Internal) Indices for parameter pack expansion.
 * @param f The callable object to apply to each integral constant.
 *
 * Example usage:
 * @code
 * auto print_index = [](auto index) { ::std::cout << index << ' '; };
 * unroll<5>(print_index); // Output: 0 1 2 3 4
 * @endcode
 *
 * Note: Since this function is `constexpr`, it can be used at compile-time if `f` is a
 * compile-time invocable object.
 */
template <size_t n, typename F, size_t... i>
constexpr void unroll(F&& f, ::std::index_sequence<i...> = {})
{
  if constexpr (sizeof...(i) != n)
  {
    return unroll<n>(::std::forward<F>(f), ::std::make_index_sequence<n>());
  }
  else
  {
    using result_t = decltype(f(::std::integral_constant<size_t, 0>()));
    if constexpr (::std::is_same_v<result_t, void>)
    {
      (f(::std::integral_constant<size_t, i>()), ...);
    }
    else
    {
      (f(::std::integral_constant<size_t, i>()) && ...);
    }
  }
}

/**
 * @brief Prepends an element to a tuple.
 *
 * This function creates a new tuple by prepending the element `t` to the tuple `p`.
 *
 * @tparam T The type of the element to prepend.
 * @tparam P The types of the elements in the tuple.
 * @param prefix The element to prepend.
 * @param tuple The tuple to which the element is prepended.
 *
 * @return std::tuple<T, P...> A new tuple with `t` prepended to `p`.
 *
 * @par Example:
 * @code
 *  int a = 1;
 *  std::tuple<int, double, char> t = std::make_tuple(2, 3.0, 'c');
 *  auto result = tuple_prepend(a, t);
 *  // result is std::tuple<int, int, double, char>(1, 2, 3.0, 'c')
 * @endcode
 */
template <typename T, typename... P>
constexpr auto tuple_prepend(T&& prefix, ::std::tuple<P...> tuple)
{
  return ::std::apply(
    [&](auto&&... p) {
      return ::std::tuple(::std::forward<T>(prefix), ::std::forward<decltype(p)>(p)...);
    },
    mv(tuple));
}

namespace reserved
{
// Like ::std::make_tuple, but skips all values of the same type as `::std::ignore`.
inline constexpr auto make_tuple()
{
  return ::std::tuple<>();
}

template <typename T, typename... P>
constexpr auto make_tuple([[maybe_unused]] T t, P... p)
{
  if constexpr (::std::is_same_v<const T, const decltype(::std::ignore)>)
  {
    // Recurse skipping the first parameter
    return make_tuple(mv(p)...);
  }
  else
  {
    // Keep first parameter, concatenate with recursive call
    return tuple_prepend(mv(t), make_tuple(mv(p)...));
  }
}
} // namespace reserved

/**
 * @brief Creates a `std::tuple` by applying a callable object `f` to each integral constant within a given range `[0,
 * n)`.
 *
 * This function template takes a callable object `f` and applies it to each integral constant in the range `[0, n)`.
 * The results of the calls are collected into a `std::tuple` and returned. The callable object is expected to take a
 * single argument of type `std::integral_constant<size_t, i>`, where `i` is the current index, and return a value of
 * the desired type and value.
 *
 * If `f` returns `std::ignore` for any argument(s), the corresponding value(s) will be skipped in the resulting
 * tuple.
 *
 * @tparam n The number of times the callable object `f` should be applied.
 * @tparam F Type of the callable object.
 * @tparam i... (Internal) Indices for parameter pack expansion.
 * @param f The callable object to apply to each integral constant.
 * @return A `std::tuple` containing the results of applying `f` to each integral constant in the range `[0, n)`.
 *
 * Example usage:
 * @code
 * auto make_double = [](auto index) {
 *     if constexpr (index == 2)
 *         return std::ignore;
 *     else
 *         return static_cast<double>(index);
 * };
 * auto result = make_tuple_indexwise<5>(make_double);
 * // result is std::tuple<double, double, double, double>{0.0, 1.0, 3.0, 4.0}
 * @endcode
 *
 * Note: Since this function is `constexpr`, it can be used at compile-time if `f` is a compile-time invocable object.
 */
template <size_t n, typename F, size_t... i>
constexpr auto make_tuple_indexwise(F&& f, ::std::index_sequence<i...> = {})
{
  if constexpr (sizeof...(i) != n)
  {
    return make_tuple_indexwise<n>(::std::forward<F>(f), ::std::make_index_sequence<n>());
  }
  else
  {
    return reserved::make_tuple(f(::std::integral_constant<size_t, i>())...);
  }
}

/**
 * @brief Applies a transformation to each element of a tuple, optionally passing the index.
 *
 * Iterates over the elements of a tuple and applies the provided functor `f` to each element.
 * If `f` is invocable with both the index and the element (`f(index, element)`), that form is used.
 * Otherwise, it falls back to `f(element)`. The index is a compile-time constant of type
 * `std::integral_constant<std::size_t, i>`, where `i` is the index of the element in the tuple.
 *
 * @tparam Tuple The type of the input tuple.
 * @tparam F The type of the transformation functor.
 * @param t The input tuple to transform.
 * @param f The transformation functor. Must be invocable with either `f(element)` or `f(index, element)`.
 * @return A new tuple containing the results of applying `f` to each element of `t`.
 */
template <typename Tuple, typename F>
constexpr auto tuple_transform(Tuple&& t, F&& f)
{
  constexpr size_t n = ::std::tuple_size_v<::std::remove_reference_t<Tuple>>;
  return make_tuple_indexwise<n>([&](auto j) {
    if constexpr (::std::is_invocable_v<F, decltype(j), decltype(::std::get<j>(::std::forward<Tuple>(t)))>)
    {
      return f(j, ::std::get<j>(::std::forward<Tuple>(t)));
    }
    else
    {
      return f(::std::get<j>(::std::forward<Tuple>(t)));
    }
  });
}

/**
 * @brief Iterates over the elements of a tuple, applying a function to each element.
 *
 * @details
 * This function traverses the elements of a tuple at compile time and applies the provided
 * callable `f` to each element. If the callable can be invoked with both the element's index
 * and value (`f(index, element)`), both are passed; otherwise, only the element is passed (`f(element)`).
 *
 * The iteration is performed using compile-time unrolling, making this utility suitable for
 * constexpr and template metaprogramming scenarios.
 *
 * @tparam Tuple The type of the tuple to iterate over. May be an lvalue or rvalue reference.
 * @tparam F The type of the callable to apply. Must be invocable with either `(index, element)` or `(element)`.
 *
 * @param[in] t The tuple whose elements will be visited.
 * @param[in] f The function or functor to apply to each element (and optionally its index).
 *
 * @note
 *   - The index is provided as a compile-time constant (such as a `std::integral_constant`).
 *   - The function is `constexpr` and can be used in constant expressions.
 *   - This utility requires an `unroll<N>(lambda)` facility for compile-time iteration.
 */
template <typename Tuple, typename F>
constexpr void each_in_tuple(Tuple&& t, F&& f)
{
  constexpr size_t n = ::std::tuple_size_v<::std::remove_reference_t<Tuple>>;
  unroll<n>([&](auto j) {
    if constexpr (::std::is_invocable_v<F, decltype(j), decltype(::std::get<j>(::std::forward<Tuple>(t)))>)
    {
      f(j, ::std::get<j>(::std::forward<Tuple>(t)));
    }
    else
    {
      f(::std::get<j>(::std::forward<Tuple>(t)));
    }
  });
}

namespace reserved
{
// Implementation of each_in_pack below
template <typename F, size_t... i, typename... P>
constexpr void each_in_pack(F&& f, ::std::index_sequence<i...>, P&&... p)
{
  if constexpr (::std::is_invocable_v<F,
                                      ::std::integral_constant<size_t, 0>,
                                      ::std::tuple_element_t<0, ::std::tuple<P&&...>>>)
  {
    (f(::std::integral_constant<size_t, i>(), ::std::forward<P>(p)), ...);
  }
  else
  {
    (f(::std::forward<P>(p)), ...);
  }
}
} // namespace reserved

/**
 * @brief Applies a given function to each element in a parameter pack.
 *
 * If the callable object `f` accepts two parameters, `each_in_pack` calls `f(std::integral_constant<size_t, i>(),
 * std::forward<P>(p))` for each `p` at position `i` in the parameter pack. Otherwise, `each_in_pack` calls
 * `f(std::forward<P>(p))` for each `p` in the parameter pack.
 *
 * @tparam F Callable object type
 * @tparam P Variadic template parameter pack type
 * @param f Callable object to apply to each element in the parameter pack
 * @param p Variadic parameter pack
 *
 * \code{.cpp}
 * std::string s;
 * cuda::experimental::stf::each_in_pack([&](auto&& p) { s += std::to_string(p) + ", "; }, 1, 2, 3, 4, 5);
 * assert(s == "1, 2, 3, 4, 5, ");
 * s = "";
 * cuda::experimental::stf::each_in_pack([&](auto i, auto&& p) { s += (i ? ", " : "") + std::to_string(i) + p; }, "a",
 * "b", "c"); assert(s == "0a, 1b, 2c");
 * \endcode
 */
template <typename F, typename... P>
constexpr void each_in_pack(F&& f, P&&... p)
{
  if constexpr (sizeof...(P) > 0)
  {
    reserved::each_in_pack(::std::forward<F>(f), ::std::make_index_sequence<sizeof...(P)>(), ::std::forward<P>(p)...);
  }
}

/**
 * @brief Casts an enum value to its underlying type.
 *
 * This function template takes an enum value and returns its representation
 * in the underlying integral type of the enum.
 *
 * @tparam E The type of the enum.
 * @param value The enum value to be cast to its underlying type.
 * @return The underlying integral value of the given enum value.
 */
template <typename E>
auto as_underlying(E value)
{
  return static_cast<::std::underlying_type_t<E>>(value);
}

using ::cuda::experimental::optionally_static;

} // namespace cuda::experimental::stf
