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
 * @brief Mechanism to ensure a function is only called once (for a set of arguments)
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

#include <unordered_map>

namespace cuda::experimental::stf
{
/**
 * @brief A class that ensures a function is executed only once for a given set of arguments (including none).
 *
 * The `run_once` class is a utility that caches the result of a function call based on its arguments,
 * ensuring that the function is only run once for each unique set of arguments.
 * It is particularly useful in scenarios where a function call is expensive and
 * the result is invariant for the same input parameters.
 *
 * @tparam Ts Types of the arguments used to uniquely identify the function call.
 */
template <typename... Ts>
class run_once
{
public:
  /**
   * @brief Constructs the `run_once` object with given arguments.
   *
   * These arguments are used to uniquely identify the function call.
   *
   * @param val Arguments that are used to determine if the function has been run before.
   */
  run_once(Ts... val)
      : val(mv(val)...) {};

  /**
   * @brief Invokes the function if it has not been run with the stored arguments before.
   *
   * If the function has been run before with the same arguments, returns the cached result.
   *
   * @tparam Fun The type of the function to be executed.
   * @param fun The function to be executed.
   * @return The result of the function call. The type of the result is determined by the return type of the function.
   */
  template <typename Fun>
  auto& operator->*(Fun&& fun) &&
  {
    using ReturnType = std::invoke_result_t<Fun, Ts...>;

    // Static assertions to ensure ReturnType meets the requirements
    static_assert(
      std::is_constructible_v<ReturnType, decltype(std::invoke(std::forward<Fun>(fun), std::declval<Ts>()...))>,
      "ReturnType must be constructible from the result of fun()");
    static_assert(std::is_move_constructible_v<ReturnType>, "ReturnType must be MoveConstructible");
    static_assert(std::is_move_assignable_v<ReturnType>, "ReturnType must be MoveAssignable");

    if constexpr (sizeof...(Ts) == 0)
    {
      static auto result = fun();
      return result;
    }
    else
    {
      using ReturnType = ::std::invoke_result_t<decltype(fun), Ts...>;
      static ::std::unordered_map<::std::tuple<Ts...>, ReturnType, ::cuda::experimental::stf::hash<::std::tuple<Ts...>>>
        cache;

      if (auto it = cache.find(val); it != cache.end())
      {
        return it->second;
      }

      // We only set the cache AFTER the end of the computation
      auto result = ::std::apply(::std::forward<Fun&&>(fun), ::std::move(val));

      return cache[val] = mv(result);
    }
  }

private:
  // Stores the arguments used to invoke the function.
  [[no_unique_address]] ::std::tuple<Ts...> val;
};
} // end namespace cuda::experimental::stf
