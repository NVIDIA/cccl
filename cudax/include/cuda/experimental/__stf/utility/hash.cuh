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
 * @brief Implements hash_combine which updates a hash value by combining it
 *        with another value to form a new hash
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

#include <cuda/experimental/__stf/utility/unittest.cuh>

namespace cuda::experimental::stf
{

/**
 * @brief We define a hash trait class in our namespace
 */
template <typename T>
struct hash;

namespace reserved
{

/**
 * @brief Trait to check if std::hash<E> is defined.
 *
 * Primary template assumes std::hash<E> is not defined.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E, typename = void>
struct has_std_hash : ::std::false_type
{};

/**
 * @brief Specialization of has_std_hash for types where std::hash<E> is defined.
 *
 * Uses SFINAE to detect if std::hash<E> can be instantiated.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E>
struct has_std_hash<E, ::std::void_t<decltype(::std::declval<::std::hash<E>>()(::std::declval<E>()))>>
    : ::std::true_type
{};

/**
 * @brief Helper variable template to simplify usage of has_std_hash.
 *
 * Provides a convenient way to check if std::hash<E> is defined.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E>
inline constexpr bool has_std_hash_v = has_std_hash<E>::value;

} // end namespace reserved

/**
 * @brief Update a hash value by combining it with another value to form a new
 *        hash
 *
 * For some reason, C++ does not seem to provide this ...
 * Taken from WG21 P0814R0
 */
template <typename T>
void hash_combine(size_t& seed, const T& val)
{
  if constexpr (reserved::has_std_hash_v<T>)
  {
    // Use std::hash if it is specialized for T
    seed ^= ::std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  else
  {
    // Otherwise, use cuda::experimental::stf::hash
    seed ^= ::cuda::experimental::stf::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
}

//! Computes a combined hash value for one or more values.
//!
//! This function computes a hash value for a variable number of arguments.
//! - If called with a single argument, it uses `std::hash` if available for the type,
//!   otherwise it falls back to a custom hash implementation.
//! - If called with multiple arguments, it combines the hash values of all arguments
//!   using `hash_combine`.
//!
//! This utility is useful for creating composite hash values for tuples, structures, or
//! multiple parameters, and is compatible with both standard and custom hash functions.
//!
//! note:
//!   - At least one argument must be provided.
//!   - For single arguments, `std::hash` is preferred if available.
//!   - For multiple arguments, the order of arguments affects the result.
//!   - Requires `hash_combine` and `each_in_pack` utilities.
//!
//! \tparam Ts The types of the values to hash.
//! \param[in] vals The values to hash and combine.
//! \return The combined hash value as a `size_t`.
template <typename... Ts>
size_t hash_all(const Ts&... vals)
{
  if constexpr (sizeof...(Ts) == 1)
  {
    // Special case: single value, use std::hash if possible
    if constexpr (reserved::has_std_hash_v<Ts...>)
    {
      return ::std::hash<Ts...>()(vals...);
    }
    else
    {
      return ::cuda::experimental::stf::hash<Ts...>()(vals...);
    }
  }
  else
  {
    static_assert(sizeof...(Ts) != 0);
    size_t seed = 0;
    each_in_pack(
      [&](auto& val) {
        hash_combine(seed, val);
      },
      vals...);
    return seed;
  }
}

/**
 * @brief Specialization of `hash` for `std::pair<T1, T2>`.
 *
 * This hash specialization combines the individual hash values of the two elements in the pair to produce a unique hash
 * value for the pair.
 *
 * @tparam T1 The type of the first element in the pair.
 * @tparam T2 The type of the second element in the pair.
 */
template <class T1, class T2>
struct hash<::std::pair<T1, T2>>
{
  /**
   * @brief Computes a hash value for a given `std::pair`.
   *
   * This function applies a hash function to each element of the pair and combines
   * these hash values into a single hash value representing the pair.
   *
   * @param p The pair to hash.
   * @return size_t The hash value of the tuple.
   */
  size_t operator()(const ::std::pair<T1, T2>& p) const
  {
    return cuda::experimental::stf::hash_all(p.first, p.second);
  }
};

/**
 * @brief Specialization of hash for std::tuple.
 *
 * Provides a hash function for std::tuple, allowing tuples to be used
 * as keys in associative containers such as std::unordered_map or std::unordered_set.
 *
 * @tparam Ts Types of the elements in the tuple.
 */
template <typename... Ts>
struct hash<::std::tuple<Ts...>>
{
  /**
   * @brief Computes a hash value for a given std::tuple.
   *
   * This function applies a hash function to each element of the tuple and combines
   * these hash values into a single hash value representing the entire tuple.
   *
   * @param p The tuple to hash.
   * @return size_t The hash value of the tuple.
   */
  size_t operator()(const ::std::tuple<Ts...>& p) const
  {
    return ::std::apply(cuda::experimental::stf::hash_all<Ts...>, p);
  }
};

namespace reserved
{
/**
 * @brief Trait to check if std::hash<E> is defined.
 *
 * Primary template assumes std::hash<E> is not defined.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E, typename = void>
struct has_cudastf_hash : ::std::false_type
{};

/**
 * @brief Specialization of has_std_hash for types where std::hash<E> is defined.
 *
 * Uses SFINAE to detect if std::hash<E> can be instantiated.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E>
struct has_cudastf_hash<
  E,
  ::std::void_t<decltype(::std::declval<::cuda::experimental::stf::hash<E>>()(::std::declval<E>()))>> : ::std::true_type
{};

/**
 * @brief Helper variable template to simplify usage of has_std_hash.
 *
 * Provides a convenient way to check if std::hash<E> is defined.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E>
inline constexpr bool has_cudastf_hash_v = has_cudastf_hash<E>::value;

} // end namespace reserved

UNITTEST("hash for tuples")
{
  ::std::unordered_map<::std::tuple<int, int>, int, ::cuda::experimental::stf::hash<::std::tuple<int, int>>> m;
  m[::std::tuple(1, 2)] = 42;
};

} // end namespace cuda::experimental::stf
