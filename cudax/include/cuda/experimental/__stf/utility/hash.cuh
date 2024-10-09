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

#include <cuda/experimental/__stf/utility/unittest.cuh>

namespace cuda::experimental::stf {

/**
 * @brief Update a hash value by combining it with another value to form a new
 *        hash
 *
 * For some reason, C++ does not seem to provide this ...
 * Taken from WG21 P0814R0
 */
template <typename T>
void hash_combine(size_t& seed, const T& val) {
    seed ^= ::std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename... Ts>
size_t hash_all(const Ts&... vals) {
    if constexpr (sizeof...(Ts) == 1) {
        return ::std::hash<Ts...>()(vals...);
    } else {
        static_assert(sizeof...(Ts) != 0);
        size_t seed = 0;
        each_in_pack([&](auto& val) { hash_combine(seed, val); }, vals...);
        return seed;
    }
}

}  // end namespace cuda::experimental::stf

/**
 * @brief Specialization of `std::hash` for `std::pair<T1, T2>`.
 *
 * This hash specialization combines the individual hash values of the two elements in the pair to produce a unique hash
 * value for the pair.
 *
 * @tparam T1 The type of the first element in the pair.
 * @tparam T2 The type of the second element in the pair.
 */
template <class T1, class T2>
struct std::hash<::std::pair<T1, T2>> {
    /**
     * @brief Computes a hash value for a given `std::pair`.
     *
     * This function applies a hash function to each element of the pair and combines
     * these hash values into a single hash value representing the pair.
     *
     * @param p The pair to hash.
     * @return size_t The hash value of the tuple.
     */
    size_t operator()(const ::std::pair<T1, T2>& p) const {
        return cuda::experimental::stf::hash_all(p.first, p.second);
    }
};

/**
 * @brief Specialization of std::hash for std::tuple.
 *
 * Provides a hash function for std::tuple, allowing tuples to be used
 * as keys in associative containers such as std::unordered_map or std::unordered_set.
 *
 * @tparam Ts Types of the elements in the tuple.
 */
template <typename... Ts>
struct std::hash<::std::tuple<Ts...>> {
    /**
     * @brief Computes a hash value for a given std::tuple.
     *
     * This function applies a hash function to each element of the tuple and combines
     * these hash values into a single hash value representing the entire tuple.
     *
     * @param p The tuple to hash.
     * @return size_t The hash value of the tuple.
     */
    size_t operator()(const ::std::tuple<Ts...>& p) const {
        return ::std::apply(cuda::experimental::stf::hash_all<Ts...>, p);
    }
};

/**
 * @brief Trait to check if std::hash<E> is defined.
 *
 * Primary template assumes std::hash<E> is not defined.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E, typename = void>
struct has_std_hash : ::std::false_type {};

/**
 * @brief Specialization of has_std_hash for types where std::hash<E> is defined.
 *
 * Uses SFINAE to detect if std::hash<E> can be instantiated.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E>
struct has_std_hash<E, ::std::void_t<decltype(::std::declval<::std::hash<E>>()(::std::declval<E>()))>>
        : ::std::true_type {};

/**
 * @brief Helper variable template to simplify usage of has_std_hash.
 *
 * Provides a convenient way to check if std::hash<E> is defined.
 *
 * @tparam E The type to check for std::hash definition.
 */
template <typename E>
inline constexpr bool has_std_hash_v = has_std_hash<E>::value;

UNITTEST("hash for tuples") {
    ::std::unordered_map<::std::tuple<int, int>, int> m;
    m[::std::tuple(1, 2)] = 42;
};
