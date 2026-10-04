//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCUDACXX_CUDA_FUNCTIONAL_HASH_TEST_HELPER_H
#define TEST_LIBCUDACXX_CUDA_FUNCTIONAL_HASH_TEST_HELPER_H

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <cuda::std::int32_t Words>
struct large_key
{
  TEST_FUNC constexpr large_key(cuda::std::int32_t value)
  {
    for (cuda::std::int32_t i = 0; i < Words; ++i)
    {
      data_[i] = value;
    }
  }

private:
  cuda::std::int32_t data_[Words];
};

template <cuda::hash_algorithm Algorithm>
struct hash_result;

template <>
struct hash_result<cuda::hash_algorithm::xxhash_32>
{
  using type = cuda::std::uint32_t;
};

template <>
struct hash_result<cuda::hash_algorithm::xxhash_64>
{
  using type = cuda::std::uint64_t;
};

template <>
struct hash_result<cuda::hash_algorithm::murmurhash3_32>
{
  using type = cuda::std::uint32_t;
};

#if _CCCL_HAS_INT128()
template <>
struct hash_result<cuda::hash_algorithm::murmurhash3_x86_128>
{
  using type = __uint128_t;
};

template <>
struct hash_result<cuda::hash_algorithm::murmurhash3_x64_128>
{
  using type = __uint128_t;
};
#endif // _CCCL_HAS_INT128()

template <cuda::hash_algorithm Algorithm>
struct hash_test
{
  template <typename Key, typename ResultT, typename... HashConstructorArgs>
  TEST_FUNC void operator()(const Key& key, ResultT expected, HashConstructorArgs&&... hash_constructor_args)
  {
    using result_type = typename hash_result<Algorithm>::type;

    cuda::hash<Key, Algorithm> hasher(cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);
    cuda::std::array<Key, 1> keys = {key};
    auto keys_span                = cuda::std::span<Key>{keys.data(), keys.size()};

    static_assert(cuda::std::is_same_v<decltype(hasher(key)), result_type>);
    static_assert(cuda::std::is_same_v<decltype(hasher(keys_span)), result_type>);

    assert(hasher(key) == expected);
    assert(hasher(keys_span) == expected);
  }
};

#endif // TEST_LIBCUDACXX_CUDA_FUNCTIONAL_HASH_TEST_HELPER_H
