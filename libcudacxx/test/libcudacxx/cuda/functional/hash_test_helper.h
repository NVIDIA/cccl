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
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
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

    cuda::hash<Key, Algorithm> hasher(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);

    cuda::std::array<Key, 1> arr_keys = {key};
    auto keys_span                    = cuda::std::span<Key, 1>{arr_keys};
    auto const_keys_span              = cuda::std::span<const Key, 1>{arr_keys};

    static_assert(cuda::std::is_same_v<decltype(hasher(key)), result_type>);
    static_assert(cuda::std::is_same_v<decltype(hasher(keys_span)), result_type>);
    static_assert(cuda::std::is_same_v<decltype(hasher(const_keys_span)), result_type>);
    static_assert(
      noexcept(cuda::hash<Key, Algorithm>(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...)));
    static_assert(noexcept(hasher(key)));
    static_assert(noexcept(hasher(keys_span)));
    static_assert(noexcept(hasher(const_keys_span)));

    assert(hasher(key) == expected);
    assert(hasher(keys_span) == expected);
    assert(hasher(const_keys_span) == expected);
  }
};

struct noncopyable_key
{
  TEST_FUNC constexpr explicit noncopyable_key(cuda::std::int32_t value)
      : value_{value}
  {}

  noncopyable_key(const noncopyable_key&) = delete;
  noncopyable_key(noncopyable_key&&)      = default;

  cuda::std::int32_t value_;
};

static_assert(cuda::std::is_trivially_copyable_v<noncopyable_key>);

template <cuda::hash_algorithm Algorithm>
TEST_FUNC void test_noncopyable_key()
{
  noncopyable_key key{42};
  cuda::hash<noncopyable_key, Algorithm> hasher;
  assert(hasher(key) == hasher(cuda::std::span<const noncopyable_key, 1>{&key, 1}));
}

template <cuda::std::size_t Size>
struct byte_key
{
  cuda::std::byte data_[Size];
};

template <cuda::hash_algorithm Algorithm, cuda::std::size_t Size>
TEST_FUNC void test_sized_key()
{
  byte_key<Size> key{};
  for (cuda::std::size_t i = 0; i < Size; ++i)
  {
    key.data_[i] = static_cast<cuda::std::byte>(i);
  }

  cuda::hash<byte_key<Size>, Algorithm> hasher;
  cuda::std::array<byte_key<Size>, 2> keys = {key, key};
  keys[1].data_[0]                         = static_cast<cuda::std::byte>(42);

  assert(hasher(key) == hasher(cuda::std::span<const byte_key<Size>, 1>{&key, 1}));
  assert(hasher(cuda::std::span<byte_key<Size>, 2>{keys}) == hasher(cuda::std::span<const byte_key<Size>, 2>{keys}));
}

template <cuda::hash_algorithm Algorithm>
TEST_FUNC void test_sized_keys()
{
  test_sized_key<Algorithm, 2>();
  test_sized_key<Algorithm, 3>();
  test_sized_key<Algorithm, 5>();
  test_sized_key<Algorithm, 6>();
  test_sized_key<Algorithm, 7>();
  test_sized_key<Algorithm, 9>();
  test_sized_key<Algorithm, 15>();
  test_sized_key<Algorithm, 17>();
}

template <cuda::hash_algorithm Algorithm, class Result>
TEST_FUNC void test_empty_span(Result expected)
{
  cuda::std::span<const cuda::std::uint32_t> empty;
  assert((cuda::hash<cuda::std::uint32_t, Algorithm>{}(empty) == expected));
}

#endif // TEST_LIBCUDACXX_CUDA_FUNCTIONAL_HASH_TEST_HELPER_H
