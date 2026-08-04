//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvcc-12.0

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if _CCCL_CUDA_COMPILATION()
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILATION()

#include "test_macros.h"

template <cuda::std::int32_t Words>
struct large_key
{
  TEST_HOST_DEVICE_FUNC constexpr large_key(cuda::std::int32_t value) noexcept
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
  TEST_HOST_DEVICE_FUNC void
  operator()(const Key& key, ResultT expected, HashConstructorArgs&&... hash_constructor_args) noexcept
  {
    using result_type = typename hash_result<Algorithm>::type;

    const cuda::hash<Key, Algorithm> hasher(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);

    cuda::std::array<Key, 1> arr_keys             = {key};
    const cuda::std::array<Key, 1> const_arr_keys = {key};
    const auto keys_span                          = cuda::std::span<Key, 1>{arr_keys};
    const auto const_keys_span                    = cuda::std::span<const Key, 1>{const_arr_keys};

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

struct test_xxhash32
{
  hash_test<cuda::hash_algorithm::xxhash_32> xxhash32_test;

  TEST_HOST_DEVICE_FUNC void operator()()
  {
    xxhash32_test(static_cast<char>(0), 3479547966u, 0);
    xxhash32_test(static_cast<char>(42), 3774771295u, 0);
    xxhash32_test(static_cast<char>(0), 2099223482u, 42);
    xxhash32_test(static_cast<cuda::std::int32_t>(0), 148298089u, 0);
    xxhash32_test(static_cast<cuda::std::int32_t>(0), 2132181312u, 42);
    xxhash32_test(static_cast<cuda::std::int32_t>(42), 1161967057u, 0);
    xxhash32_test(static_cast<cuda::std::int32_t>(123456789), 2987034094u, 0);
    xxhash32_test(static_cast<cuda::std::int64_t>(0), 3736311059u, 0);
    xxhash32_test(static_cast<cuda::std::int64_t>(0), 1076387279u, 42);
    xxhash32_test(static_cast<cuda::std::int64_t>(42), 2332451213u, 0);
    xxhash32_test(static_cast<cuda::std::int64_t>(123456789), 1561711919u, 0);
#if _CCCL_HAS_INT128()
    xxhash32_test(static_cast<__int128_t>(123456789), 1846633701u, 0);
#endif
    xxhash32_test(large_key<32>(123456789), 3715432378u, 0);
  }
};

struct test_xxhash64
{
  hash_test<cuda::hash_algorithm::xxhash_64> xxhash64_test;

  TEST_HOST_DEVICE_FUNC void operator()()
  {
    xxhash64_test(static_cast<char>(0), 16804241149081757544ull, 0);
    xxhash64_test(static_cast<char>(42), 765293966243412708ull, 0);
    xxhash64_test(static_cast<char>(0), 9486749600008296231ull, 42);
    xxhash64_test(static_cast<cuda::std::int32_t>(0), 4246796580750024372ull, 0);
    xxhash64_test(static_cast<cuda::std::int32_t>(0), 3614696996920510707ull, 42);
    xxhash64_test(static_cast<cuda::std::int32_t>(42), 15516826743637085169ull, 0);
    xxhash64_test(static_cast<cuda::std::int32_t>(123456789), 9462334144942111946ull, 0);
    xxhash64_test(static_cast<cuda::std::int64_t>(0), 3803688792395291579ull, 0);
    xxhash64_test(static_cast<cuda::std::int64_t>(0), 13194218611613725804ull, 42);
    xxhash64_test(static_cast<cuda::std::int64_t>(42), 13066772586158965587ull, 0);
    xxhash64_test(static_cast<cuda::std::int64_t>(123456789), 14662639848940634189ull, 0);
#if _CCCL_HAS_INT128()
    xxhash64_test(static_cast<__int128_t>(123456789), 7986913354431084250ull, 0);
#endif
    xxhash64_test(large_key<32>(123456789), 2031761887105658523ull, 0);
  }
};

struct test_murmurhash3_32
{
  hash_test<cuda::hash_algorithm::murmurhash3_32> murmurhash3_32_test;

  TEST_HOST_DEVICE_FUNC void operator()()
  {
    murmurhash3_32_test(static_cast<char>(0), 1364076727u, 0);
    murmurhash3_32_test(static_cast<char>(42), 338914844u, 0);
    murmurhash3_32_test(static_cast<char>(0), 3712240066u, 42);
    murmurhash3_32_test(static_cast<cuda::std::int32_t>(0), 593689054u, 0);
    murmurhash3_32_test(static_cast<cuda::std::int32_t>(0), 933211791u, 42);
    murmurhash3_32_test(static_cast<cuda::std::int32_t>(42), 3160117731u, 0);
    murmurhash3_32_test(static_cast<cuda::std::int32_t>(123456789), 3206620847u, 0);
    murmurhash3_32_test(static_cast<cuda::std::int64_t>(0), 1669671676u, 0);
    murmurhash3_32_test(static_cast<cuda::std::int64_t>(0), 2624043101u, 42);
    murmurhash3_32_test(static_cast<cuda::std::int64_t>(42), 1871679806u, 0);
    murmurhash3_32_test(static_cast<cuda::std::int64_t>(123456789), 690028081u, 0);
#if _CCCL_HAS_INT128()
    murmurhash3_32_test(static_cast<__int128_t>(123456789), 2191144977u, 0);
#endif
    murmurhash3_32_test(large_key<32>(123456789), 2555553099u, 0);
  }
};

#if _CCCL_HAS_INT128()
struct test_murmurhash3_x86_128
{
  hash_test<cuda::hash_algorithm::murmurhash3_x86_128> murmurhash3_x86_128_test;

  TEST_HOST_DEVICE_FUNC __uint128_t conv(const cuda::std::array<cuda::std::uint32_t, 4>& arr) const
  {
    return cuda::std::bit_cast<__uint128_t>(arr);
  }

  TEST_HOST_DEVICE_FUNC void operator()()
  {
    murmurhash3_x86_128_test(cuda::std::int32_t(0), conv({3422973727u, 2656139328u, 2656139328u, 2656139328u}), 0);
    murmurhash3_x86_128_test(cuda::std::int32_t(9), conv({2808089785u, 314604614u, 314604614u, 314604614u}), 0);
    murmurhash3_x86_128_test(cuda::std::int32_t(42), conv({3611919118u, 1962256489u, 1962256489u, 1962256489u}), 0);
    murmurhash3_x86_128_test(cuda::std::int32_t(42), conv({3399017053u, 732469929u, 732469929u, 732469929u}), 42);

    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int32_t, 2>{2, 2}, conv({1234494082u, 1431451587u, 431049201u, 431049201u}), 0);
    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int32_t, 3>{1, 4, 9}, conv({2516796247u, 2757675829u, 778406919u, 2453259553u}), 42);
    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int32_t, 4>{42, 64, 108, 1024},
                             conv({2686265656u, 591236665u, 3797082165u, 2731908938u}),
                             63);
    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int32_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      conv({3918256832u, 4205523739u, 1707810111u, 1625952473u}),
      1024);

    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int64_t, 2>{2, 2}, conv({3811075945u, 727160712u, 3510740342u, 235225510u}), 0);
    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int64_t, 3>{1, 4, 9}, conv({2817194959u, 206796677u, 3391242768u, 248681098u}), 42);
    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int64_t, 4>{42, 64, 108, 1024},
                             conv({2335912146u, 1566515912u, 760710030u, 452077451u}),
                             63);
    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int64_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      conv({1101169764u, 1758958147u, 2406511780u, 2903571412u}),
      1024);
  }
};

struct test_murmurhash3_x64_128
{
  hash_test<cuda::hash_algorithm::murmurhash3_x64_128> murmurhash3_x64_128_test;

  TEST_HOST_DEVICE_FUNC __uint128_t conv(const cuda::std::array<cuda::std::uint64_t, 2>& arr) const
  {
    return cuda::std::bit_cast<__uint128_t>(arr);
  }

  TEST_HOST_DEVICE_FUNC void operator()()
  {
    murmurhash3_x64_128_test(cuda::std::int32_t(0), conv({14961230494313510588ull, 6383328099726337777ull}), 0);
    murmurhash3_x64_128_test(cuda::std::int32_t(9), conv({1779292183511753683ull, 16298496441448380334ull}), 0);
    murmurhash3_x64_128_test(cuda::std::int32_t(42), conv({2913627637088662735ull, 16344193523890567190ull}), 0);
    murmurhash3_x64_128_test(cuda::std::int32_t(42), conv({2248879576374326886ull, 18006515275339376488ull}), 42);

    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int32_t, 2>{2, 2}, conv({12221386834995143465ull, 6690950894782946573ull}), 0);
    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int32_t, 3>{1, 4, 9}, conv({299140022350411792ull, 9891903873182035274ull}), 42);
    murmurhash3_x64_128_test(cuda::std::array<cuda::std::int32_t, 4>{42, 64, 108, 1024},
                             conv({4333511168876981289ull, 4659486988434316416ull}),
                             63);
    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int32_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      conv({3302412811061286680ull, 7070355726356610672ull}),
      1024);

    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int64_t, 2>{2, 2}, conv({8554944597931919519ull, 14938998000509429729ull}), 0);
    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int64_t, 3>{1, 4, 9}, conv({13442629947720186435ull, 7061727494178573325ull}), 42);
    murmurhash3_x64_128_test(cuda::std::array<cuda::std::int64_t, 4>{42, 64, 108, 1024},
                             conv({8786399719555989948ull, 14954183901757012458ull}),
                             63);
    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int64_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      conv({15409921801541329777ull, 10546487400963404004ull}),
      1024);
  }
};
#endif // _CCCL_HAS_INT128()

struct noncopyable_key
{
  TEST_HOST_DEVICE_FUNC constexpr explicit noncopyable_key(cuda::std::int32_t value)
      : value_{value}
  {}

  noncopyable_key(const noncopyable_key&) = delete;
  noncopyable_key(noncopyable_key&&)      = default;

  cuda::std::int32_t value_;
};

static_assert(cuda::std::is_trivially_copyable_v<noncopyable_key>);

template <cuda::hash_algorithm Algorithm>
TEST_HOST_DEVICE_FUNC void test_noncopyable_key()
{
  const noncopyable_key key{42};
  const cuda::hash<noncopyable_key, Algorithm> hasher;
  assert(hasher(key) == hasher(cuda::std::span<const noncopyable_key, 1>{&key, 1}));
}

template <cuda::std::size_t Size>
struct byte_key
{
  cuda::std::byte data_[Size];
};

template <cuda::hash_algorithm Algorithm, cuda::std::size_t Size>
TEST_HOST_DEVICE_FUNC void test_sized_key()
{
  byte_key<Size> key{};
  for (cuda::std::size_t i = 0; i < Size; ++i)
  {
    key.data_[i] = static_cast<cuda::std::byte>(i);
  }

  const cuda::hash<byte_key<Size>, Algorithm> hasher;
  cuda::std::array<byte_key<Size>, 2> keys             = {key, key};
  keys[1].data_[0]                                     = static_cast<cuda::std::byte>(42);
  const cuda::std::array<byte_key<Size>, 2> const_keys = keys;

  assert(hasher(key) == hasher(cuda::std::span<const byte_key<Size>, 1>{&key, 1}));
  assert(hasher(cuda::std::span<byte_key<Size>, 2>{keys})
         == hasher(cuda::std::span<const byte_key<Size>, 2>{const_keys}));
}

template <cuda::hash_algorithm Algorithm>
TEST_HOST_DEVICE_FUNC void test_sized_keys()
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

TEST_HOST_DEVICE_FUNC void test_empty_spans()
{
  const cuda::std::span<const cuda::std::uint32_t> empty;
  assert((cuda::hash<cuda::std::uint32_t, cuda::hash_algorithm::xxhash_32>{}(empty) == 0x02cc5d05u));
  assert((cuda::hash<cuda::std::uint32_t, cuda::hash_algorithm::xxhash_64>{}(empty) == 0xef46db3751d8e999ull));
  assert((cuda::hash<cuda::std::uint32_t, cuda::hash_algorithm::murmurhash3_32>{}(empty) == 0u));
#if _CCCL_HAS_INT128()
  assert((cuda::hash<cuda::std::uint32_t, cuda::hash_algorithm::murmurhash3_x86_128>{}(empty) == 0));
  assert((cuda::hash<cuda::std::uint32_t, cuda::hash_algorithm::murmurhash3_x64_128>{}(empty) == 0));
#endif // _CCCL_HAS_INT128()
}

#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
static_assert(cuda::hash<cuda::std::uint32_t>{}(0u) == 148298089u);
static_assert((cuda::hash<cuda::std::uint32_t, cuda::hash_algorithm::murmurhash3_32>{}(0u) == 593689054u));
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()

TEST_HOST_DEVICE_FUNC void test()
{
  test_xxhash32{}();
  test_xxhash64{}();
  test_murmurhash3_32{}();
  test_sized_keys<cuda::hash_algorithm::xxhash_32>();
  test_sized_keys<cuda::hash_algorithm::xxhash_64>();
  test_sized_keys<cuda::hash_algorithm::murmurhash3_32>();
  test_noncopyable_key<cuda::hash_algorithm::xxhash_32>();
  test_noncopyable_key<cuda::hash_algorithm::xxhash_64>();
  test_noncopyable_key<cuda::hash_algorithm::murmurhash3_32>();
#if _CCCL_HAS_INT128()
  test_murmurhash3_x86_128{}();
  test_murmurhash3_x64_128{}();
  test_sized_keys<cuda::hash_algorithm::murmurhash3_x86_128>();
  test_sized_keys<cuda::hash_algorithm::murmurhash3_x64_128>();
  test_noncopyable_key<cuda::hash_algorithm::murmurhash3_x86_128>();
  test_noncopyable_key<cuda::hash_algorithm::murmurhash3_x64_128>();
#endif // _CCCL_HAS_INT128()
  test_empty_spans();
}

#if _CCCL_CUDA_COMPILATION()
__global__ void test_kernel()
{
  test();
}
#endif // _CCCL_CUDA_COMPILATION()

int main(int, char**)
{
  test();
#if _CCCL_CUDA_COMPILATION()
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(); assert(cudaDeviceSynchronize() == cudaSuccess);))
#endif // _CCCL_CUDA_COMPILATION()
  return 0;
}
