//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <testing.cuh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

template <int32_t Words>
struct large_key
{
  constexpr _CCCL_HOST_DEVICE large_key(int32_t value) noexcept
  {
    for (int32_t i = 0; i < Words; ++i)
    {
      data_[i] = value;
    }
  }

private:
  int32_t data_[Words];
};

template <cudax::cuco::hash_algorithm Algorithm>
struct hash_test
{
  template <typename Key, typename ResultT, typename... HashConstructorArgs>
  _CCCL_HOST_DEVICE void
  operator()(Key const& key, ResultT expected, HashConstructorArgs&&... hash_constructor_args) noexcept
  {
    cudax::cuco::hash<Key, Algorithm> hasher(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);

    cuda::std::array<Key, 1> arr_keys = {key};

    CUDAX_REQUIRE(hasher(key) == expected);
    CUDAX_REQUIRE(hasher(cuda::std::span<Key>(thrust::raw_pointer_cast(arr_keys.data()), arr_keys.size())) == expected);
  }
};

struct test_xxhash32
{
  hash_test<cudax::cuco::hash_algorithm::xxhash_32> xxhash32_test;

  _CCCL_HOST_DEVICE void operator()()
  {
    xxhash32_test(static_cast<char>(0), 3479547966u, 0);
    xxhash32_test(static_cast<char>(42), 3774771295u, 0);
    xxhash32_test(static_cast<char>(0), 2099223482u, 42);
    xxhash32_test(static_cast<int32_t>(0), 148298089u, 0);
    xxhash32_test(static_cast<int32_t>(0), 2132181312u, 42);
    xxhash32_test(static_cast<int32_t>(42), 1161967057u, 0);
    xxhash32_test(static_cast<int32_t>(123456789), 2987034094u, 0);
    xxhash32_test(static_cast<int64_t>(0), 3736311059u, 0);
    xxhash32_test(static_cast<int64_t>(0), 1076387279u, 42);
    xxhash32_test(static_cast<int64_t>(42), 2332451213u, 0);
    xxhash32_test(static_cast<int64_t>(123456789), 1561711919u, 0);
#if _CCCL_HAS_INT128()
    xxhash32_test(static_cast<__int128_t>(123456789), 1846633701u, 0);
#endif
    xxhash32_test(large_key<32>(123456789), 3715432378u, 0);
  }
};

struct test_xxhash64
{
  hash_test<cudax::cuco::hash_algorithm::xxhash_64> xxhash64_test;

  _CCCL_HOST_DEVICE void operator()()
  {
    xxhash64_test(static_cast<char>(0), 16804241149081757544ull, 0);
    xxhash64_test(static_cast<char>(42), 765293966243412708ull, 0);
    xxhash64_test(static_cast<char>(0), 9486749600008296231ull, 42);
    xxhash64_test(static_cast<int32_t>(0), 4246796580750024372ull, 0);
    xxhash64_test(static_cast<int32_t>(0), 3614696996920510707ull, 42);
    xxhash64_test(static_cast<int32_t>(42), 15516826743637085169ull, 0);
    xxhash64_test(static_cast<int32_t>(123456789), 9462334144942111946ull, 0);
    xxhash64_test(static_cast<int64_t>(0), 3803688792395291579ull, 0);
    xxhash64_test(static_cast<int64_t>(0), 13194218611613725804ull, 42);
    xxhash64_test(static_cast<int64_t>(42), 13066772586158965587ull, 0);
    xxhash64_test(static_cast<int64_t>(123456789), 14662639848940634189ull, 0);
#if _CCCL_HAS_INT128()
    xxhash64_test(static_cast<__int128_t>(123456789), 7986913354431084250ull, 0);
#endif
    xxhash64_test(large_key<32>(123456789), 2031761887105658523ull, 0);
  }
};

struct test_murmurhash3_32
{
  hash_test<cudax::cuco::hash_algorithm::murmurhash3_32> murmurhash3_32_test;

  _CCCL_HOST_DEVICE void operator()()
  {
    murmurhash3_32_test(static_cast<char>(0), 1364076727u, 0);
    murmurhash3_32_test(static_cast<char>(42), 338914844u, 0);
    murmurhash3_32_test(static_cast<char>(0), 3712240066u, 42);
    murmurhash3_32_test(static_cast<int32_t>(0), 593689054u, 0);
    murmurhash3_32_test(static_cast<int32_t>(0), 933211791u, 42);
    murmurhash3_32_test(static_cast<int32_t>(42), 3160117731u, 0);
    murmurhash3_32_test(static_cast<int32_t>(123456789), 3206620847u, 0);
    murmurhash3_32_test(static_cast<int64_t>(0), 1669671676u, 0);
    murmurhash3_32_test(static_cast<int64_t>(0), 2624043101u, 42);
    murmurhash3_32_test(static_cast<int64_t>(42), 1871679806u, 0);
    murmurhash3_32_test(static_cast<int64_t>(123456789), 690028081u, 0);
#if _CCCL_HAS_INT128()
    murmurhash3_32_test(static_cast<__int128_t>(123456789), 2191144977u, 0);
#endif
    murmurhash3_32_test(large_key<32>(123456789), 2555553099u, 0);
  }
};

#if _CCCL_HAS_INT128()
struct test_murmurhash3_x86_128
{
  hash_test<cudax::cuco::hash_algorithm::murmurhash3_x86_128> murmurhash3_x86_128_test;

  _CCCL_HOST_DEVICE __uint128_t conv(cuda::std::array<uint32_t, 4> const& arr) const
  {
    return cuda::std::bit_cast<__uint128_t>(arr);
  }

  _CCCL_HOST_DEVICE void operator()()
  {
    murmurhash3_x86_128_test(int32_t(0), conv({3422973727u, 2656139328u, 2656139328u, 2656139328u}), 0);
    murmurhash3_x86_128_test(int32_t(9), conv({2808089785u, 314604614u, 314604614u, 314604614u}), 0);
    murmurhash3_x86_128_test(int32_t(42), conv({3611919118u, 1962256489u, 1962256489u, 1962256489u}), 0);
    murmurhash3_x86_128_test(int32_t(42), conv({3399017053u, 732469929u, 732469929u, 732469929u}), 42);

    murmurhash3_x86_128_test(
      cuda::std::array<int32_t, 2>{2, 2}, conv({1234494082u, 1431451587u, 431049201u, 431049201u}), 0);
    murmurhash3_x86_128_test(
      cuda::std::array<int32_t, 3>{1, 4, 9}, conv({2516796247u, 2757675829u, 778406919u, 2453259553u}), 42);
    murmurhash3_x86_128_test(
      cuda::std::array<int32_t, 4>{42, 64, 108, 1024}, conv({2686265656u, 591236665u, 3797082165u, 2731908938u}), 63);
    murmurhash3_x86_128_test(cuda::std::array<int32_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                             conv({3918256832u, 4205523739u, 1707810111u, 1625952473u}),
                             1024);

    murmurhash3_x86_128_test(
      cuda::std::array<int64_t, 2>{2, 2}, conv({3811075945u, 727160712u, 3510740342u, 235225510u}), 0);
    murmurhash3_x86_128_test(
      cuda::std::array<int64_t, 3>{1, 4, 9}, conv({2817194959u, 206796677u, 3391242768u, 248681098u}), 42);
    murmurhash3_x86_128_test(
      cuda::std::array<int64_t, 4>{42, 64, 108, 1024}, conv({2335912146u, 1566515912u, 760710030u, 452077451u}), 63);
    murmurhash3_x86_128_test(cuda::std::array<int64_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                             conv({1101169764u, 1758958147u, 2406511780u, 2903571412u}),
                             1024);
  }
};

struct test_murmurhash3_x64_128
{
  hash_test<cudax::cuco::hash_algorithm::murmurhash3_x64_128> murmurhash3_x64_128_test;

  _CCCL_HOST_DEVICE __uint128_t conv(cuda::std::array<uint64_t, 2> const& arr) const
  {
    return cuda::std::bit_cast<__uint128_t>(arr);
  }

  _CCCL_HOST_DEVICE void operator()()
  {
    murmurhash3_x64_128_test(int32_t(0), conv({14961230494313510588ull, 6383328099726337777ull}), 0);
    murmurhash3_x64_128_test(int32_t(9), conv({1779292183511753683ull, 16298496441448380334ull}), 0);
    murmurhash3_x64_128_test(int32_t(42), conv({2913627637088662735ull, 16344193523890567190ull}), 0);
    murmurhash3_x64_128_test(int32_t(42), conv({2248879576374326886ull, 18006515275339376488ull}), 42);

    murmurhash3_x64_128_test(
      cuda::std::array<int32_t, 2>{2, 2}, conv({12221386834995143465ull, 6690950894782946573ull}), 0);
    murmurhash3_x64_128_test(
      cuda::std::array<int32_t, 3>{1, 4, 9}, conv({299140022350411792ull, 9891903873182035274ull}), 42);
    murmurhash3_x64_128_test(
      cuda::std::array<int32_t, 4>{42, 64, 108, 1024}, conv({4333511168876981289ull, 4659486988434316416ull}), 63);
    murmurhash3_x64_128_test(cuda::std::array<int32_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                             conv({3302412811061286680ull, 7070355726356610672ull}),
                             1024);

    murmurhash3_x64_128_test(
      cuda::std::array<int64_t, 2>{2, 2}, conv({8554944597931919519ull, 14938998000509429729ull}), 0);
    murmurhash3_x64_128_test(
      cuda::std::array<int64_t, 3>{1, 4, 9}, conv({13442629947720186435ull, 7061727494178573325ull}), 42);
    murmurhash3_x64_128_test(
      cuda::std::array<int64_t, 4>{42, 64, 108, 1024}, conv({8786399719555989948ull, 14954183901757012458ull}), 63);
    murmurhash3_x64_128_test(cuda::std::array<int64_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                             conv({15409921801541329777ull, 10546487400963404004ull}),
                             1024);
  }
};
#endif // _CCCL_HAS_INT128()

template <typename TestFn>
__global__ void test_hasher_kernel(TestFn test_fn)
{
  test_fn();
}

template <typename TestFn>
void test_hasher_on_device(TestFn test_fn)
{
  test_hasher_kernel<<<1, 1>>>(test_fn);
  CUDAX_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("Test Hasher's on host and device", "")
{
  SECTION("host-generated hash values match the reference implementation.")
  {
    test_xxhash32{}();
    test_xxhash64{}();
    test_murmurhash3_32{}();
#if _CCCL_HAS_INT128()
    test_murmurhash3_x86_128{}();
    test_murmurhash3_x64_128{}();
#endif // _CCCL_HAS_INT128()
  }

  SECTION("device-generated hash values match the reference implementation.")
  {
    test_hasher_on_device(test_xxhash32{});
    test_hasher_on_device(test_xxhash64{});
    test_hasher_on_device(test_murmurhash3_32{});
#if _CCCL_HAS_INT128()
    test_hasher_on_device(test_murmurhash3_x86_128{});
    test_hasher_on_device(test_murmurhash3_x64_128{});
#endif // _CCCL_HAS_INT128()
  }
}
