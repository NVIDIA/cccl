//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

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

template <typename Hash, typename Key, typename ResultT, typename... HashConstructorArgs>
static _CCCL_HOST_DEVICE bool
check_hash_result(Key const& key, ResultT expected, HashConstructorArgs&&... hash_constructor_args) noexcept
{
  Hash h(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);

  cuda::std::array<Key, 1> arr_keys = {key};

  return (h(key) == expected)
      && (h(cuda::std::span<Key>(thrust::raw_pointer_cast(arr_keys.data()), arr_keys.size())) == expected);
}

struct test_xxhash32
{
  _CCCL_HOST_DEVICE bool operator()()
  {
    return check_hash_result<cudax::cuco::Hash<char>>(static_cast<char>(0), 3479547966u, 0)
        && check_hash_result<cudax::cuco::Hash<char>>(static_cast<char>(42), 3774771295u, 0)
        && check_hash_result<cudax::cuco::Hash<char>>(static_cast<char>(0), 2099223482u, 42)
        && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(0), 148298089u, 0)
        && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(0), 2132181312u, 42)
        && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(42), 1161967057u, 0)
        && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(123456789), 2987034094u, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(0), 3736311059u, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(0), 1076387279u, 42)
        && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(42), 2332451213u, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(123456789), 1561711919u, 0)
#if _CCCL_HAS_INT128()
        && check_hash_result<cudax::cuco::Hash<__int128_t>>(static_cast<__int128_t>(123456789), 1846633701u, 0)
#endif
        && check_hash_result<cudax::cuco::Hash<large_key<32>>>(large_key<32>(123456789), 3715432378u, 0);
  }
};

struct test_xxhash64
{
  _CCCL_HOST_DEVICE bool operator()()
  {
    return check_hash_result<cudax::cuco::Hash<char, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<char>(0), 16804241149081757544ull, 0)
        && check_hash_result<cudax::cuco::Hash<char, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<char>(42), 765293966243412708ull, 0)
        && check_hash_result<cudax::cuco::Hash<char, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<char>(0), 9486749600008296231ull, 42)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int32_t>(0), 4246796580750024372ull, 0)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int32_t>(0), 3614696996920510707ull, 42)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int32_t>(42), 15516826743637085169ull, 0)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int32_t>(123456789), 9462334144942111946ull, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int64_t>(0), 3803688792395291579ull, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int64_t>(0), 13194218611613725804ull, 42)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int64_t>(42), 13066772586158965587ull, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<int64_t>(123456789), 14662639848940634189ull, 0)
#if _CCCL_HAS_INT128()
        && check_hash_result<cudax::cuco::Hash<__int128_t, cudax::cuco::HashStrategy::XXHash_64>>(
             static_cast<__int128_t>(123456789), 7986913354431084250ull, 0)
#endif
        && check_hash_result<cudax::cuco::Hash<large_key<32>, cudax::cuco::HashStrategy::XXHash_64>>(
             large_key<32>(123456789), 2031761887105658523ull, 0);
  }
};

struct test_murmurhash3_32
{
  _CCCL_HOST_DEVICE bool operator()()
  {
    return check_hash_result<cudax::cuco::Hash<char, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<char>(0), 1364076727u, 0)
        && check_hash_result<cudax::cuco::Hash<char, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<char>(42), 338914844u, 0)
        && check_hash_result<cudax::cuco::Hash<char, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<char>(0), 3712240066u, 42)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int32_t>(0), 593689054u, 0)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int32_t>(0), 933211791u, 42)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int32_t>(42), 3160117731u, 0)
        && check_hash_result<cudax::cuco::Hash<int32_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int32_t>(123456789), 3206620847u, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int64_t>(0), 1669671676u, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int64_t>(0), 2624043101u, 42)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int64_t>(42), 1871679806u, 0)
        && check_hash_result<cudax::cuco::Hash<int64_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<int64_t>(123456789), 690028081u, 0)
#if _CCCL_HAS_INT128()
        && check_hash_result<cudax::cuco::Hash<__int128_t, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             static_cast<__int128_t>(123456789), 2191144977u, 0)
#endif
        && check_hash_result<cudax::cuco::Hash<large_key<32>, cudax::cuco::HashStrategy::MurmurHash3_32>>(
             large_key<32>(123456789), 2555553099u, 0);
  }
};

template <typename TestFn, typename ResultIt>
__global__ void test_hasher_kernel(TestFn test_fn, ResultIt result)
{
  result[0] = test_fn();
}

template <typename TestFn>
void test_hasher_on_device(TestFn test_fn)
{
  thrust::device_vector<bool, 1> result{false};
  test_hasher_kernel<<<1, 1>>>(test_fn, result.begin());
  CUDAX_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  CUDAX_REQUIRE(result[0]);
}

TEST_CASE("Utility Hasher _XXHash_32 test", "")
{
  SECTION("host-generated hash values match the reference implementation.")
  {
    CUDAX_REQUIRE(test_xxhash32{}());
  }

  SECTION("device-generated hash values match the reference implementation.")
  {
    test_hasher_on_device(test_xxhash32{});
  }
}

TEST_CASE("Utility Hasher _XXHash_64 test", "")
{
  SECTION("host-generated hash values match the reference implementation."){CUDAX_REQUIRE(test_xxhash64{}())}

  SECTION("device-generated hash values match the reference implementation.")
  {
    test_hasher_on_device(test_xxhash64{});
  }
}

TEST_CASE("Utility Hasher _MurmurHash3_32 test", "")
{
  SECTION("host-generated hash values match the reference implementation.")
  {
    CUDAX_REQUIRE(test_murmurhash3_32{}());
  }

  SECTION("device-generated hash values match the reference implementation.")
  {
    test_hasher_on_device(test_murmurhash3_32{});
  }
}
