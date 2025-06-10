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

#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/limits>

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

template <typename Hash, typename Key, typename... HashConstructorArgs>
static _CCCL_HOST_DEVICE bool
check_hash_result(Key const& key, std::uint32_t expected, HashConstructorArgs&&... hash_constructor_args) noexcept
{
  Hash h(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);
  return (h(key) == expected);
}

template <typename TestFn, typename... Args>
__global__ void test_fn_on_device(TestFn test_fn, Args... args)
{
  test_fn(args...);
}

struct test_xxhash_32
{
  template <typename OutputIter>
  _CCCL_HOST_DEVICE void operator()(OutputIter result)
  {
    int i = 0;

    result[i++] = check_hash_result<cudax::cuco::Hash<char>>(0, 3479547966, 0);
    result[i++] = check_hash_result<cudax::cuco::Hash<char>>(42, 3774771295, 0);
    result[i++] = check_hash_result<cudax::cuco::Hash<char>>(0, 2099223482, 42);

    result[i++] = check_hash_result<cudax::cuco::Hash<int32_t>>(0, 148298089, 0);
    result[i++] = check_hash_result<cudax::cuco::Hash<int32_t>>(0, 2132181312, 42);
    result[i++] = check_hash_result<cudax::cuco::Hash<int32_t>>(42, 1161967057, 0);
    result[i++] = check_hash_result<cudax::cuco::Hash<int32_t>>(123456789, 2987034094, 0);

    result[i++] = check_hash_result<cudax::cuco::Hash<int64_t>>(0, 3736311059, 0);
    result[i++] = check_hash_result<cudax::cuco::Hash<int64_t>>(0, 1076387279, 42);
    result[i++] = check_hash_result<cudax::cuco::Hash<int64_t>>(42, 2332451213, 0);
    result[i++] = check_hash_result<cudax::cuco::Hash<int64_t>>(123456789, 1561711919, 0);

#if defined(CUCO_HAS_INT128)
    result[i++] = check_hash_result<cudax::cuco::Hash<__int128>>(123456789, 1846633701, 0);
#endif

    result[i++] = check_hash_result<cudax::cuco::Hash<large_key<32>>>(123456789, 3715432378, 0);
  }
};

TEST_CASE("Utility Hasher _XXHash_32 test", "")
{
  SECTION("host-generated hash values match the reference implementation.")
  {
    thrust::host_vector<bool> result(20, true);

    test_xxhash_32 test;
    test(result.begin());

    CHECK(thrust::all_of(thrust::host, result.begin(), result.end(), ::cuda::std::identity{}));
  }

  SECTION("device-generated hash values match the reference implementation.")
  {
    thrust::device_vector<bool> result(20, true);

    test_fn_on_device<<<1, 1>>>(test_xxhash_32{}, result.begin());

    CHECK(thrust::all_of(result.begin(), result.end(), ::cuda::std::identity{}));
  }
}
