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

template <typename Hash, typename Key, typename... HashConstructorArgs>
static _CCCL_HOST_DEVICE bool
check_hash_result(Key const& key, std::uint32_t expected, HashConstructorArgs&&... hash_constructor_args) noexcept
{
  Hash h(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);

  cuda::std::array<Key, 1> arr_keys = {key};

  return (h(key) == expected)
      && (h(cuda::std::span<Key>(thrust::raw_pointer_cast(arr_keys.data()), arr_keys.size())) == expected);
}

_CCCL_HOST_DEVICE bool test()
{
  return check_hash_result<cudax::cuco::Hash<char>>(static_cast<char>(0), 3479547966, 0)
      && check_hash_result<cudax::cuco::Hash<char>>(static_cast<char>(42), 3774771295, 0)
      && check_hash_result<cudax::cuco::Hash<char>>(static_cast<char>(0), 2099223482, 42)
      && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(0), 148298089, 0)
      && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(0), 2132181312, 42)
      && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(42), 1161967057, 0)
      && check_hash_result<cudax::cuco::Hash<int32_t>>(static_cast<int32_t>(123456789), 2987034094, 0)
      && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(0), 3736311059, 0)
      && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(0), 1076387279, 42)
      && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(42), 2332451213, 0)
      && check_hash_result<cudax::cuco::Hash<int64_t>>(static_cast<int64_t>(123456789), 1561711919, 0)
#if _CCCL_HAS_INT128()
      && check_hash_result<cudax::cuco::Hash<__int128_t>>(static_cast<__int128_t>(123456789), 1846633701, 0)
#endif
      && check_hash_result<cudax::cuco::Hash<large_key<32>>>(large_key<32>(123456789), 3715432378, 0);
}

template <typename ResultIt>
__global__ void test_on_device(ResultIt result)
{
  const bool res = test();
  result[0]      = res;
}

TEST_CASE("Utility Hasher _XXHash_32 test", "")
{
  SECTION("host-generated hash values match the reference implementation.")
  {
    CHECK(test());
  }

  SECTION("device-generated hash values match the reference implementation.")
  {
    cuda::std::array<bool, 1> result{false};
    test_on_device<<<1, 1>>>(result.begin());
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(result[0]);
  }
}
