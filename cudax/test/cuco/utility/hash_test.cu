/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/device_vector.h>
#include <thrust/logical.h>

#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include <cuda/experimental/__cuco/detail/__config>
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

template <typename Hash, typename... HashConstructorArgs>
static _CCCL_HOST_DEVICE bool check_hash_result(
  typename Hash::argument_type const& key,
  typename Hash::result_type expected,
  HashConstructorArgs&&... hash_constructor_args) noexcept
{
  Hash h(::cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);
  return (h(key) == expected);
}

template <typename OutputIter>
__global__ void check_hash_result_kernel_32(OutputIter result)
{
  int i = 0;

  result[i++] = check_hash_result<cudax::cuco::xxhash_32<char>>(0, 3479547966, 0);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<char>>(42, 3774771295, 0);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<char>>(0, 2099223482, 42);

  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int32_t>>(0, 148298089, 0);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int32_t>>(0, 2132181312, 42);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int32_t>>(42, 1161967057, 0);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int32_t>>(123456789, 2987034094, 0);

  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int64_t>>(0, 3736311059, 0);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int64_t>>(0, 1076387279, 42);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int64_t>>(42, 2332451213, 0);
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<int64_t>>(123456789, 1561711919, 0);

#if defined(CUCO_HAS_INT128)
  result[i++] = check_hash_result<cudax::cuco::xxhash_32<__int128>>(123456789, 1846633701, 0);
#endif

  result[i++] = check_hash_result<cudax::cuco::xxhash_32<large_key<32>>>(123456789, 3715432378, 0);
}

TEST_CASE("utility cudax::cuco::xxhash_32 test", "")
{
  // Reference hash values were computed using https://github.com/Cyan4973/xxHash
  SECTION("Check if host-generated hash values match the reference implementation.")
  {
    CHECK(check_hash_result<cudax::cuco::xxhash_32<char>>(0, 3479547966, 0));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<char>>(42, 3774771295, 0));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<char>>(0, 2099223482, 42));

    CHECK(check_hash_result<cudax::cuco::xxhash_32<int32_t>>(0, 148298089, 0));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<int32_t>>(0, 2132181312, 42));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<int32_t>>(42, 1161967057, 0));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<int32_t>>(123456789, 2987034094, 0));

    CHECK(check_hash_result<cudax::cuco::xxhash_32<int64_t>>(0, 3736311059, 0));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<int64_t>>(0, 1076387279, 42));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<int64_t>>(42, 2332451213, 0));
    CHECK(check_hash_result<cudax::cuco::xxhash_32<int64_t>>(123456789, 1561711919, 0));

#if defined(CUCO_HAS_INT128)
    CHECK(check_hash_result<cudax::cuco::xxhash_32<__int128>>(123456789, 1846633701, 0));
#endif

    // 32*4=128-byte key to test the pipelined outermost hashing loop
    CHECK(check_hash_result<cudax::cuco::xxhash_32<large_key<32>>>(123456789, 3715432378, 0));
  }

  SECTION("Check if device-generated hash values match the reference implementation.")
  {
    thrust::device_vector<bool> result(20, true);

    check_hash_result_kernel_32<<<1, 1>>>(result.begin());

    CHECK(thrust::all_of(result.begin(), result.end(), ::cuda::std::identity{}));
  }
}
