// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUDAX_TEST_CUCO_FIXED_CAPACITY_SET_TEST_COMMON_CUH
#define CUDAX_TEST_CUCO_FIXED_CAPACITY_SET_TEST_COMMON_CUH

#include <cuda/__cccl_config>
#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = cuda::std::integral_constant<int, N>;

struct matches_membership
{
  const int* results;
  int num_present;

  [[nodiscard]] _CCCL_DEVICE_API bool operator()(int i) const noexcept
  {
    return results[i] == static_cast<int>(i < num_present);
  }
};

struct equals_value
{
  int expected;

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(int value) const noexcept
  {
    return value == expected;
  }
};

struct test_context
{
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(stream.device());

  [[nodiscard]] _CCCL_HOST_API auto policy()
  {
    return cuda::execution::gpu.with(cuda::get_stream, stream).with(cuda::mr::get_memory_resource, mr);
  }

  [[nodiscard]] _CCCL_HOST_API bool matches(const int* results, int count, int num_present)
  {
    return cuda::std::all_of(
      policy(),
      cuda::counting_iterator<int>{0},
      cuda::counting_iterator<int>{count},
      matches_membership{results, num_present});
  }

  [[nodiscard]] _CCCL_HOST_API bool all_equal(const int* results, int count, int expected)
  {
    return cuda::std::all_of(policy(), results, results + count, equals_value{expected});
  }
};

#endif // CUDAX_TEST_CUCO_FIXED_CAPACITY_SET_TEST_COMMON_CUH
