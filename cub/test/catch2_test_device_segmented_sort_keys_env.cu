// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
#include <cuda/__functional/call_or.h>
#include <cuda/std/cstdint>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortKeysDescending, sort_keys_descending);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::StableSortKeys, stable_sort_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::StableSortKeysDescending, stable_sort_keys_descending);

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortKeys, sort_keys_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortKeysDescending, sort_keys_descending_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortKeys, stable_sort_keys_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortKeysDescending, stable_sort_keys_descending_two_phase);

// %PARAM% TEST_LAUNCH lid 0:1

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("DeviceSegmentedSort::SortKeys works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeys(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortKeysDescending works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeysDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("DeviceSegmentedSort::StableSortKeys works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeys(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("DeviceSegmentedSort::StableSortKeysDescending works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeysDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortKeys DoubleBuffer works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::SortKeys(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortKeysDescending DoubleBuffer works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::SortKeysDescending(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

CUB_TEST_CASE("DeviceSegmentedSort::StableSortKeys DoubleBuffer works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::StableSortKeys(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

CUB_TEST_CASE("DeviceSegmentedSort::StableSortKeysDescending DoubleBuffer works with default environment",
              "[segmented_sort][keys][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::StableSortKeysDescending(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

#endif

CUB_TEST("DeviceSegmentedSort::SortKeys uses environment", "[segmented_sort][keys][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys(thrust::raw_pointer_cast(keys_in.data()),
            thrust::raw_pointer_cast(keys_out.data()),
            static_cast<cuda::std::int64_t>(keys_in.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1,
            env);

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

CUB_TEST("DeviceSegmentedSort::SortKeysDescending uses environment", "[segmented_sort][keys][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

CUB_TEST("DeviceSegmentedSort::StableSortKeys uses environment", "[segmented_sort][keys][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeys(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

CUB_TEST("DeviceSegmentedSort::StableSortKeysDescending uses environment", "[segmented_sort][keys][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

CUB_TEST("DeviceSegmentedSort::SortKeys DoubleBuffer uses environment", "[segmented_sort][keys][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys(d_keys,
            static_cast<int>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1,
            env);

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

CUB_TEST("DeviceSegmentedSort::SortKeysDescending DoubleBuffer uses environment",
         "[segmented_sort][keys][device]",
         CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys_descending(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

CUB_TEST("DeviceSegmentedSort::StableSortKeys DoubleBuffer uses environment",
         "[segmented_sort][keys][device]",
         CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeys(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  const c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

CUB_TEST("DeviceSegmentedSort::StableSortKeysDescending DoubleBuffer uses environment",
         "[segmented_sort][keys][device]",
         CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys_descending(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  const c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

// The keys-only two-phase overloads, with their launch wrappers and sort order
struct sort_keys_api
{
  using launch_t                   = sort_keys_two_phase_t;
  static constexpr bool descending = false;

  template <class... Ts>
  static cudaError_t call(Ts&&... args)
  {
    return cub::DeviceSegmentedSort::SortKeys(cuda::std::forward<Ts>(args)...);
  }
};

struct sort_keys_descending_api
{
  using launch_t                   = sort_keys_descending_two_phase_t;
  static constexpr bool descending = true;

  template <class... Ts>
  static cudaError_t call(Ts&&... args)
  {
    return cub::DeviceSegmentedSort::SortKeysDescending(cuda::std::forward<Ts>(args)...);
  }
};

struct stable_sort_keys_api
{
  using launch_t                   = stable_sort_keys_two_phase_t;
  static constexpr bool descending = false;

  template <class... Ts>
  static cudaError_t call(Ts&&... args)
  {
    return cub::DeviceSegmentedSort::StableSortKeys(cuda::std::forward<Ts>(args)...);
  }
};

struct stable_sort_keys_descending_api
{
  using launch_t                   = stable_sort_keys_descending_two_phase_t;
  static constexpr bool descending = true;

  template <class... Ts>
  static cudaError_t call(Ts&&... args)
  {
    return cub::DeviceSegmentedSort::StableSortKeysDescending(cuda::std::forward<Ts>(args)...);
  }
};

using two_phase_apis =
  c2h::type_list<sort_keys_api, sort_keys_descending_api, stable_sort_keys_api, stable_sort_keys_descending_api>;

// Expected result of sorting the keys {8, 6, 7, 5, 3, 0, 9} in the segments [0, 3) and [3, 7) with Api
template <class Api>
[[nodiscard]] c2h::device_vector<int> expected_two_segments()
{
  return Api::descending ? c2h::device_vector<int>{8, 7, 6, 9, 5, 3, 0} : c2h::device_vector<int>{6, 7, 8, 0, 3, 5, 9};
}

// Calls the two-phase overload of Api with args and env: queries the temporary storage size, executes with the
// allocated storage, and checks that all kernels are launched on the stream of env.
template <class Api, class EnvT, class... Args>
void run_two_phase(const EnvT& env, Args&&... args)
{
  const stream_scope scope{cuda::__call_or(cuda::get_stream, cuda::stream_ref{cudaStream_t{}}, env).get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == Api::call(nullptr, temp_storage_bytes, args..., env));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == Api::call(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, args..., env));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// Copies the current buffer of a DoubleBuffer back after sorting
[[nodiscard]] c2h::device_vector<int> current_keys(cub::DoubleBuffer<int> d_keys, int num_items)
{
  const auto current = thrust::device_pointer_cast(d_keys.Current());
  return c2h::device_vector<int>(current, current + num_items);
}

#if TEST_LAUNCH == 0

// The launch wrappers copy their arguments, which neither the non-copyable stream types of test_with_custom_streams nor
// a DoubleBuffer, whose selector is updated by the algorithm, allow. So the two-phase overloads are called directly
// with custom streams. Test them with host launch only.
template <class Api>
void test_two_phase_with_custom_streams()
{
  auto keys_buf0       = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1       = c2h::device_vector<int>(keys_buf0.size(), thrust::no_init);
  const auto offsets   = c2h::device_vector<int>{0, 3, 7};
  const int* d_offsets = thrust::raw_pointer_cast(offsets.data());
  const int num_items  = static_cast<int>(keys_buf0.size());
  const auto expected  = expected_two_segments<Api>();

  SECTION("pointers")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase<Api>(
        env,
        thrust::raw_pointer_cast(keys_buf0.data()),
        thrust::raw_pointer_cast(keys_buf1.data()),
        num_items,
        2,
        d_offsets,
        d_offsets + 1);
      REQUIRE(keys_buf1 == expected);
    });
  }

  SECTION("DoubleBuffer")
  {
    test_with_custom_streams([&](const auto& env) {
      cub::DoubleBuffer<int> d_keys(
        thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
      run_two_phase<Api>(env, d_keys, num_items, 2, d_offsets, d_offsets + 1);
      REQUIRE(current_keys(d_keys, num_items) == expected);
    });
  }
}

CUB_TEST("DeviceSegmentedSort keys two-phase overloads work with custom streams",
         "[segmented_sort][keys][device]",
         CUB_SMALL,
         two_phase_apis)
{
  test_two_phase_with_custom_streams<c2h::get<0, TestType>>();
}

#endif // TEST_LAUNCH == 0

// Before the environment parameter, the two-phase overloads took `cudaStream_t stream = nullptr`, so callers passing
// nullptr, a literal 0 or a default constructed cudaStream_t for the stream exist. All must keep compiling and keep
// running on the default stream.
template <class Api, class StreamT>
void check_legacy_null_stream(StreamT stream)
{
  auto keys_buf0       = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1       = c2h::device_vector<int>(keys_buf0.size(), thrust::no_init);
  const auto offsets   = c2h::device_vector<int>{0, 3, 7};
  const int* d_offsets = thrust::raw_pointer_cast(offsets.data());
  const int num_items  = static_cast<int>(keys_buf0.size());
  const auto expected  = expected_two_segments<Api>();

  SECTION("pointers")
  {
    const stream_scope scope{cudaStream_t{}};
    typename Api::launch_t{}(
      thrust::raw_pointer_cast(keys_buf0.data()),
      thrust::raw_pointer_cast(keys_buf1.data()),
      num_items,
      2,
      d_offsets,
      d_offsets + 1,
      stream);
    REQUIRE(keys_buf1 == expected);
  }

#if TEST_LAUNCH == 0
  // The launch wrappers copy the DoubleBuffer, so call the overload directly
  SECTION("DoubleBuffer")
  {
    cub::DoubleBuffer<int> d_keys(
      thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
    run_two_phase<Api>(stream, d_keys, num_items, 2, d_offsets, d_offsets + 1);
    REQUIRE(current_keys(d_keys, num_items) == expected);
  }
#endif // TEST_LAUNCH == 0
}

CUB_TEST("DeviceSegmentedSort keys two-phase overloads accept legacy null stream arguments",
         "[segmented_sort][keys][device]",
         CUB_SMALL,
         two_phase_apis)
{
  using api_t = c2h::get<0, TestType>;

  SECTION("nullptr")
  {
    check_legacy_null_stream<api_t>(nullptr);
  }

  SECTION("literal 0")
  {
    check_legacy_null_stream<api_t>(0);
  }

  SECTION("cudaStream_t{}")
  {
    check_legacy_null_stream<api_t>(cudaStream_t{});
  }
}

template <int BlockThreads>
struct segmented_sort_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::SegmentedSortPolicy
  {
    return cub::SegmentedSortPolicy{
      cub::SegmentedSortRadixSortPolicy{
        BlockThreads, 1, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::RADIX_RANK_BASIC, cub::BLOCK_SCAN_WARP_SCANS, 4},
      cub::SegmentedSortSubWarpMergeSortPolicy{
        BlockThreads, 32, 1, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT},
      cub::SegmentedSortSubWarpMergeSortPolicy{
        BlockThreads, 4, 1, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT},
      1000000};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

CUB_TEST("DeviceSegmentedSort::SortKeys can be tuned", "[segmented_sort][keys][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  sort_keys(thrust::raw_pointer_cast(keys_in.data()),
            thrust::raw_pointer_cast(keys_out.data()),
            static_cast<int>(keys_in.size()),
            1,
            thrust::raw_pointer_cast(begin_offsets.data()),
            end_offsets,
            env);

  const c2h::device_vector<int> expected{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "DeviceSegmentedSort::SortKeysDescending can be tuned", "[segmented_sort][keys][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  const c2h::device_vector<int> expected{9, 8, 7, 6, 5, 3, 0};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedSort::StableSortKeys can be tuned", "[segmented_sort][keys][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  stable_sort_keys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  const c2h::device_vector<int> expected{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedSort::StableSortKeysDescending can be tuned",
         "[segmented_sort][keys][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  stable_sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  const c2h::device_vector<int> expected{9, 8, 7, 6, 5, 3, 0};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedSort keys two-phase overloads can be tuned",
         "[segmented_sort][keys][device]",
         CUB_SMALL,
         two_phase_apis,
         block_sizes)
{
  using api_t                              = c2h::get<0, TestType>;
  constexpr unsigned int target_block_size = c2h::get<1, TestType>::value;
  auto keys_buf0                           = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1                           = c2h::device_vector<int>(7, thrust::no_init);
  const auto begin_offsets                 = c2h::device_vector<int>{0};
  const int* d_begin_offsets               = thrust::raw_pointer_cast(begin_offsets.data());
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));
  const auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});
  const auto expected =
    api_t::descending ? c2h::device_vector<int>{9, 8, 7, 6, 5, 3, 0} : c2h::device_vector<int>{0, 3, 5, 6, 7, 8, 9};

  SECTION("pointers")
  {
    typename api_t::launch_t{}(
      thrust::raw_pointer_cast(keys_buf0.data()),
      thrust::raw_pointer_cast(keys_buf1.data()),
      7,
      1,
      d_begin_offsets,
      end_offsets,
      env);
    REQUIRE(keys_buf1 == expected);
  }

  SECTION("DoubleBuffer")
  {
    cub::DoubleBuffer<int> d_keys(
      thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
    run_two_phase<api_t>(env, d_keys, 7, 1, d_begin_offsets, end_offsets);
    REQUIRE(current_keys(d_keys, 7) == expected);
  }

  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test SegmentedSortPolicy properties", "[segmented_sort][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedSortPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedSortPolicy>);
  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedSortRadixSortPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedSortRadixSortPolicy>);
  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedSortSubWarpMergeSortPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedSortSubWarpMergeSortPolicy>);

  // aggregate init
  constexpr auto p1_large = cub::SegmentedSortRadixSortPolicy{
    256, 16, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::RADIX_RANK_MEMOIZE, cub::BLOCK_SCAN_RAKING_MEMOIZE, 6};
  constexpr auto p1_medium = cub::SegmentedSortSubWarpMergeSortPolicy{
    256, 32, 7, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT};
  constexpr auto p1_small = cub::SegmentedSortSubWarpMergeSortPolicy{
    256, 4, 7, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT};
  constexpr auto p1 = cub::SegmentedSortPolicy{p1_large, p1_medium, p1_small, 300};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2_large = cub::SegmentedSortRadixSortPolicy{
    .threads_per_block = 256,
    .items_per_thread  = 16,
    .load_algorithm    = cub::BLOCK_LOAD_DIRECT,
    .load_modifier     = cub::LOAD_DEFAULT,
    .rank_algorithm    = cub::RADIX_RANK_MEMOIZE,
    .scan_algorithm    = cub::BLOCK_SCAN_RAKING_MEMOIZE,
    .radix_bits        = 6};
  constexpr auto p2_medium = cub::SegmentedSortSubWarpMergeSortPolicy{
    .threads_per_block = 256,
    .threads_per_warp  = 32,
    .items_per_thread  = 7,
    .load_algorithm    = cub::WARP_LOAD_DIRECT,
    .load_modifier     = cub::LOAD_DEFAULT,
    .store_algorithm   = cub::WARP_STORE_DIRECT};
  constexpr auto p2_small = cub::SegmentedSortSubWarpMergeSortPolicy{
    .threads_per_block = 256,
    .threads_per_warp  = 4,
    .items_per_thread  = 7,
    .load_algorithm    = cub::WARP_LOAD_DIRECT,
    .load_modifier     = cub::LOAD_DEFAULT,
    .store_algorithm   = cub::WARP_STORE_DIRECT};
  constexpr auto p2 = cub::SegmentedSortPolicy{
    .large_segment = p2_large, .medium_segment = p2_medium, .small_segment = p2_small, .partitioning_threshold = 300};
#  else
  constexpr auto p2_large  = p1_large;
  constexpr auto p2_small  = p1_small;
  constexpr auto p2_medium = p1_medium;
  constexpr auto p2        = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1_large == p2_large);
  STATIC_REQUIRE_FALSE(p1_large != p2_large);

  STATIC_REQUIRE(p1_small == p2_small);
  STATIC_REQUIRE_FALSE(p1_small != p2_small);

  STATIC_REQUIRE(p1_medium == p2_medium);
  STATIC_REQUIRE_FALSE(p1_medium != p2_medium);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1_large)
          == "SegmentedSortRadixSortPolicy { .threads_per_block = 256, .items_per_thread = 16"
             ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
             ", .rank_algorithm = RADIX_RANK_MEMOIZE, .scan_algorithm = BLOCK_SCAN_RAKING_MEMOIZE"
             ", .radix_bits = 6 }");
  REQUIRE(to_string(p1_small)
          == "SegmentedSortSubWarpMergeSortPolicy { .threads_per_block = 256, .threads_per_warp = 4"
             ", .items_per_thread = 7, .load_algorithm = WARP_LOAD_DIRECT"
             ", .load_modifier = LOAD_DEFAULT, .store_algorithm = WARP_STORE_DIRECT }");
  REQUIRE(to_string(p1_medium)
          == "SegmentedSortSubWarpMergeSortPolicy { .threads_per_block = 256, .threads_per_warp = 32"
             ", .items_per_thread = 7, .load_algorithm = WARP_LOAD_DIRECT"
             ", .load_modifier = LOAD_DEFAULT, .store_algorithm = WARP_STORE_DIRECT }");
  REQUIRE(
    to_string(p1)
    == "SegmentedSortPolicy { .large_segment = SegmentedSortRadixSortPolicy {"
       " .threads_per_block = 256, .items_per_thread = 16"
       ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
       ", .rank_algorithm = RADIX_RANK_MEMOIZE, .scan_algorithm = BLOCK_SCAN_RAKING_MEMOIZE"
       ", .radix_bits = 6 }"
       ", .medium_segment = SegmentedSortSubWarpMergeSortPolicy { .threads_per_block = 256"
       ", .threads_per_warp = 32, .items_per_thread = 7, .load_algorithm = WARP_LOAD_DIRECT"
       ", .load_modifier = LOAD_DEFAULT, .store_algorithm = WARP_STORE_DIRECT }"
       ", .small_segment = SegmentedSortSubWarpMergeSortPolicy { .threads_per_block = 256"
       ", .threads_per_warp = 4, .items_per_thread = 7, .load_algorithm = WARP_LOAD_DIRECT"
       ", .load_modifier = LOAD_DEFAULT, .store_algorithm = WARP_STORE_DIRECT }"
       ", .partitioning_threshold = 300 }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
