// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/std/functional>
#include <cuda/stream>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortKeys, fixed_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortKeysDescending, fixed_keys_descending);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortPairs, fixed_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortPairsDescending, fixed_pairs_descending);

struct fixed_sort_tuning
{
  _CCCL_HOST_DEVICE constexpr cub::SegmentedSortPolicy operator()(cuda::compute_capability cc) const
  {
    auto policy                             = cub::detail::segmented_sort::policy_selector_from_types<int, int>{}(cc);
    policy.small_segment.threads_per_block  = 128;
    policy.medium_segment.threads_per_block = 128;
    policy.large_segment.threads_per_block  = 128;
    return policy;
  }
};

template <cub::SortOrder Order, typename ValueT>
void* tuned_small_kernel()
{
  using namespace cub::detail::segmented_sort;
  using offset_it = decltype(cuda::make_transform_iterator(
    cuda::counting_iterator<global_segment_offset_t>{0}, fixed_segment_offset<int>{3}));
  return reinterpret_cast<void*>(
    DeviceSegmentedSortKernelSmall<
      Order,
      fixed_sort_tuning,
      int,
      ValueT,
      offset_it,
      offset_it,
      global_segment_offset_t,
      cuda::counting_iterator<local_segment_index_t>>);
}

CUB_TEST("DeviceSegmentedSort: fixed-size environment", "[segmented_sort][fixed][env]", CUB_SMALL)
{
  const bool pairs         = GENERATE(false, true);
  const bool descending    = GENERATE(false, true);
  const bool double_buffer = GENERATE(false, true);
  c2h::device_vector<int> keys_in{8, 6, 7, 5, 3, 0};
  c2h::device_vector<int> keys_out(6);
  c2h::device_vector<int> values_in{0, 1, 2, 3, 4, 5};
  c2h::device_vector<int> values_out(6);
  auto ki = thrust::raw_pointer_cast(keys_in.data());
  auto ko = thrust::raw_pointer_cast(keys_out.data());
  auto vi = thrust::raw_pointer_cast(values_in.data());
  auto vo = thrust::raw_pointer_cast(values_out.data());
  cub::DoubleBuffer<int> keys(ki, ko);
  cub::DoubleBuffer<int> values(vi, vo);

  const auto check = [&](auto invoke, auto launch, auto... args) {
    size_t bytes{};
    REQUIRE(cudaSuccess == invoke(nullptr, cuda::std::ref(bytes), args...));
    const auto allocation = expected_allocation_size(bytes);
    SECTION("allocation and stream")
    {
      launch(args..., cuda::std::execution::env{allocation});
    }
#if TEST_LAUNCH == 0
    SECTION("default environment")
    {
      REQUIRE(cudaSuccess == invoke(args...));
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    }
#endif
#if TEST_LAUNCH != 1
    SECTION("explicit stream")
    {
      cuda::stream stream{cuda::devices[0]};
      launch(args..., cuda::std::execution::env{allocation, cuda::stream_ref{stream}});
    }
    SECTION("custom tuning")
    {
      // Requiring the actual kernel instantiated with this selector detects silently ignored tuning.
      void* kernels[] = {tuned_small_kernel<cub::SortOrder::Ascending, cub::NullType>(),
                         tuned_small_kernel<cub::SortOrder::Descending, cub::NullType>(),
                         tuned_small_kernel<cub::SortOrder::Ascending, int>(),
                         tuned_small_kernel<cub::SortOrder::Descending, int>()};
      auto env        = cuda::std::execution::env{
        allocation, cuda::execution::tune(fixed_sort_tuning{}), allowed_kernels(cuda::std::span<void*>{kernels})};
      launch(args..., env);
    }
#endif
  };

  if (pairs)
  {
    if (double_buffer)
    {
      if (descending)
      {
        check(fixed_pairs_descending_invocable_t{}, fixed_pairs_descending, keys, values, 3, 2);
      }
      else
      {
        check(fixed_pairs_invocable_t{}, fixed_pairs, keys, values, 3, 2);
      }
    }
    else
    {
      if (descending)
      {
        check(fixed_pairs_descending_invocable_t{}, fixed_pairs_descending, ki, ko, vi, vo, 3, 2);
      }
      else
      {
        check(fixed_pairs_invocable_t{}, fixed_pairs, ki, ko, vi, vo, 3, 2);
      }
    }
  }
  else if (double_buffer)
  {
    if (descending)
    {
      check(fixed_keys_descending_invocable_t{}, fixed_keys_descending, keys, 3, 2);
    }
    else
    {
      check(fixed_keys_invocable_t{}, fixed_keys, keys, 3, 2);
    }
  }
  else
  {
    if (descending)
    {
      check(fixed_keys_descending_invocable_t{}, fixed_keys_descending, ki, ko, 3, 2);
    }
    else
    {
      check(fixed_keys_invocable_t{}, fixed_keys, ki, ko, 3, 2);
    }
  }
  const c2h::host_vector<int> expected_keys =
    descending ? c2h::host_vector<int>{8, 7, 6, 5, 3, 0} : c2h::host_vector<int>{6, 7, 8, 0, 3, 5};
  REQUIRE(keys_out == expected_keys);
  if (pairs)
  {
    const c2h::host_vector<int> expected_values =
      descending ? c2h::host_vector<int>{0, 2, 1, 3, 4, 5} : c2h::host_vector<int>{1, 2, 0, 5, 4, 3};
    REQUIRE(values_out == expected_values);
  }
}
