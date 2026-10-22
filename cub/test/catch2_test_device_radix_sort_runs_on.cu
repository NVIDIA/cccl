// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_device.cuh>
#include <cub/util_runs_on.cuh>

#include <cuda/execution>
#include <cuda/std/tuple>

#include <cstdint>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// Note: no TEST_LAUNCH=1, runs-on isn't meant for device (because it's obvious which device
// you are running on).
//
// %PARAM% TEST_LAUNCH lid 0:2

CUB_TEST("Device radix sort keys with runs_on sorts correctly", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  int device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device));
  int sm_count{};
  REQUIRE(cudaSuccess == cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  const auto provide_description = GENERATE(false, true);
  CAPTURE(provide_description);
  const auto runs_on =
    provide_description
      ? cub::experimental::RunsOn{cub::experimental::DeviceDescription{cc, static_cast<std::uint32_t>(sm_count)}}
      : cub::experimental::RunsOn{};
  const auto env = cuda::execution::guarantee(runs_on);

  c2h::device_vector<std::int32_t> keys_in(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> keys_out(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, false);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      const auto error      = cub::DeviceRadixSort::SortKeys(
        storage, bytes, keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, launch_env);

      if (storage == nullptr)
      {
        REQUIRE(cudaSuccess == error);

        size_t direct_bytes = 0;
        REQUIRE(
          cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            nullptr,
            direct_bytes,
            keys_in.data().get(),
            keys_out.data().get(),
            num_items,
            0,
            32,
            cuda::stream_ref{stream}));

        CAPTURE(bytes, direct_bytes);
        REQUIRE(bytes >= direct_bytes);
        // RunsOn should always provide a worst-case storage estimate.
        REQUIRE(bytes <= 5 * direct_bytes);

        if (!provide_description)
        {
          REQUIRE(bytes == direct_bytes);
        }
      }

      return error;
    });
    REQUIRE(keys_out == expected);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, true);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      const auto error      = cub::DeviceRadixSort::SortKeysDescending(
        storage, bytes, keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, launch_env);

      if (storage == nullptr)
      {
        REQUIRE(cudaSuccess == error);

        size_t direct_bytes = 0;
        REQUIRE(
          cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            nullptr,
            direct_bytes,
            keys_in.data().get(),
            keys_out.data().get(),
            num_items,
            0,
            32,
            cuda::stream_ref{stream}));

        CAPTURE(bytes, direct_bytes);
        REQUIRE(bytes >= direct_bytes);
        // RunsOn should always provide a worst-case storage estimate.
        REQUIRE(bytes <= 5 * direct_bytes);

        if (!provide_description)
        {
          REQUIRE(bytes == direct_bytes);
        }
      }

      return error;
    });
    REQUIRE(keys_out == expected);
  }
}

CUB_TEST("Device radix sort keys with runs_on handles an SM limit", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const auto env = cuda::execution::guarantee(cub::experimental::RunsOn{cub::experimental::DeviceDescription{cc, 1}});

  c2h::device_vector<std::int32_t> keys_in(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> keys_out(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, false);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      return cub::DeviceRadixSort::SortKeys(
        storage, bytes, keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, launch_env);
    });
    REQUIRE(keys_out == expected);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, true);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      return cub::DeviceRadixSort::SortKeysDescending(
        storage, bytes, keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, launch_env);
    });
    REQUIRE(keys_out == expected);
  }
}

CUB_TEST("Device radix sort pairs with runs_on sorts correctly", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  int device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device));
  int sm_count{};
  REQUIRE(cudaSuccess == cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  const auto provide_description = GENERATE(false, true);
  CAPTURE(provide_description);
  const auto runs_on =
    provide_description
      ? cub::experimental::RunsOn{cub::experimental::DeviceDescription{cc, static_cast<std::uint32_t>(sm_count)}}
      : cub::experimental::RunsOn{};
  const auto env = cuda::execution::guarantee(runs_on);

  c2h::device_vector<std::int32_t> keys_in(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> keys_out(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> values_in(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> values_out(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in, std::int32_t{-10}, std::int32_t{10});
  c2h::gen(C2H_SEED(1), values_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, false);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      const auto error      = cub::DeviceRadixSort::SortPairs(
        storage,
        bytes,
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        0,
        32,
        launch_env);

      if (storage == nullptr)
      {
        REQUIRE(cudaSuccess == error);

        size_t direct_bytes = 0;
        REQUIRE(
          cudaSuccess
          == cub::DeviceRadixSort::SortPairs(
            nullptr,
            direct_bytes,
            keys_in.data().get(),
            keys_out.data().get(),
            values_in.data().get(),
            values_out.data().get(),
            num_items,
            0,
            32,
            cuda::stream_ref{stream}));

        CAPTURE(bytes, direct_bytes);
        REQUIRE(bytes >= direct_bytes);
        // RunsOn should always provide a worst-case storage estimate.
        REQUIRE(bytes <= 5 * direct_bytes);

        if (!provide_description)
        {
          REQUIRE(bytes == direct_bytes);
        }
      }

      return error;
    });
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, true);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      const auto error      = cub::DeviceRadixSort::SortPairsDescending(
        storage,
        bytes,
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        0,
        32,
        launch_env);

      if (storage == nullptr)
      {
        REQUIRE(cudaSuccess == error);

        size_t direct_bytes = 0;
        REQUIRE(
          cudaSuccess
          == cub::DeviceRadixSort::SortPairsDescending(
            nullptr,
            direct_bytes,
            keys_in.data().get(),
            keys_out.data().get(),
            values_in.data().get(),
            values_out.data().get(),
            num_items,
            0,
            32,
            cuda::stream_ref{stream}));

        CAPTURE(bytes, direct_bytes);
        REQUIRE(bytes >= direct_bytes);
        // RunsOn should always provide a worst-case storage estimate.
        REQUIRE(bytes <= 5 * direct_bytes);

        if (!provide_description)
        {
          REQUIRE(bytes == direct_bytes);
        }
      }

      return error;
    });
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }
}

CUB_TEST("Device radix sort pairs with runs_on handles an SM limit", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const auto env = cuda::execution::guarantee(cub::experimental::RunsOn{cub::experimental::DeviceDescription{cc, 1}});

  c2h::device_vector<std::int32_t> keys_in(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> keys_out(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> values_in(num_items, thrust::no_init);
  c2h::device_vector<std::int32_t> values_out(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys_in, std::int32_t{-10}, std::int32_t{10});
  c2h::gen(C2H_SEED(1), values_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, false);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      return cub::DeviceRadixSort::SortPairs(
        storage,
        bytes,
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        0,
        32,
        launch_env);
    });
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, true);
    launch([&](std::uint8_t* storage, size_t& bytes, cudaStream_t stream = nullptr) {
      const auto launch_env = cuda::std::execution::env{cuda::stream_ref{stream}, env};
      return cub::DeviceRadixSort::SortPairsDescending(
        storage,
        bytes,
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        0,
        32,
        launch_env);
    });
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }
}

#if TEST_LAUNCH == 0
CUB_TEST("Device radix sort queries storage with a mismatched runs_on capability",
         "[radix_sort][device][runs_on]",
         CUB_SMALL)
{
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));

  const auto wrong_cc = [&] {
    for (auto&& target : cuda::__target_compute_capabilities())
    {
      if (target != cc)
      {
        return target;
      }
    }
    SKIP("Test requires building with more than one compute-capability");
  }();

  CAPTURE(cc.get());
  CAPTURE(wrong_cc.get());

  const auto env =
    cuda::execution::guarantee(cub::experimental::RunsOn{cub::experimental::DeviceDescription{wrong_cc, 1}});
  c2h::device_vector<std::int32_t> keys_in{3, 1, 2};
  c2h::device_vector<std::int32_t> keys_out(3);

  size_t bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys_in.data().get(), keys_out.data().get(), 3, 0, 32, env));
  REQUIRE(bytes > 0);
}
#endif // TEST_LAUNCH == 0
