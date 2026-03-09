// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairs, device_radix_sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairsDescending, device_radix_sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeys, device_radix_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeysDescending, device_radix_sort_keys_descending);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("Device radix sort pairs works with default environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairs(
            keys_in.data().get(),
            keys_out.data().get(),
            values_in.data().get(),
            values_out.data().get(),
            static_cast<int>(keys_in.size())));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs descending works with default environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairsDescending(
            keys_in.data().get(),
            keys_out.data().get(),
            values_in.data().get(),
            values_out.data().get(),
            static_cast<int>(keys_in.size())));

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort keys works with default environment", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            keys_in.data().get(),
            keys_out.data().get(),
            static_cast<int>(keys_in.size()),
            0,
            static_cast<int>(static_cast<int>(sizeof(int) * 8))));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};

  REQUIRE(keys_out == expected_keys);
}

TEST_CASE("Device radix sort keys descending works with default environment", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size())));

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};

  REQUIRE(keys_out == expected_keys);
}

#endif

C2H_TEST("Device radix sort pairs uses environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  // calculate expected_bytes_allocated - call CUB API directly, not through wrapper
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("Device radix sort pairs descending uses environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("Device radix sort keys uses environment", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      nullptr, expected_bytes_allocated, keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(static_cast<int>(sizeof(int) * 8)),
    env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};

  REQUIRE(keys_out == expected_keys);
}

C2H_TEST("Device radix sort keys descending uses environment", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      0,
      static_cast<int>(static_cast<int>(sizeof(int) * 8))));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(static_cast<int>(sizeof(int) * 8)),
    env);

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};

  REQUIRE(keys_out == expected_keys);
}

TEST_CASE("Device radix sort pairs uses custom stream", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size())));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device radix sort pairs descending uses custom stream", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size())));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device radix sort keys uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      nullptr, expected_bytes_allocated, keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size())));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(static_cast<int>(sizeof(int) * 8)),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));
  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("Device radix sort keys descending uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      nullptr, expected_bytes_allocated, keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size())));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(static_cast<int>(sizeof(int) * 8)),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));
  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

// using different block sizes yields to different temporary storage sizes, so use a custom policy to influence that
template <typename KeyT, typename ValueT, int BlockThreads>
struct tiny_onesweep_policy_selector
{
  _CCCL_API constexpr auto operator()(cuda::arch_id arch) const -> cub::detail::radix_sort::radix_sort_policy
  {
    using default_selector_t         = cub::detail::radix_sort::policy_selector_from_types<KeyT, ValueT, int>;
    auto policy                      = default_selector_t{}(arch);
    policy.use_onesweep              = true;
    policy.onesweep.block_threads    = BlockThreads;
    policy.onesweep.items_per_thread = 1;
    return policy;
  }
};

template <typename CallableT, typename PolicySelector>
std::size_t measure_allocated_bytes(CallableT&& run, PolicySelector policy_selector)
{
  cuda::stream_ref stream{cudaStream_t{}};
  size_t bytes_allocated   = 0;
  size_t bytes_deallocated = 0;
  auto env                 = stdexec::env{
    cuda::std::execution::prop{cuda::mr::__get_memory_resource_t{},
                               device_memory_resource{{}, stream.get(), &bytes_allocated, &bytes_deallocated}},
    cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}},
    cuda::execution::__tune(policy_selector)};
  REQUIRE(cudaSuccess == run(env));
  stream.sync();
  CHECK(bytes_allocated > 0);
  CHECK(bytes_allocated == bytes_deallocated);
  return bytes_allocated;
}

TEST_CASE("DeviceRadixSort::SortPairs can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    return cub::DeviceRadixSort::SortPairs(
      data.data().get(),
      data.data().get(),
      data.data().get(),
      data.data().get(),
      static_cast<int>(data.size()),
      0,
      32,
      env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 64>{});

  CHECK(default_bytes != tuned_bytes);
}

TEST_CASE("DeviceRadixSort::SortPairs DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortPairs(double_buf, double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 64>{});

  CHECK(default_bytes != tuned_bytes);
}

TEST_CASE("DeviceRadixSort::SortPairsDescending can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    return cub::DeviceRadixSort::SortPairsDescending(
      data.data().get(),
      data.data().get(),
      data.data().get(),
      data.data().get(),
      static_cast<int>(data.size()),
      0,
      32,
      env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 64>{});

  CHECK(default_bytes != tuned_bytes);
}

TEST_CASE("DeviceRadixSort::SortPairsDescending DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortPairsDescending(double_buf, double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 64>{});

  CHECK(default_bytes != tuned_bytes);
}

TEST_CASE("DeviceRadixSort::SortKeys can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    return cub::DeviceRadixSort::SortKeys(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), 0, 32, env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 64>{});

  CHECK(default_bytes != tuned_bytes);
}

TEST_CASE("DeviceRadixSort::SortKeys DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortKeys(double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 64>{});

  CHECK(default_bytes != tuned_bytes);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    return cub::DeviceRadixSort::SortKeysDescending(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), 0, 32, env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 64>{});

  CHECK(default_bytes != tuned_bytes);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortKeysDescending(double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  auto default_bytes = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  auto tuned_bytes   = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 64>{});

  CHECK(default_bytes != tuned_bytes);
}
