// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairs, device_radix_sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortPairsDescending, device_radix_sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeys, device_radix_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeysDescending, device_radix_sort_keys_descending);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

struct custom_key_t
{
  int key;
  int payload;
};

struct custom_decomposer_t
{
  __host__ __device__ auto operator()(custom_key_t& k) const -> ::cuda::std::tuple<int&>
  {
    return {k.key};
  }
};

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

TEST_CASE("Device radix sort pairs decomposer works with default environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_key_t>(3);
  auto values_in  = c2h::device_vector<int>{0, 1, 2};
  auto values_out = c2h::device_vector<int>(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      custom_decomposer_t{}));

  c2h::host_vector<custom_key_t> h_keys_out(keys_out);
  REQUIRE(h_keys_out[0].key == 1);
  REQUIRE(h_keys_out[1].key == 2);
  REQUIRE(h_keys_out[2].key == 3);
  c2h::host_vector<int> h_values_out(values_out);
  REQUIRE(h_values_out[0] == 1);
  REQUIRE(h_values_out[1] == 2);
  REQUIRE(h_values_out[2] == 0);
}

TEST_CASE("Device radix sort pairs decomposer with bits works with default environment", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_key_t>(3);
  auto values_in  = c2h::device_vector<int>{0, 1, 2};
  auto values_out = c2h::device_vector<int>(3);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      custom_decomposer_t{},
      0,
      sizeof(int) * 8));

  c2h::host_vector<custom_key_t> h_keys_out(keys_out);
  REQUIRE(h_keys_out[0].key == 1);
  REQUIRE(h_keys_out[1].key == 2);
  REQUIRE(h_keys_out[2].key == 3);
  c2h::host_vector<int> h_values_out(values_out);
  REQUIRE(h_values_out[0] == 1);
  REQUIRE(h_values_out[1] == 2);
  REQUIRE(h_values_out[2] == 0);
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

  cuda::stream custom_stream{cuda::devices[0]};

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

  cuda::stream_ref stream_ref{custom_stream};
  auto env = stdexec::env{stream_ref, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  custom_stream.sync();

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs descending uses custom stream", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  cuda::stream custom_stream{cuda::devices[0]};

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

  cuda::stream_ref stream_ref{custom_stream};
  auto env = stdexec::env{stream_ref, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  custom_stream.sync();

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort keys uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  cuda::stream custom_stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      nullptr, expected_bytes_allocated, keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size())));

  cuda::stream_ref stream_ref{custom_stream};
  auto env = stdexec::env{stream_ref, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(static_cast<int>(sizeof(int) * 8)),
    env);

  custom_stream.sync();
  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(keys_out == expected_keys);
}

TEST_CASE("Device radix sort keys descending uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  cuda::stream custom_stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      nullptr, expected_bytes_allocated, keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size())));

  cuda::stream_ref stream_ref{custom_stream};
  auto env = stdexec::env{stream_ref, expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    static_cast<int>(static_cast<int>(sizeof(int) * 8)),
    env);

  custom_stream.sync();
  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  REQUIRE(keys_out == expected_keys);
}

TEST_CASE("Device radix sort pairs decomposer uses custom stream", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_key_t>(3);
  auto values_in  = c2h::device_vector<int>{0, 1, 2};
  auto values_out = c2h::device_vector<int>(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      custom_decomposer_t{},
      stream_ref));

  stream.sync();

  c2h::host_vector<custom_key_t> h_keys_out(keys_out);
  REQUIRE(h_keys_out[0].key == 1);
  REQUIRE(h_keys_out[1].key == 2);
  REQUIRE(h_keys_out[2].key == 3);
  c2h::host_vector<int> h_values_out(values_out);
  REQUIRE(h_values_out[0] == 1);
  REQUIRE(h_values_out[1] == 2);
  REQUIRE(h_values_out[2] == 0);
}

#if TEST_LAUNCH == 0

TEST_CASE("Device radix sort pairs DB decomposer works with default environment", "[radix_sort][device]")
{
  c2h::device_vector<custom_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(d_keys, d_values, static_cast<int>(keys_buf0.size()), custom_decomposer_t{}));

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  c2h::host_vector<custom_key_t> h_keys(keys);
  REQUIRE(h_keys[0].key == 1);
  REQUIRE(h_keys[1].key == 2);
  REQUIRE(h_keys[2].key == 3);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  c2h::host_vector<int> h_values(values);
  REQUIRE(h_values[0] == 1);
  REQUIRE(h_values[1] == 2);
  REQUIRE(h_values[2] == 0);
}

TEST_CASE("Device radix sort pairs DB decomposer with bits works with default environment", "[radix_sort][device]")
{
  c2h::device_vector<custom_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairs(
            d_keys, d_values, static_cast<int>(keys_buf0.size()), custom_decomposer_t{}, 0, sizeof(int) * 8));

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  c2h::host_vector<custom_key_t> h_keys(keys);
  REQUIRE(h_keys[0].key == 1);
  REQUIRE(h_keys[1].key == 2);
  REQUIRE(h_keys[2].key == 3);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  c2h::host_vector<int> h_values(values);
  REQUIRE(h_values[0] == 1);
  REQUIRE(h_values[1] == 2);
  REQUIRE(h_values[2] == 0);
}

#endif

TEST_CASE("Device radix sort pairs DB decomposer uses custom stream", "[radix_sort][device]")
{
  c2h::device_vector<custom_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairs(
            d_keys, d_values, static_cast<int>(keys_buf0.size()), custom_decomposer_t{}, stream_ref));

  stream.sync();

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  c2h::host_vector<custom_key_t> h_keys(keys);
  REQUIRE(h_keys[0].key == 1);
  REQUIRE(h_keys[1].key == 2);
  REQUIRE(h_keys[2].key == 3);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  c2h::host_vector<int> h_values(values);
  REQUIRE(h_values[0] == 1);
  REQUIRE(h_values[1] == 2);
  REQUIRE(h_values[2] == 0);
}
