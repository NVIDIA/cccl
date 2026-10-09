// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <cuda/std/execution>

#include <cstddef>

#include "catch2_radix_sort_env_helper.cuh"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortPairs, device_radix_sort_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortPairsDescending, device_radix_sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortKeys, device_radix_sort_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortKeysDescending, device_radix_sort_keys_descending);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

CUB_TEST("Device radix sort keys decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

CUB_TEST("Device radix sort keys decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}, env);

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

CUB_TEST("Device radix sort keys DB decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            nullptr, expected_bytes_allocated, d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, env);
}

CUB_TEST("Device radix sort keys DB decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<int>(keys_buf0.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(
    d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8), env);
}

CUB_TEST("Device radix sort pairs decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_pair_key_t>(3);
  auto values_in  = c2h::device_vector<int>{0, 1, 2};
  auto values_out = c2h::device_vector<int>(3);

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
      static_cast<int>(keys_in.size()),
      pairs_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    pairs_decomposer_t{},
    env);

  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

CUB_TEST("Device radix sort pairs decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_pair_key_t>(3);
  auto values_in  = c2h::device_vector<int>{0, 1, 2};
  auto values_out = c2h::device_vector<int>(3);

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
      static_cast<int>(keys_in.size()),
      pairs_decomposer_t{},
      0,
      sizeof(int) * 8));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    pairs_decomposer_t{},
    0,
    sizeof(int) * 8,
    env);

  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

CUB_TEST("Device radix sort keys descending decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

CUB_TEST("Device radix sort keys descending decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}, env);

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

CUB_TEST("Device radix sort keys descending DB decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            nullptr, expected_bytes_allocated, d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, env);
}

CUB_TEST("Device radix sort keys descending DB decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<int>(keys_buf0.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(
    d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8), env);
}

CUB_TEST("Device radix sort pairs descending decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
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
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("Device radix sort pairs descending decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
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
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    env);

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("Device radix sort pairs descending DB decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      nullptr, expected_bytes_allocated, d_keys, d_values, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(d_keys, d_values, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, env);
}

CUB_TEST("Device radix sort pairs descending DB decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<int>(keys_buf0.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    keys_decomposer_t{},
    0,
    static_cast<int>(sizeof(int) * 8),
    env);
}

CUB_TEST("Device radix sort pairs DB decomposer uses environment", "[radix_sort][device]", CUB_SMALL)
{
  c2h::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_pair_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      nullptr, expected_bytes_allocated, d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}, env);
}

CUB_TEST("Device radix sort pairs DB decomposer+bits uses environment", "[radix_sort][device]", CUB_SMALL)
{
  c2h::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_pair_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<int>(keys_buf0.size()),
      pairs_decomposer_t{},
      0,
      sizeof(int) * 8));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}, 0, sizeof(int) * 8, env);
}

CUB_TEST("Device radix sort pairs DB uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairs(
            nullptr, expected_bytes_allocated, d_keys, d_values, static_cast<int>(keys_buf0.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs(
    d_keys, d_values, static_cast<int>(keys_buf0.size()), 0, static_cast<int>(sizeof(int) * 8), env);
}

CUB_TEST("Device radix sort pairs descending DB uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairsDescending(
            nullptr, expected_bytes_allocated, d_keys, d_values, static_cast<int>(keys_buf0.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_pairs_descending(
    d_keys, d_values, static_cast<int>(keys_buf0.size()), 0, static_cast<int>(sizeof(int) * 8), env);
}

CUB_TEST("Device radix sort keys DB uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(nullptr, expected_bytes_allocated, d_keys, static_cast<int>(keys_buf0.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys(d_keys, static_cast<int>(keys_buf0.size()), 0, static_cast<int>(sizeof(int) * 8), env);
}

CUB_TEST("Device radix sort keys descending DB uses environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            nullptr, expected_bytes_allocated, d_keys, static_cast<int>(keys_buf0.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_radix_sort_keys_descending(
    d_keys, static_cast<int>(keys_buf0.size()), 0, static_cast<int>(sizeof(int) * 8), env);
}
