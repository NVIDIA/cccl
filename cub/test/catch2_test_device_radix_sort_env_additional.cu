// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <cuda/std/execution>
#include <cuda/stream>

#include "catch2_radix_sort_env_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/device_and_stream.h>
#include <c2h/vector.h>

// %PARAM% TEST_LAUNCH lid 0:2

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device radix sort pairs works with default environment", "[radix_sort][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending works with default environment", "[radix_sort][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort keys works with default environment", "[radix_sort][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};

  REQUIRE(keys_out == expected_keys);
}

CUB_TEST_CASE("Device radix sort keys descending works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size())));

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};

  REQUIRE(keys_out == expected_keys);
}

CUB_TEST_CASE("Device radix sort keys decomposer+bits works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys decomposer works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}));

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys DB decomposer works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(
    cudaSuccess == cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort keys DB decomposer+bits works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8)));

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort pairs decomposer works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_pair_key_t>(3);
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
      pairs_decomposer_t{}));

  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs decomposer+bits works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_pair_key_t>(3);
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
      pairs_decomposer_t{},
      0,
      sizeof(int) * 8));

  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort keys descending decomposer+bits works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys descending decomposer works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}));

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys descending DB decomposer works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort keys descending DB decomposer+bits works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8)));

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort pairs descending decomposer+bits works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8)));

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending decomposer works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{}));

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending DB decomposer works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairsDescending(
            d_keys, d_values, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending DB decomposer+bits works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      d_keys, d_values, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8)));

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs DB decomposer works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  c2h::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_pair_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairs(d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}));

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs DB decomposer+bits works with default environment",
              "[radix_sort][device]",
              CUB_SMALL)
{
  c2h::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_pair_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairs(
            d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}, 0, sizeof(int) * 8));

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs DB works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortPairs(d_keys, d_values, static_cast<int>(keys_buf0.size())));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending DB works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  REQUIRE(
    cudaSuccess == cub::DeviceRadixSort::SortPairsDescending(d_keys, d_values, static_cast<int>(keys_buf0.size())));

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort keys DB works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(keys_buf0.size())));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}

CUB_TEST_CASE("Device radix sort keys descending DB works with default environment", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortKeysDescending(d_keys, static_cast<int>(keys_buf0.size())));

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}

#endif // TEST_LAUNCH == 0

// Host streams are not usable in device-side launches.
#if TEST_LAUNCH != 1

// Reference captures preserve DoubleBuffer selectors across the launch helper's argument copies.
CUB_TEST_CASE("Device radix sort pairs uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairs(
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        0,
        static_cast<int>(sizeof(int) * 8),
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairsDescending(
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        0,
        static_cast<int>(sizeof(int) * 8),
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort keys uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeys(
        keys_in.data().get(),
        keys_out.data().get(),
        num_items,
        0,
        static_cast<int>(static_cast<int>(sizeof(int) * 8)),
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};

  REQUIRE(keys_out == expected_keys);
}

CUB_TEST_CASE("Device radix sort keys descending uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeysDescending(
        keys_in.data().get(), keys_out.data().get(), num_items, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};

  REQUIRE(keys_out == expected_keys);
}

CUB_TEST_CASE("Device radix sort keys decomposer+bits uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeys(
        keys_in.data().get(),
        keys_out.data().get(),
        num_items,
        keys_decomposer_t{},
        0,
        static_cast<int>(sizeof(int) * 8),
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeys(
        keys_in.data().get(), keys_out.data().get(), num_items, keys_decomposer_t{}, env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys DB decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeys(d_keys, num_items, keys_decomposer_t{}, env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort keys DB decomposer+bits uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeys(
        d_keys, num_items, keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort pairs decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_pair_key_t>(3);
  auto values_in  = c2h::device_vector<int>{0, 1, 2};
  auto values_out = c2h::device_vector<int>(3);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairs(
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        pairs_decomposer_t{},
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs decomposer+bits uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_pair_key_t>(3);
  auto values_in  = c2h::device_vector<int>{0, 1, 2};
  auto values_out = c2h::device_vector<int>(3);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairs(
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        pairs_decomposer_t{},
        0,
        sizeof(int) * 8,
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort keys descending decomposer+bits uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeysDescending(
        keys_in.data().get(),
        keys_out.data().get(),
        num_items,
        keys_decomposer_t{},
        0,
        static_cast<int>(sizeof(int) * 8),
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys descending decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeysDescending(
        keys_in.data().get(), keys_out.data().get(), num_items, keys_decomposer_t{}, env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

CUB_TEST_CASE("Device radix sort keys descending DB decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeysDescending(d_keys, num_items, keys_decomposer_t{}, env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort keys descending DB decomposer+bits uses custom stream",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeysDescending(
        d_keys, num_items, keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

CUB_TEST_CASE("Device radix sort pairs descending decomposer+bits uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairsDescending(
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        keys_decomposer_t{},
        0,
        static_cast<int>(sizeof(int) * 8),
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairsDescending(
        keys_in.data().get(),
        keys_out.data().get(),
        values_in.data().get(),
        values_out.data().get(),
        num_items,
        keys_decomposer_t{},
        env);
    },
    static_cast<int>(keys_in.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending DB decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairsDescending(d_keys, d_values, num_items, keys_decomposer_t{}, env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending DB decomposer+bits uses custom stream",
              "[radix_sort][device]",
              CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairsDescending(
        d_keys, d_values, num_items, keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs DB decomposer uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  c2h::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_pair_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairs(d_keys, d_values, num_items, pairs_decomposer_t{}, env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs DB decomposer+bits uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  c2h::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_pair_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairs(d_keys, d_values, num_items, pairs_decomposer_t{}, 0, sizeof(int) * 8, env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  const c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  const c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs DB uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairs(d_keys, d_values, num_items, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort pairs descending DB uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortPairsDescending(
        d_keys, d_values, num_items, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(values == expected_values);
}

CUB_TEST_CASE("Device radix sort keys DB uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeys(d_keys, num_items, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}

CUB_TEST_CASE("Device radix sort keys descending DB uses custom stream", "[radix_sort][device]", CUB_SMALL)
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};

  launch_env(
    [&](int num_items, auto env) {
      return cub::DeviceRadixSort::SortKeysDescending(d_keys, num_items, 0, static_cast<int>(sizeof(int) * 8), env);
    },
    static_cast<int>(keys_buf0.size()),
    stream_ref);

  stream.sync();

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}

#endif // TEST_LAUNCH != 1
