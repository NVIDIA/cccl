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

  __host__ __device__ friend bool operator==(const custom_key_t& a, const custom_key_t& b)
  {
    return a.key == b.key;
  }

  __host__ __device__ friend bool operator!=(const custom_key_t& a, const custom_key_t& b)
  {
    return a.key != b.key;
  }

  friend std::ostream& operator<<(std::ostream& os, const custom_key_t& ck)
  {
    return os << "{" << ck.key << "}";
  }
};

struct custom_pair_key_t
{
  int key;
  int payload;

  __host__ __device__ friend bool operator==(const custom_pair_key_t& a, const custom_pair_key_t& b)
  {
    return a.key == b.key && a.payload == b.payload;
  }

  __host__ __device__ friend bool operator!=(const custom_pair_key_t& a, const custom_pair_key_t& b)
  {
    return !(a == b);
  }

  friend std::ostream& operator<<(std::ostream& os, const custom_pair_key_t& cpk)
  {
    return os << "{" << cpk.key << ", " << cpk.payload << "}";
  }
};

struct keys_decomposer_t
{
  __host__ __device__ auto operator()(custom_key_t& k) const -> ::cuda::std::tuple<int&>
  {
    return {k.key};
  }
};

struct pairs_decomposer_t
{
  __host__ __device__ auto operator()(custom_pair_key_t& k) const -> ::cuda::std::tuple<int&>
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

TEST_CASE("Device radix sort keys decomposer+bits works with default environment", "[radix_sort][device]")
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

  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys decomposer works with default environment", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}));

  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys DB decomposer works with default environment", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(
    cudaSuccess == cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort keys DB decomposer+bits works with default environment", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8)));

  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort pairs decomposer works with default environment", "[radix_sort][device]")
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

  c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs decomposer with bits works with default environment", "[radix_sort][device]")
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

  c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort keys descending decomposer+bits works with default environment", "[radix_sort][device]")
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

  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys descending decomposer works with default environment", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}));

  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys descending DB decomposer works with default environment", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}));

  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort keys descending DB decomposer+bits works with default environment", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8)));

  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort pairs descending decomposer+bits works with default environment", "[radix_sort][device]")
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

  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs descending decomposer works with default environment", "[radix_sort][device]")
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

  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs descending DB decomposer works with default environment", "[radix_sort][device]")
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

  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

TEST_CASE("Device radix sort pairs descending DB decomposer+bits works with default environment",
          "[radix_sort][device]")
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

  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
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
  auto keys_in    = c2h::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = c2h::device_vector<custom_pair_key_t>(3);
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
      pairs_decomposer_t{},
      stream_ref));

  stream.sync();

  c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

#if TEST_LAUNCH == 0

TEST_CASE("Device radix sort pairs DB decomposer works with default environment", "[radix_sort][device]")
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
  c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

TEST_CASE("Device radix sort pairs DB decomposer with bits works with default environment", "[radix_sort][device]")
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
  c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

#endif

TEST_CASE("Device radix sort pairs DB decomposer uses custom stream", "[radix_sort][device]")
{
  c2h::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  c2h::device_vector<custom_pair_key_t> keys_buf1(3);
  c2h::device_vector<int> values_buf0{0, 1, 2};
  c2h::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairs(
            d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}, stream_ref));

  stream.sync();

  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  c2h::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  c2h::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

TEST_CASE("Device radix sort keys decomposer+bits uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8),
      env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys decomposer uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}, env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys DB decomposer uses custom stream", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort keys DB decomposer+bits uses custom stream", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeys(
      d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8), env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort keys descending decomposer+bits uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      keys_in.data().get(),
      keys_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8),
      env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys descending decomposer uses custom stream", "[radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out = c2h::device_vector<custom_key_t>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeysDescending(
            keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}, env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  REQUIRE(keys_out == expected);
}

TEST_CASE("Device radix sort keys descending DB decomposer uses custom stream", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort keys descending DB decomposer+bits uses custom stream", "[radix_sort][device]")
{
  auto keys_buf0 = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1 = c2h::device_vector<custom_key_t>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortKeysDescending(
      d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, static_cast<int>(sizeof(int) * 8), env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

TEST_CASE("Device radix sort pairs descending decomposer+bits uses custom stream", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

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
      static_cast<int>(sizeof(int) * 8),
      env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs descending decomposer uses custom stream", "[radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_out   = c2h::device_vector<custom_key_t>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      static_cast<int>(keys_in.size()),
      keys_decomposer_t{},
      env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("Device radix sort pairs descending DB decomposer uses custom stream", "[radix_sort][device]")
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortPairsDescending(
            d_keys, d_values, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

TEST_CASE("Device radix sort pairs descending DB decomposer+bits uses custom stream", "[radix_sort][device]")
{
  auto keys_buf0   = c2h::device_vector<custom_key_t>{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  auto keys_buf1   = c2h::device_vector<custom_key_t>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  REQUIRE(
    cudaSuccess
    == cub::DeviceRadixSort::SortPairsDescending(
      d_keys,
      d_values,
      static_cast<int>(keys_buf0.size()),
      keys_decomposer_t{},
      0,
      static_cast<int>(sizeof(int) * 8),
      env));

  stream.sync();
  c2h::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

#if TEST_LAUNCH == 0

// Radix sort does not accept user-provided functors or iterators, so we cannot use the block_size_extracting_op or
// block_size_extracting_constant_iterator approach. Instead, we verify tuning by measuring allocation sizes: different
// block sizes produce different temporary storage requirements, which is an observable side effect of the tuning being
// applied.
template <typename KeyT, typename ValueT, int BlockThreads>
struct tiny_onesweep_policy_selector
{
  _CCCL_API constexpr auto operator()(cuda::compute_capability cc) const -> cub::detail::radix_sort::radix_sort_policy
  {
    using default_selector_t               = cub::detail::radix_sort::policy_selector_from_types<KeyT, ValueT, int>;
    auto policy                            = default_selector_t{}(cc);
    policy.use_onesweep                    = true;
    policy.onesweep.threads_per_block      = BlockThreads;
    policy.onesweep.items_per_thread       = 1;
    policy.single_tile.threads_per_block   = BlockThreads;
    policy.single_tile.items_per_thread    = 1;
    policy.downsweep.threads_per_block     = BlockThreads;
    policy.downsweep.items_per_thread      = 1;
    policy.alt_downsweep.threads_per_block = BlockThreads;
    policy.alt_downsweep.items_per_thread  = 1;
    policy.histogram.num_parts             = 1;
    return policy;
  }
};

template <typename CallableT, typename PolicySelector>
std::size_t measure_allocated_bytes(CallableT&& run, PolicySelector policy_selector)
{
  cuda::stream_ref stream{cudaStream_t{}};
  size_t bytes_allocated   = 0;
  size_t bytes_deallocated = 0;
  auto env                 = stdexec::env{device_memory_resource{stream.get(), &bytes_allocated, &bytes_deallocated},
                          stream,
                          cuda::execution::tune(policy_selector)};
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

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairs DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortPairs(double_buf, double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
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

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairsDescending DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortPairsDescending(double_buf, double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeys can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    return cub::DeviceRadixSort::SortKeys(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeys DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortKeys(double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    return cub::DeviceRadixSort::SortKeysDescending(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending DoubleBuffer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    cub::DoubleBuffer<int> double_buf(data.data().get(), data.data().get());
    return cub::DeviceRadixSort::SortKeysDescending(double_buf, static_cast<int>(data.size()), 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

// decomposer variants: keys

TEST_CASE("DeviceRadixSort::SortKeys decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<custom_key_t>(10'000);
    return cub::DeviceRadixSort::SortKeys(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), keys_decomposer_t{}, 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeys decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<custom_key_t>(10'000);
    return cub::DeviceRadixSort::SortKeys(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), keys_decomposer_t{}, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeys DB decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    return cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeys DB decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    return cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

// decomposer variants: keys descending

TEST_CASE("DeviceRadixSort::SortKeysDescending decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<custom_key_t>(10'000);
    return cub::DeviceRadixSort::SortKeysDescending(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), keys_decomposer_t{}, 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto data = c2h::device_vector<custom_key_t>(10'000);
    return cub::DeviceRadixSort::SortKeysDescending(
      data.data().get(), data.data().get(), static_cast<int>(data.size()), keys_decomposer_t{}, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending DB decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    return cub::DeviceRadixSort::SortKeysDescending(d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortKeysDescending DB decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    return cub::DeviceRadixSort::SortKeysDescending(
      d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

// decomposer variants: pairs

TEST_CASE("DeviceRadixSort::SortPairs decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    return cub::DeviceRadixSort::SortPairs(
      d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairs decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto keys = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vals = c2h::device_vector<int>(10'000);
    return cub::DeviceRadixSort::SortPairs(
      keys.data().get(),
      keys.data().get(),
      vals.data().get(),
      vals.data().get(),
      static_cast<int>(keys.size()),
      pairs_decomposer_t{},
      env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairs DB decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    return cub::DeviceRadixSort::SortPairs(d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairs DB decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    return cub::DeviceRadixSort::SortPairs(
      d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

// decomposer variants: pairs descending

TEST_CASE("DeviceRadixSort::SortPairsDescending decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto keys = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vals = c2h::device_vector<int>(10'000);
    return cub::DeviceRadixSort::SortPairsDescending(
      keys.data().get(),
      keys.data().get(),
      vals.data().get(),
      vals.data().get(),
      static_cast<int>(keys.size()),
      pairs_decomposer_t{},
      0,
      32,
      env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairsDescending decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto keys = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vals = c2h::device_vector<int>(10'000);
    return cub::DeviceRadixSort::SortPairsDescending(
      keys.data().get(),
      keys.data().get(),
      vals.data().get(),
      vals.data().get(),
      static_cast<int>(keys.size()),
      pairs_decomposer_t{},
      env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairsDescending DB decomposer can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    return cub::DeviceRadixSort::SortPairsDescending(
      d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

TEST_CASE("DeviceRadixSort::SortPairsDescending DB decomposer+bits can be tuned", "[radix_sort][device]")
{
  auto l = [&](auto env) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    return cub::DeviceRadixSort::SortPairsDescending(
      d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, 0, 32, env);
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

#endif // TEST_LAUNCH != 1
