// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

// example-begin radix-sort-keys-custom-decomposer
struct custom_key_t
{
  int key;
};

struct keys_decomposer_t
{
  __host__ __device__ auto operator()(custom_key_t& k) const -> ::cuda::std::tuple<int&>
  {
    return {k.key};
  }
};
// example-end radix-sort-keys-custom-decomposer

__host__ __device__ bool operator==(const custom_key_t& a, const custom_key_t& b)
{
  return a.key == b.key;
}

__host__ __device__ bool operator!=(const custom_key_t& a, const custom_key_t& b)
{
  return a.key != b.key;
}

std::ostream& operator<<(std::ostream& os, const custom_key_t& ck)
{
  return os << "{" << ck.key << "}";
}

// example-begin radix-sort-pairs-custom-decomposer
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

struct pairs_decomposer_t
{
  __host__ __device__ auto operator()(custom_pair_key_t& k) const -> ::cuda::std::tuple<int&>
  {
    return {k.key};
  }
};
// example-end radix-sort-pairs-custom-decomposer

C2H_TEST("cub::DeviceRadixSort::SortPairs env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-env
  auto keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = thrust::device_vector<int>(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);

  auto error = cub::DeviceRadixSort::SortPairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()));

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairs failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  thrust::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  // example-end radix-sort-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortPairsDescending env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-descending-env
  auto keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = thrust::device_vector<int>(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);

  auto error = cub::DeviceRadixSort::SortPairsDescending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()));

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairsDescending failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  thrust::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  // example-end radix-sort-pairs-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortKeys env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);

  auto error = cub::DeviceRadixSort::SortKeys(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), 0, sizeof(int) * 8);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeys failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end radix-sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
}

C2H_TEST("cub::DeviceRadixSort::SortKeys DoubleBuffer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-db-env
  thrust::device_vector<int> keys_buf0{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> keys_buf1(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  auto error = cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(keys_buf0.size()), 0, sizeof(int) * 8);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeys (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end radix-sort-keys-db-env

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}

C2H_TEST("cub::DeviceRadixSort::SortKeysDescending env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-descending-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);

  auto error = cub::DeviceRadixSort::SortKeysDescending(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), 0, sizeof(int) * 8);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeysDescending failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  // example-end radix-sort-keys-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
}

C2H_TEST("cub::DeviceRadixSort::SortKeysDescending DoubleBuffer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-descending-db-env
  thrust::device_vector<int> keys_buf0{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> keys_buf1(7);

  cub::DoubleBuffer<int> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  auto error = cub::DeviceRadixSort::SortKeysDescending(d_keys, static_cast<int>(keys_buf0.size()), 0, sizeof(int) * 8);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortKeysDescending (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  // example-end radix-sort-keys-descending-db-env

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected_keys);
}

C2H_TEST("cub::DeviceRadixSort::SortPairs decomposer with bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-decomposer-bits-env
  auto keys_in    = thrust::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = thrust::device_vector<custom_pair_key_t>(3);
  auto values_in  = thrust::device_vector<int>{0, 1, 2};
  auto values_out = thrust::device_vector<int>(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    pairs_decomposer_t{},
    0,
    sizeof(int) * 8,
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairs (decomposer+bits) failed with status: " << error << '\n';
  }
  // example-end radix-sort-pairs-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  thrust::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortPairs decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-decomposer-env
  auto keys_in    = thrust::device_vector<custom_pair_key_t>{{3, 100}, {1, 200}, {2, 300}};
  auto keys_out   = thrust::device_vector<custom_pair_key_t>(3);
  auto values_in  = thrust::device_vector<int>{0, 1, 2};
  auto values_out = thrust::device_vector<int>(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    pairs_decomposer_t{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairs (decomposer) failed with status: " << error << '\n';
  }
  // example-end radix-sort-pairs-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys_out == expected_keys);
  thrust::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortPairs DoubleBuffer decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-db-decomposer-env
  thrust::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  thrust::device_vector<custom_pair_key_t> keys_buf1(3);
  thrust::device_vector<int> values_buf0{0, 1, 2};
  thrust::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairs(
    d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairs (DB decomposer) failed with status: " << error << '\n';
  }
  // example-end radix-sort-pairs-db-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  thrust::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  thrust::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortPairs DoubleBuffer decomposer with bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-db-decomposer-bits-env
  thrust::device_vector<custom_pair_key_t> keys_buf0{{3, 100}, {1, 200}, {2, 300}};
  thrust::device_vector<custom_pair_key_t> keys_buf1(3);
  thrust::device_vector<int> values_buf0{0, 1, 2};
  thrust::device_vector<int> values_buf1(3);

  cub::DoubleBuffer<custom_pair_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairs(
    d_keys, d_values, static_cast<int>(keys_buf0.size()), pairs_decomposer_t{}, 0, sizeof(int) * 8, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRadixSort::SortPairs (DB decomposer+bits) failed with status: " << error << '\n';
  }
  // example-end radix-sort-pairs-db-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  thrust::device_vector<custom_pair_key_t> expected_keys{{1, 200}, {2, 300}, {3, 100}};
  REQUIRE(keys == expected_keys);
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  thrust::device_vector<int> expected_values{1, 2, 0};
  REQUIRE(values == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortKeys decomposer+bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-decomposer-bits-env
  thrust::device_vector<custom_key_t> keys_in{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_out(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortKeys(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    0,
    sizeof(int) * 8,
    stream_ref);

  thrust::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  // example-end radix-sort-keys-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortKeys decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-decomposer-env
  thrust::device_vector<custom_key_t> keys_in{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_out(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortKeys(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}, stream_ref);

  thrust::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  // example-end radix-sort-keys-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortKeys DB decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-db-decomposer-env
  thrust::device_vector<custom_key_t> keys_buf0{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_buf1(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error =
    cub::DeviceRadixSort::SortKeys(d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, stream_ref);

  thrust::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  // example-end radix-sort-keys-db-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortKeys DB decomposer+bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-db-decomposer-bits-env
  thrust::device_vector<custom_key_t> keys_buf0{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_buf1(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortKeys(
    d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, sizeof(int) * 8, stream_ref);

  thrust::device_vector<custom_key_t> expected{{0}, {3}, {5}, {6}, {7}, {8}, {9}};
  // example-end radix-sort-keys-db-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortKeysDescending decomposer+bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-descending-decomposer-bits-env
  thrust::device_vector<custom_key_t> keys_in{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_out(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortKeysDescending(
    keys_in.data().get(),
    keys_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    0,
    sizeof(int) * 8,
    stream_ref);

  thrust::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  // example-end radix-sort-keys-descending-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortKeysDescending decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-descending-decomposer-env
  thrust::device_vector<custom_key_t> keys_in{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_out(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortKeysDescending(
    keys_in.data().get(), keys_out.data().get(), static_cast<int>(keys_in.size()), keys_decomposer_t{}, stream_ref);

  thrust::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  // example-end radix-sort-keys-descending-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortKeysDescending DB decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-descending-db-decomposer-env
  thrust::device_vector<custom_key_t> keys_buf0{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_buf1(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortKeysDescending(
    d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, stream_ref);

  thrust::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  // example-end radix-sort-keys-descending-db-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortKeysDescending DB decomposer+bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-keys-descending-db-decomposer-bits-env
  thrust::device_vector<custom_key_t> keys_buf0{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_buf1(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortKeysDescending(
    d_keys, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, sizeof(int) * 8, stream_ref);

  thrust::device_vector<custom_key_t> expected{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  // example-end radix-sort-keys-descending-db-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  REQUIRE(keys == expected);
}

C2H_TEST("cub::DeviceRadixSort::SortPairsDescending decomposer+bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-descending-decomposer-bits-env
  thrust::device_vector<custom_key_t> keys_in{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_out(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairsDescending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    0,
    sizeof(int) * 8,
    stream_ref);

  thrust::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  thrust::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  // example-end radix-sort-pairs-descending-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortPairsDescending decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-descending-decomposer-env
  thrust::device_vector<custom_key_t> keys_in{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_out(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairsDescending(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    keys_decomposer_t{},
    stream_ref);

  thrust::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  thrust::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  // example-end radix-sort-pairs-descending-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortPairsDescending DB decomposer env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-descending-db-decomposer-env
  thrust::device_vector<custom_key_t> keys_buf0{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_buf1(7);
  thrust::device_vector<int> values_buf0{0, 1, 2, 3, 4, 5, 6};
  thrust::device_vector<int> values_buf1(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairsDescending(
    d_keys, d_values, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, stream_ref);

  thrust::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  thrust::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  // example-end radix-sort-pairs-descending-db-decomposer-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}

C2H_TEST("cub::DeviceRadixSort::SortPairsDescending DB decomposer+bits env-based API", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-descending-db-decomposer-bits-env
  thrust::device_vector<custom_key_t> keys_buf0{{8}, {6}, {7}, {5}, {3}, {0}, {9}};
  thrust::device_vector<custom_key_t> keys_buf1(7);
  thrust::device_vector<int> values_buf0{0, 1, 2, 3, 4, 5, 6};
  thrust::device_vector<int> values_buf1(7);

  cub::DoubleBuffer<custom_key_t> d_keys(keys_buf0.data().get(), keys_buf1.data().get());
  cub::DoubleBuffer<int> d_values(values_buf0.data().get(), values_buf1.data().get());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceRadixSort::SortPairsDescending(
    d_keys, d_values, static_cast<int>(keys_buf0.size()), keys_decomposer_t{}, 0, sizeof(int) * 8, stream_ref);

  thrust::device_vector<custom_key_t> expected_keys{{9}, {8}, {7}, {6}, {5}, {3}, {0}};
  thrust::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  // example-end radix-sort-pairs-descending-db-decomposer-bits-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  auto& keys   = d_keys.selector == 0 ? keys_buf0 : keys_buf1;
  auto& values = d_values.selector == 0 ? values_buf0 : values_buf1;
  REQUIRE(keys == expected_keys);
  REQUIRE(values == expected_values);
}
