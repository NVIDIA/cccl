// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <c2h/catch2_test_helper.h>

// example-begin bulk-square-t
struct square_t
{
  int* d_ptr;

  __device__ void operator()(int i)
  {
    d_ptr[i] *= d_ptr[i];
  }
};
// example-end bulk-square-t

// example-begin bulk-square-ref-t
struct square_ref_t
{
  __device__ void operator()(int& i)
  {
    i *= i;
  }
};
// example-end bulk-square-ref-t

// example-begin bulk-odd-count-t
struct odd_count_t
{
  int* d_count;

  __device__ void operator()(int i)
  {
    if (i % 2 == 1)
    {
      atomicAdd(d_count, 1);
    }
  }
};
// example-end bulk-odd-count-t

C2H_TEST("Device bulk works with temporary storage", "[bulk][device]")
{
  // example-begin bulk-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_t op{thrust::raw_pointer_cast(vec.data())};

  // 1) Get temp storage size
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};
  cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, vec.size(), op);

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform bulk operation
  cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, vec.size(), op);

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end bulk-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device bulk works without temporary storage", "[bulk][device]")
{
  // example-begin bulk-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_t op{thrust::raw_pointer_cast(vec.data())};

  cub::DeviceFor::Bulk(vec.size(), op);

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end bulk-wo-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each n works with temporary storage", "[for_each][device]")
{
  // example-begin for-each-n-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  // 1) Get temp storage size
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};
  cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each n operation
  cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end for-each-n-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each n works without temporary storage", "[for_each][device]")
{
  // example-begin for-each-n-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  cub::DeviceFor::ForEachN(vec.begin(), vec.size(), op);

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end for-each-n-wo-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each works with temporary storage", "[for_each][device]")
{
  // example-begin for-each-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  // 1) Get temp storage size
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};
  cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each operation
  cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end for-each-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each works without temporary storage", "[for_each][device]")
{
  // example-begin for-each-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  cub::DeviceFor::ForEach(vec.begin(), vec.end(), op);

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end for-each-wo-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each n copy works with temporary storage", "[for_each][device]")
{
  // example-begin for-each-copy-n-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  c2h::device_vector<int> count(1);
  odd_count_t op{thrust::raw_pointer_cast(count.data())};

  // 1) Get temp storage size
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};
  cub::DeviceFor::ForEachCopyN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each n operation
  cub::DeviceFor::ForEachCopyN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);

  c2h::device_vector<int> expected = {2};
  // example-end for-each-copy-n-temp-storage

  REQUIRE(count == expected);
}

C2H_TEST("Device for each n copy works without temporary storage", "[for_each][device]")
{
  // example-begin for-each-copy-n-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  c2h::device_vector<int> count(1);
  odd_count_t op{thrust::raw_pointer_cast(count.data())};

  cub::DeviceFor::ForEachCopyN(vec.begin(), vec.size(), op);

  c2h::device_vector<int> expected = {2};
  // example-end for-each-copy-n-wo-temp-storage

  REQUIRE(count == expected);
}

C2H_TEST("Device for each copy works with temporary storage", "[for_each][device]")
{
  // example-begin for-each-copy-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  c2h::device_vector<int> count(1);
  odd_count_t op{thrust::raw_pointer_cast(count.data())};

  // 1) Get temp storage size
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};
  cub::DeviceFor::ForEachCopy(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each n operation
  cub::DeviceFor::ForEachCopy(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);

  c2h::device_vector<int> expected = {2};
  // example-end for-each-copy-temp-storage

  REQUIRE(count == expected);
}

C2H_TEST("Device for each copy works without temporary storage", "[for_each][device]")
{
  // example-begin for-each-copy-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  c2h::device_vector<int> count(1);
  odd_count_t op{thrust::raw_pointer_cast(count.data())};

  cub::DeviceFor::ForEachCopy(vec.begin(), vec.end(), op);

  c2h::device_vector<int> expected = {2};
  // example-end for-each-copy-wo-temp-storage

  REQUIRE(count == expected);
}
