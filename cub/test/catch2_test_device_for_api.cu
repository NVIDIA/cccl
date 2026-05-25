// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/iterator>
#include <cuda/std/tuple>

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

struct assign_zip_value_t
{
  template <class Tuple>
  __device__ void operator()(Tuple tuple) const
  {
    cuda::std::get<1>(tuple) = cuda::std::get<0>(tuple);
  }
};

template <class OutputIt>
struct tabulate_output_op
{
  OutputIt output;

  __device__ void operator()(int idx, int value) const
  {
    output[idx] = value;
  }
};

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
  auto result = cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "Bulk operation failed with error code: " << result << '\n';
  }

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end bulk-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device bulk works without temporary storage", "[bulk][device]")
{
  // example-begin bulk-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_t op{thrust::raw_pointer_cast(vec.data())};

  auto result = cub::DeviceFor::Bulk(vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "Bulk operation failed with error code: " << result << '\n';
  }

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
  auto result = cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachN operation failed with error code: " << result << '\n';
  }

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each n operation
  result = cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachN operation failed with error code: " << result << '\n';
  }

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end for-each-n-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each n works without temporary storage", "[for_each][device]")
{
  // example-begin for-each-n-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  auto result = cub::DeviceFor::ForEachN(vec.begin(), vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachN operation failed with error code: " << result << '\n';
  }

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end for-each-n-wo-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each n works with a tabulate output iterator in a thrust zip iterator", "[for_each][device]")
{
  c2h::device_vector<int> input = {1, 2, 3, 4};
  c2h::device_vector<int> output(input.size());

  auto output_it = cuda::tabulate_output_iterator{tabulate_output_op<decltype(output.begin())>{output.begin()}};
  auto zipped_it = thrust::make_zip_iterator(input.begin(), output_it);

  auto result = cub::DeviceFor::ForEachN(zipped_it, input.size(), assign_zip_value_t{});
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachN operation failed with error code: " << result << '\n';
  }

  REQUIRE(output == input);
}

C2H_TEST("Device for each n works with a tabulate output iterator in a cuda zip iterator", "[for_each][device]")
{
  c2h::device_vector<int> input = {1, 2, 3, 4};
  c2h::device_vector<int> output(input.size());

  auto output_it = cuda::tabulate_output_iterator{tabulate_output_op<decltype(output.begin())>{output.begin()}};
  auto zipped_it = cuda::make_zip_iterator(input.begin(), output_it);

  auto result = cub::DeviceFor::ForEachN(zipped_it, input.size(), assign_zip_value_t{});
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachN operation failed with error code: " << result << '\n';
  }

  REQUIRE(output == input);
}

C2H_TEST("Device for each works with temporary storage", "[for_each][device]")
{
  // example-begin for-each-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  // 1) Get temp storage size
  std::uint8_t* d_temp_storage{};
  std::size_t temp_storage_bytes{};
  auto result = cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEach operation failed with error code: " << result << '\n';
  }

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each operation
  result = cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEach operation failed with error code: " << result << '\n';
  }

  c2h::device_vector<int> expected = {1, 4, 9, 16};
  // example-end for-each-temp-storage

  REQUIRE(vec == expected);
}

C2H_TEST("Device for each works without temporary storage", "[for_each][device]")
{
  // example-begin for-each-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  auto result = cub::DeviceFor::ForEach(vec.begin(), vec.end(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEach operation failed with error code: " << result << '\n';
  }

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
  auto result = cub::DeviceFor::ForEachCopyN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachCopyN operation failed with error code: " << result << '\n';
  }

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each n operation
  result = cub::DeviceFor::ForEachCopyN(d_temp_storage, temp_storage_bytes, vec.begin(), vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachCopyN operation failed with error code: " << result << '\n';
  }

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

  auto result = cub::DeviceFor::ForEachCopyN(vec.begin(), vec.size(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachCopyN operation failed with error code: " << result << '\n';
  }

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
  auto result = cub::DeviceFor::ForEachCopy(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachCopy operation failed with error code: " << result << '\n';
  }

  // 2) Allocate temp storage
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // 3) Perform for each n operation
  result = cub::DeviceFor::ForEachCopy(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachCopy operation failed with error code: " << result << '\n';
  }

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

  auto result = cub::DeviceFor::ForEachCopy(vec.begin(), vec.end(), op);
  if (result != cudaSuccess)
  {
    std::cerr << "ForEachCopy operation failed with error code: " << result << '\n';
  }

  c2h::device_vector<int> expected = {2};
  // example-end for-each-copy-wo-temp-storage

  REQUIRE(count == expected);
}

// Guard tests: each public DeviceFor method must resolve unambiguously
// to the legacy temp-storage overload when called in its minimal form
// (no explicit stream, all defaults left implicit), even though the env
// and bare-stream overloads are also in scope. If the env-overload
// SFINAE is wrong, these become "ambiguous overload" compile errors.

struct noop_t
{
  __device__ void operator()(int) const {}
};

struct noop_ref_t
{
  __device__ void operator()(int&) const {}
};

C2H_TEST("DeviceFor::Bulk legacy size-query is unambiguous", "[for][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int n                     = 0;

  REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, n, noop_t{}));
}

C2H_TEST("DeviceFor::ForEachN legacy size-query is unambiguous", "[for][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int n                     = 0;

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, d_in, n, noop_ref_t{}));
}

C2H_TEST("DeviceFor::ForEach legacy size-query is unambiguous", "[for][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_first              = nullptr;
  int* d_last               = nullptr;

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, d_first, d_last, noop_ref_t{}));
}

C2H_TEST("DeviceFor::ForEachCopyN legacy size-query is unambiguous", "[for][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_in                 = nullptr;
  int n                     = 0;

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopyN(d_temp_storage, temp_storage_bytes, d_in, n, noop_t{}));
}

C2H_TEST("DeviceFor::ForEachCopy legacy size-query is unambiguous", "[for][device]")
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_first              = nullptr;
  int* d_last               = nullptr;

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopy(d_temp_storage, temp_storage_bytes, d_first, d_last, noop_t{}));
}

C2H_TEST("Device for each n fails when using stream from another device", "[for_each][device]")
{
  int num_devices = 0;
  REQUIRE(cudaGetDeviceCount(&num_devices) == cudaSuccess);

  if (num_devices < 2)
  {
    SKIP("Test requires at least 2 CUDA devices");
  }

  REQUIRE(cudaSetDevice(1) == cudaSuccess);
  cudaStream_t stream_on_device_1;
  REQUIRE(cudaStreamCreate(&stream_on_device_1) == cudaSuccess);
  REQUIRE(cudaSetDevice(0) == cudaSuccess);

  // example-begin for-each-n-wo-temp-storage
  c2h::device_vector<int> vec = {1, 2, 3, 4};
  square_ref_t op{};

  auto result = cub::DeviceFor::ForEachN(vec.begin(), vec.size(), op, stream_on_device_1);
  REQUIRE(result == cudaErrorInvalidDevice);
  // example-end for-each-n-wo-temp-storage

  REQUIRE(cudaStreamDestroy(stream_on_device_1) == cudaSuccess);

}

// todo(giannis): extents/layout guards once a default-constructible 0-extent is wired up
