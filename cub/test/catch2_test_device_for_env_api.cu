// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

// example-begin bulk-square-env-t
struct square_t
{
  int* d_ptr;

  __device__ void operator()(int i)
  {
    d_ptr[i] *= d_ptr[i];
  }
};
// example-end bulk-square-env-t

// example-begin square-ref-env-t
struct square_ref_t
{
  __device__ void operator()(int& i)
  {
    i *= i;
  }
};
// example-end square-ref-env-t

// example-begin odd-count-env-t
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
// example-end odd-count-env-t

C2H_TEST("cub::DeviceFor::Bulk env-based API", "[for][env]")
{
  // example-begin bulk-env
  auto vec = thrust::device_vector<int>{1, 2, 3, 4};
  square_t op{thrust::raw_pointer_cast(vec.data())};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceFor::Bulk(static_cast<int>(vec.size()), op, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFor::Bulk failed: " << error << std::endl;
  }

  thrust::device_vector<int> expected{1, 4, 9, 16};
  // example-end bulk-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(vec == expected);
}

C2H_TEST("cub::DeviceFor::ForEachN env-based API", "[for][env]")
{
  // example-begin for-each-n-env
  auto vec = thrust::device_vector<int>{1, 2, 3, 4};
  square_ref_t op{};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceFor::ForEachN(vec.begin(), static_cast<int>(vec.size()), op, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFor::ForEachN failed: " << error << std::endl;
  }

  thrust::device_vector<int> expected{1, 4, 9, 16};
  // example-end for-each-n-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(vec == expected);
}

C2H_TEST("cub::DeviceFor::ForEach env-based API", "[for][env]")
{
  // example-begin for-each-env
  auto vec = thrust::device_vector<int>{1, 2, 3, 4};
  square_ref_t op{};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceFor::ForEach(vec.begin(), vec.end(), op, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFor::ForEach failed: " << error << std::endl;
  }

  thrust::device_vector<int> expected{1, 4, 9, 16};
  // example-end for-each-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(vec == expected);
}

C2H_TEST("cub::DeviceFor::ForEachCopyN env-based API", "[for][env]")
{
  // example-begin for-each-copy-n-env
  auto vec   = thrust::device_vector<int>{1, 2, 3, 4};
  auto count = thrust::device_vector<int>(1);
  odd_count_t op{thrust::raw_pointer_cast(count.data())};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceFor::ForEachCopyN(vec.begin(), static_cast<int>(vec.size()), op, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFor::ForEachCopyN failed: " << error << std::endl;
  }

  thrust::device_vector<int> expected_count{2};
  // example-end for-each-copy-n-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(count == expected_count);
}

C2H_TEST("cub::DeviceFor::ForEachCopy env-based API", "[for][env]")
{
  // example-begin for-each-copy-env
  auto vec   = thrust::device_vector<int>{1, 2, 3, 4};
  auto count = thrust::device_vector<int>(1);
  odd_count_t op{thrust::raw_pointer_cast(count.data())};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceFor::ForEachCopy(vec.begin(), vec.end(), op, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFor::ForEachCopy failed: " << error << std::endl;
  }

  thrust::device_vector<int> expected_count{2};
  // example-end for-each-copy-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(count == expected_count);
}
