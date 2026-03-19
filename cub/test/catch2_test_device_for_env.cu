// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

struct square_ref_op
{
  __device__ void operator()(int& i)
  {
    i *= i;
  }
};

struct square_idx_op
{
  int* d_ptr;

  __device__ void operator()(int i)
  {
    d_ptr[i] *= d_ptr[i];
  }
};

struct odd_count_op
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

// -----------------------------------------------------------------------
// Bulk
// -----------------------------------------------------------------------

C2H_TEST("DeviceFor::Bulk env uses custom stream", "[for][env]")
{
  auto vec = c2h::device_vector<int>{1, 2, 3, 4};
  square_idx_op op{thrust::raw_pointer_cast(vec.data())};

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}};

  auto error = cub::DeviceFor::Bulk(4, op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  c2h::device_vector<int> expected{1, 4, 9, 16};
  REQUIRE(vec == expected);
}

// -----------------------------------------------------------------------
// ForEachN
// -----------------------------------------------------------------------

C2H_TEST("DeviceFor::ForEachN env uses custom stream", "[for][env]")
{
  auto vec = c2h::device_vector<int>{1, 2, 3, 4};
  square_ref_op op{};

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}};

  auto error = cub::DeviceFor::ForEachN(vec.begin(), static_cast<int>(vec.size()), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  c2h::device_vector<int> expected{1, 4, 9, 16};
  REQUIRE(vec == expected);
}

// -----------------------------------------------------------------------
// ForEach
// -----------------------------------------------------------------------

C2H_TEST("DeviceFor::ForEach env uses custom stream", "[for][env]")
{
  auto vec = c2h::device_vector<int>{1, 2, 3, 4};
  square_ref_op op{};

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}};

  auto error = cub::DeviceFor::ForEach(vec.begin(), vec.end(), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  c2h::device_vector<int> expected{1, 4, 9, 16};
  REQUIRE(vec == expected);
}

// -----------------------------------------------------------------------
// ForEachCopyN
// -----------------------------------------------------------------------

C2H_TEST("DeviceFor::ForEachCopyN env uses custom stream", "[for][env]")
{
  auto vec   = c2h::device_vector<int>{1, 2, 3, 4};
  auto count = c2h::device_vector<int>(1);
  odd_count_op op{thrust::raw_pointer_cast(count.data())};

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}};

  auto error = cub::DeviceFor::ForEachCopyN(vec.begin(), static_cast<int>(vec.size()), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  c2h::device_vector<int> expected_count{2};
  REQUIRE(count == expected_count);
}

// -----------------------------------------------------------------------
// ForEachCopy
// -----------------------------------------------------------------------

C2H_TEST("DeviceFor::ForEachCopy env uses custom stream", "[for][env]")
{
  auto vec   = c2h::device_vector<int>{1, 2, 3, 4};
  auto count = c2h::device_vector<int>(1);
  odd_count_op op{thrust::raw_pointer_cast(count.data())};

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}};

  auto error = cub::DeviceFor::ForEachCopy(vec.begin(), vec.end(), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  c2h::device_vector<int> expected_count{2};
  REQUIRE(count == expected_count);
}
