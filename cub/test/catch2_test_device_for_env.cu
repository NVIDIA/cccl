// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
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

template <int BlockThreads>
struct for_each_tuning
{
  _CCCL_API constexpr auto operator()(cuda::arch_id /*arch*/) const -> cub::detail::for_each::for_policy
  {
    return {BlockThreads, 2};
  }
};

struct block_size_extracting_op
{
  unsigned int* block_size;

  __device__ void operator()(int) const
  {
    if (threadIdx.x == 0)
    {
      atomicMax(block_size, blockDim.x);
    }
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

C2H_TEST("DeviceFor::Bulk can be tuned", "[for][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(4, op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceFor::ForEachN can be tuned", "[for][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(d_data.begin(), static_cast<int>(d_data.size()), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceFor::ForEach can be tuned", "[for][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(d_data.begin(), d_data.end(), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceFor::ForEachCopyN can be tuned", "[for][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopyN(d_data.begin(), static_cast<int>(d_data.size()), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceFor::ForEachCopy can be tuned", "[for][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopy(d_data.begin(), d_data.end(), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}
