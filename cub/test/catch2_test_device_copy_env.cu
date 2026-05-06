// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_copy.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceCopy::Batched, device_copy_batched);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

template <typename T>
struct index_to_ptr
{
  T* base;
  const int* offsets;
  __host__ __device__ __forceinline__ T* operator()(int index) const
  {
    return base + offsets[index];
  }
};

struct get_size
{
  const int* offsets;
  __host__ __device__ __forceinline__ int operator()(int index) const
  {
    return offsets[index + 1] - offsets[index];
  }
};

#if TEST_LAUNCH == 0

TEST_CASE("DeviceCopy::Batched works with default environment", "[copy][device]")
{
  // 3 ranges: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  int num_ranges = 3;

  thrust::counting_iterator<int> iota(0);
  auto input_it = thrust::make_transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = thrust::make_transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = thrust::make_transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  REQUIRE(cudaSuccess == cub::DeviceCopy::Batched(input_it, output_it, sizes, num_ranges));

  REQUIRE(d_dst == d_src);
}

#endif // TEST_LAUNCH == 0

C2H_TEST("DeviceCopy::Batched uses environment", "[copy][device]")
{
  // 3 ranges: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  int num_ranges = 3;

  thrust::counting_iterator<int> iota(0);
  auto input_it = thrust::make_transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = thrust::make_transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = thrust::make_transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess == cub::DeviceCopy::Batched(nullptr, expected_bytes_allocated, input_it, output_it, sizes, num_ranges));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_copy_batched(input_it, output_it, sizes, num_ranges, env);

  REQUIRE(d_dst == d_src);
}

TEST_CASE("DeviceCopy::Batched uses custom stream", "[copy][device]")
{
  // 3 ranges: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  int num_ranges = 3;

  thrust::counting_iterator<int> iota(0);
  auto input_it = thrust::make_transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = thrust::make_transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = thrust::make_transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess == cub::DeviceCopy::Batched(nullptr, expected_bytes_allocated, input_it, output_it, sizes, num_ranges));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_copy_batched(input_it, output_it, sizes, num_ranges, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));
  REQUIRE(d_dst == d_src);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}
