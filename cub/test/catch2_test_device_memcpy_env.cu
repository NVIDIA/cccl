// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_memcpy.cuh>
#include <cub/device/dispatch/tuning/tuning_batch_memcpy.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
#include <cuda/iterator>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceMemcpy::Batched, device_memcpy_batched);

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
    return (offsets[index + 1] - offsets[index]) * static_cast<int>(sizeof(int));
  }
};

#if TEST_LAUNCH == 0

TEST_CASE("DeviceMemcpy::Batched works with default environment", "[memcpy][device]")
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  int num_buffers = 3;

  cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = cuda::transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  REQUIRE(cudaSuccess == cub::DeviceMemcpy::Batched(input_it, output_it, sizes, num_buffers));

  REQUIRE(d_dst == d_src);
}

#endif

C2H_TEST("DeviceMemcpy::Batched uses environment", "[memcpy][device]")
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  int num_buffers = 3;

  cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = cuda::transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceMemcpy::Batched(nullptr, expected_bytes_allocated, input_it, output_it, sizes, num_buffers));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_memcpy_batched(input_it, output_it, sizes, num_buffers, env);

  REQUIRE(d_dst == d_src);
}

TEST_CASE("DeviceMemcpy::Batched uses custom stream", "[memcpy][device]")
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  int num_buffers = 3;

  cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = cuda::transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  cuda::stream custom_stream(cuda::device_ref{0});

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceMemcpy::Batched(nullptr, expected_bytes_allocated, input_it, output_it, sizes, num_buffers));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_memcpy_batched(input_it, output_it, sizes, num_buffers, env);

  custom_stream.sync();
  REQUIRE(d_dst == d_src);
}

template <int BlockThreads>
struct batch_memcpy_tuning
{
  _CCCL_API constexpr auto operator()(cuda::compute_capability /*cc*/) const
    -> cub::detail::batch_memcpy::batch_memcpy_policy
  {
    return {
      {BlockThreads, 4, 8, false, 256 * 32, 128, 8 * 1024, {}, {}},
      {256, 32},
    };
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

C2H_TEST("DeviceMemcpy::Batched can be tuned", "[memcpy][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  // 3 buffers of 2 ints each (8 bytes)
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 4, 6};

  int num_buffers                = 3;
  constexpr int bytes_per_buffer = 2 * static_cast<int>(sizeof(int));

  cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});

  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_constant_iterator sizes(bytes_per_buffer, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(batch_memcpy_tuning<target_block_size>{});

  device_memcpy_batched(input_it, output_it, sizes, num_buffers, env);

  REQUIRE(d_dst == d_src);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1
