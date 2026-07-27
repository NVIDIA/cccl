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

#include <sstream>

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
  _CCCL_API constexpr auto operator()(cuda::compute_capability /*cc*/) const -> cub::BatchedCopyPolicy
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

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
C2H_TEST("Test BatchedCopyPolicy properties", "[memcpy][device]")
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::BatchedCopyPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::BatchedCopyPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::BatchedCopySmallBufferPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::BatchedCopySmallBufferPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::BatchedCopyLargeBufferPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::BatchedCopyLargeBufferPolicy>);

  // aggregate init
  constexpr auto p1_small = cub::BatchedCopySmallBufferPolicy{
    128,
    4,
    8,
    false,
    256 * 32,
    128,
    8 * 1024,
    cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 350, 450},
    cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  constexpr auto p1_large = cub::BatchedCopyLargeBufferPolicy{256, 32};
  constexpr auto p1       = cub::BatchedCopyPolicy{p1_small, p1_large};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2_small = cub::BatchedCopySmallBufferPolicy{
    .threads_per_block     = 128,
    .buffers_per_thread    = 4,
    .bytes_per_thread      = 8,
    .prefer_pow2_bits      = false,
    .block_level_tile_size = 256 * 32,
    .warp_level_threshold  = 128,
    .block_level_threshold = 8 * 1024,
    .buffer_lookback_delay =
      cub::LookbackDelayPolicy{.kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 350, .l2_write_latency = 450},
    .block_lookback_delay = cub::LookbackDelayPolicy{
      .kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 350, .l2_write_latency = 450}};
  constexpr auto p2_large = cub::BatchedCopyLargeBufferPolicy{.threads_per_block = 256, .bytes_per_thread = 32};
  constexpr auto p2       = cub::BatchedCopyPolicy{.small_buffer = p2_small, .large_buffer = p2_large};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2_small = p1_small;
  constexpr auto p2_large = p1_large;
  constexpr auto p2       = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1_small == p2_small);
  STATIC_REQUIRE_FALSE(p1_small != p2_small);

  STATIC_REQUIRE(p1_large == p2_large);
  STATIC_REQUIRE_FALSE(p1_large != p2_large);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(
    to_string(p1_small)
    == "BatchedCopySmallBufferPolicy { .threads_per_block = 128, .buffers_per_thread = 4"
       ", .bytes_per_thread = 8, .prefer_pow2_bits = 0, .block_level_tile_size = 8192"
       ", .warp_level_threshold = 128, .block_level_threshold = 8192"
       ", .buffer_lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 350, .l2_write_latency = 450 }"
       ", .block_lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 350, .l2_write_latency = 450 } }");
  REQUIRE(to_string(p1_large) == "BatchedCopyLargeBufferPolicy { .threads_per_block = 256, .bytes_per_thread = 32 }");
  REQUIRE(
    to_string(p1)
    == "BatchedCopyPolicy { .small_buffer = BatchedCopySmallBufferPolicy { .threads_per_block = 128"
       ", .buffers_per_thread = 4, .bytes_per_thread = 8, .prefer_pow2_bits = 0"
       ", .block_level_tile_size = 8192, .warp_level_threshold = 128, .block_level_threshold = 8192"
       ", .buffer_lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 350, .l2_write_latency = 450 }"
       ", .block_lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 350, .l2_write_latency = 450 } }"
       ", .large_buffer = BatchedCopyLargeBufferPolicy { .threads_per_block = 256"
       ", .bytes_per_thread = 32 } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
