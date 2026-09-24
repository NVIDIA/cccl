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

#include <cuda/__execution/policy.h>
#include <cuda/__execution/tune.h>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_launch_helper.h"
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMemcpy::Batched, device_memcpy_batched);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>

#include "cub_test_macros.h"

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

CUB_TEST_CASE("DeviceMemcpy::Batched works with default environment", "[memcpy][device]", CUB_SMALL)
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  const int num_buffers = 3;

  const cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = cuda::transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  REQUIRE(cudaSuccess == cub::DeviceMemcpy::Batched(input_it, output_it, sizes, num_buffers));

  REQUIRE(d_dst == d_src);
}

#endif

CUB_TEST("DeviceMemcpy::Batched uses environment", "[memcpy][device]", CUB_SMALL)
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  const int num_buffers = 3;

  const cuda::counting_iterator<int> iota(0);
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

CUB_TEST_CASE("DeviceMemcpy::Batched uses custom stream", "[memcpy][device]", CUB_SMALL)
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  const int num_buffers = 3;

  const cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = cuda::transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  const cuda::stream custom_stream = c2h::make_current_device_stream();

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
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability /*cc*/) const -> cub::BatchedCopyPolicy
  {
    return {
      cub::BatchedCopyAlgorithm::lookback,
      {
        {BlockThreads, 4, 8, false, 256 * 32, 128, 8 * 1024, {}, {}},
        {256, 32},
      },
    };
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

CUB_TEST("DeviceMemcpy::Batched can be tuned", "[memcpy][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  // 3 buffers of 2 ints each (8 bytes)
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 4, 6};

  const int num_buffers          = 3;
  constexpr int bytes_per_buffer = 2 * static_cast<int>(sizeof(int));

  const cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});

  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_constant_iterator sizes(bytes_per_buffer, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(batch_memcpy_tuning<target_block_size>{});

  device_memcpy_batched(input_it, output_it, sizes, num_buffers, env);

  REQUIRE(d_dst == d_src);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

#if TEST_LAUNCH == 0

// The two-phase overload takes the same environment as the single-phase one but never allocates, so it does not go
// through the launch wrappers and would run identically in every TEST_LAUNCH variant. Test it with host launch only.

// Runs the two-phase overload wrapped by two_phase once per supported kind of environment: queries the temporary
// storage size, executes with user provided storage, and checks that every kernel is launched on the stream carried
// by the environment (or on the default stream if the environment carries none).
template <class TwoPhaseFn>
void test_two_phase_env_kinds(size_t expected_temp_storage_bytes, TwoPhaseFn two_phase)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref default_stream{cudaStream_t{}};

  auto run = [&](const auto& env, cuda::stream_ref expected_stream) {
    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes, env));
    REQUIRE(temp_storage_bytes == expected_temp_storage_bytes);

    c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
    {
      // stream_registry_factory_t fails the test if a kernel is launched on any other stream
      const stream_scope scope{expected_stream.get()};
      REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, env));
    }
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    expected_stream.sync();
  };

  SECTION("default environment")
  {
    run(stdexec::env<>{}, default_stream);
  }

  SECTION("cudaStream_t")
  {
    run(stream.get(), cuda::stream_ref{stream});
  }

  SECTION("cuda::stream")
  {
    run(stream, cuda::stream_ref{stream});
  }

  SECTION("cuda::stream_ref")
  {
    run(cuda::stream_ref{stream}, cuda::stream_ref{stream});
  }

  SECTION("environment with stream")
  {
    run(stdexec::env{cuda::stream_ref{stream}}, cuda::stream_ref{stream});
  }

  SECTION("cuda::execution::gpu")
  {
    run(cuda::execution::gpu, default_stream);
  }

  SECTION("cuda::execution::gpu with stream")
  {
    run(cuda::execution::gpu.with(cuda::get_stream, cuda::stream_ref{stream}), cuda::stream_ref{stream});
  }
}

CUB_TEST_CASE("DeviceMemcpy::Batched works with user provided memory and environment", "[memcpy][device]", CUB_SMALL)
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  const int num_buffers = 3;

  const cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = cuda::transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  size_t expected_bytes{};
  REQUIRE(cudaSuccess == cub::DeviceMemcpy::Batched(nullptr, expected_bytes, input_it, output_it, sizes, num_buffers));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMemcpy::Batched(d_temp_storage, temp_storage_bytes, input_it, output_it, sizes, num_buffers, env);
  });

  REQUIRE(d_dst == d_src);
}

// Before the environment parameter, the two-phase overload took `cudaStream_t stream = nullptr`, so callers passing
// nullptr or a literal 0 for the stream exist. Both must keep compiling and keep running on the default stream.
CUB_TEST_CASE("DeviceMemcpy::Batched two-phase overload accepts legacy null stream arguments",
              "[memcpy][device]",
              CUB_SMALL)
{
  // 3 buffers: [10, 20], [30, 40, 50], [60]
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 5, 6};

  const int num_buffers = 3;

  const cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = cuda::transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  auto memcpy_batched_on = [&](const auto& stream) {
    size_t temp_storage_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceMemcpy::Batched(nullptr, temp_storage_bytes, input_it, output_it, sizes, num_buffers, stream));

    c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
    const stream_scope scope{cudaStream_t{}};
    REQUIRE(
      cudaSuccess
      == cub::DeviceMemcpy::Batched(
        thrust::raw_pointer_cast(temp_storage.data()),
        temp_storage_bytes,
        input_it,
        output_it,
        sizes,
        num_buffers,
        stream));
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  };

  SECTION("nullptr")
  {
    memcpy_batched_on(nullptr);
  }

  SECTION("literal 0")
  {
    memcpy_batched_on(0);
  }

  REQUIRE(d_dst == d_src);
}

CUB_TEST("DeviceMemcpy::Batched can be tuned with user provided memory", "[memcpy][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  // 3 buffers of 2 ints each (8 bytes)
  auto d_src     = c2h::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = c2h::device_vector<int>(6, 0);
  auto d_offsets = c2h::device_vector<int>{0, 2, 4, 6};

  const int num_buffers          = 3;
  constexpr int bytes_per_buffer = 2 * static_cast<int>(sizeof(int));

  const cuda::counting_iterator<int> iota(0);
  auto input_it = cuda::transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = cuda::transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});

  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_constant_iterator sizes(bytes_per_buffer, thrust::raw_pointer_cast(d_block_size.data()));

  const auto env = cuda::execution::tune(batch_memcpy_tuning<target_block_size>{});

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess
          == cub::DeviceMemcpy::Batched(nullptr, temp_storage_bytes, input_it, output_it, sizes, num_buffers, env));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(
    cudaSuccess
    == cub::DeviceMemcpy::Batched(
      thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, input_it, output_it, sizes, num_buffers, env));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_dst == d_src);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH == 0

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test BatchedCopyPolicy properties", "[memcpy][device]", CUB_SMALL)
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
  constexpr auto p1 =
    cub::BatchedCopyPolicy{cub::BatchedCopyAlgorithm::lookback, cub::BatchedCopyLookbackPolicy{p1_small, p1_large}};

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
  constexpr auto p2       = cub::BatchedCopyPolicy{
    .algorithm = cub::BatchedCopyAlgorithm::lookback,
    .lookback  = cub::BatchedCopyLookbackPolicy{.small_buffer = p2_small, .large_buffer = p2_large}};
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
    == "BatchedCopyPolicy { .algorithm = BatchedCopyAlgorithm::lookback"
       ", .lookback = BatchedCopyLookbackPolicy { .small_buffer = BatchedCopySmallBufferPolicy { .threads_per_block = "
       "128"
       ", .buffers_per_thread = 4, .bytes_per_thread = 8, .prefer_pow2_bits = 0"
       ", .block_level_tile_size = 8192, .warp_level_threshold = 128, .block_level_threshold = 8192"
       ", .buffer_lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 350, .l2_write_latency = 450 }"
       ", .block_lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 350, .l2_write_latency = 450 } }"
       ", .large_buffer = BatchedCopyLargeBufferPolicy { .threads_per_block = 256"
       ", .bytes_per_thread = 32 } } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
