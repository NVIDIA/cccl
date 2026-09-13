// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/span>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/detail/env.cuh>
#include <c2h/detail/generators.cuh>

namespace
{
[[nodiscard]] std::size_t get_alloc_bytes()
{
  std::size_t free_bytes{};
  std::size_t total_bytes{};
  REQUIRE(cudaSuccess == cudaMemGetInfo(&free_bytes, &total_bytes));

  // Find a size that's > free but < total, preferring to return more than total if the values are
  // too close.
  constexpr std::size_t one_MiB = 1024 * 1024;
  const std::size_t alloc_bytes = ::std::max(total_bytes - one_MiB, free_bytes + one_MiB);
  CAPTURE(free_bytes, total_bytes, alloc_bytes);
  return alloc_bytes;
}
} // namespace

CUB_TEST("c2h integral environment parser rejects invalid values", "[c2h][buffers][env]", CUB_SMALL)
{
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>(nullptr) == 0);
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>("") == 0);
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>("0") == 0);
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>("1024") == 1024);
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>("-1") == 0);
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>(" -1") == 0);
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>("1x") == 0);
  REQUIRE(c2h::detail::parse_env_integer<std::size_t>("18446744073709551616") == 0);

  REQUIRE(c2h::detail::parse_env_integer<long long>("-1") == -1);
  REQUIRE(c2h::detail::parse_env_integer<long long>("9223372036854775808") == 0);
}

CUB_TEST("c2h checked host allocation rejects invalid alignments", "[c2h][buffers][host_resource]", CUB_SMALL)
{
  REQUIRE_THROWS_AS(c2h::detail::checked_host_allocation_size(1, 3), std::bad_alloc);
}

CUB_TEST("c2h checked memory rejects sizes that overflow padding", "[c2h][buffers][device_resource]", CUB_SMALL)
{
  REQUIRE(
    c2h::detail::check_free_device_memory((std::numeric_limits<std::size_t>::max)()) == cudaErrorMemoryAllocation);
}

CUB_TEST("c2h checked device memory resource creates device buffers", "[c2h][buffers][device_resource]", CUB_SMALL)
{
  STATIC_REQUIRE(cuda::mr::synchronous_resource_with<c2h::checked_device_memory_resource, cuda::mr::device_accessible>);

  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));

  const auto device = cuda::device_ref{device_id};
  const cuda::stream stream{device};

  REQUIRE_THROWS_AS(c2h::make_device_buffer<char>(stream, device, get_alloc_bytes(), cuda::no_init), std::bad_alloc);

  constexpr std::size_t num_items = 256;
  const auto d_items              = c2h::make_device_buffer<std::int32_t>(stream, device, num_items, cuda::no_init);
  REQUIRE(d_items.size() == num_items);
  REQUIRE(d_items.data() != nullptr);

  const auto empty = c2h::make_device_buffer<std::int32_t>(stream, device, std::size_t{0}, cuda::no_init);
  REQUIRE(empty.empty());
  REQUIRE(empty.data() == nullptr);

  constexpr std::array<std::int32_t, 4> expected{1, 2, 3, 4};
  const auto d_initialized = c2h::make_device_buffer<std::int32_t>(stream, device, {1, 2, 3, 4});
  const auto h_initialized = c2h::make_host_buffer<std::int32_t>(stream, device, d_initialized);
  stream.sync();
  REQUIRE(std::equal(h_initialized.begin(), h_initialized.end(), expected.begin(), expected.end()));

  auto resource                    = c2h::checked_device_memory_resource{device};
  constexpr auto invalid_alignment = cuda::mr::default_cuda_malloc_alignment - 1;
  REQUIRE_THROWS_AS(resource.allocate_sync(1, invalid_alignment), std::bad_alloc);
}

CUB_TEST("c2h checked host memory resource creates writable host buffers", "[c2h][buffers][host_resource]", CUB_SMALL)
{
  STATIC_REQUIRE(
    cuda::mr::synchronous_resource_with<c2h::checked_host_buffer_memory_resource, cuda::mr::host_accessible>);

  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));

  const auto device = cuda::device_ref{device_id};
  const cuda::stream stream{device};

  constexpr std::size_t num_items = 256;
  auto h_items                    = c2h::make_host_buffer<std::int32_t>(stream, device, num_items, cuda::no_init);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(h_items.data() != nullptr);

  h_items.front() = 42;
  REQUIRE(h_items.front() == 42);

  const auto empty = c2h::make_host_buffer<std::int32_t>(stream, device, std::size_t{0}, cuda::no_init);
  REQUIRE(empty.empty());
  REQUIRE(empty.data() == nullptr);

  auto resource = c2h::checked_host_buffer_memory_resource{device};
  REQUIRE_THROWS_AS(resource.allocate_sync(1, 0), std::bad_alloc);
}

CUB_TEST("c2h random generator handles zero items", "[c2h][buffers][generators]", CUB_SMALL)
{
  const c2h::seed_t seed{1234};

  {
    const auto random_data = c2h::detail::prepare_random_data(seed, 0);
    REQUIRE(random_data.data() == nullptr);
  }

  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));

  const cuda::stream stream{cuda::device_ref{device_id}};
  const auto random_data = c2h::detail::prepare_random_data(stream, seed, 0);
  REQUIRE(random_data.data() == nullptr);
}

CUB_TEST("c2h buffer generators populate checked CUDA buffers", "[c2h][buffers][generators]", CUB_SMALL)
{
  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));

  const auto device = cuda::device_ref{device_id};
  const cuda::stream stream{device};

  constexpr std::size_t num_items = 256;
  constexpr std::int32_t expected = 42;
  const auto buffers = c2h::gen_buffers<std::int32_t>(stream, device, c2h::seed_t{1234}, num_items, expected, expected);

  REQUIRE(buffers.size == num_items);
  REQUIRE(buffers.d_items.size() == num_items);
  REQUIRE(buffers.h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(buffers.h_items.begin(), buffers.h_items.end(), expected)) == num_items);

  constexpr std::int32_t host_expected = -17;
  const auto h_items =
    c2h::gen_host_buffer<std::int32_t>(stream, device, c2h::seed_t{5678}, num_items, host_expected, host_expected);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(h_items.begin(), h_items.end(), host_expected)) == num_items);
}

CUB_TEST("c2h random generator supports the legacy default stream", "[c2h][buffers][generators]", CUB_SMALL)
{
  constexpr std::size_t num_items = 256;
  const c2h::seed_t seed{1234};

  {
    const auto implicit_stream_data = c2h::detail::prepare_random_data(seed, num_items);
    REQUIRE(implicit_stream_data.data() != nullptr);
  }

  {
    const auto explicit_stream_data =
      c2h::detail::prepare_random_data(cuda::stream_ref{::cudaStream_t{}}, seed, num_items);
    REQUIRE(explicit_stream_data.data() != nullptr);
  }

  REQUIRE(cudaSuccess == cudaStreamSynchronize(::cudaStream_t{}));
}

CUB_TEST("c2h random generator isolates in-flight streams", "[c2h][buffers][generators][streams]", CUB_SMALL)
{
  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));

  const auto device = cuda::device_ref{device_id};
  const cuda::stream first_stream{device};
  const cuda::stream second_stream{device};

  constexpr std::size_t num_items = 256;
  const c2h::seed_t first_seed{1234};
  const c2h::seed_t second_seed{5678};

  auto d_expected = c2h::make_device_buffer<float>(first_stream, device, num_items, cuda::no_init);
  auto d_actual   = c2h::make_device_buffer<float>(first_stream, device, num_items, cuda::no_init);

  const auto first_data = c2h::detail::prepare_random_data(first_stream, first_seed, num_items);
  cuda::copy_bytes(first_stream, cuda::std::span<const float>{first_data.data(), num_items}, d_expected);

  // Capture the first distribution before allowing the second stream to generate its distribution.
  const auto first_distribution_captured = first_stream.record_event();
  second_stream.wait(first_distribution_captured);

  const auto second_data                = c2h::detail::prepare_random_data(second_stream, second_seed, num_items);
  const auto second_generation_complete = second_stream.record_event();
  first_stream.wait(second_generation_complete);

  // Delay consumption of the first distribution until the second stream has generated its distribution.
  cuda::copy_bytes(first_stream, cuda::std::span<const float>{first_data.data(), num_items}, d_actual);

  const auto h_expected = c2h::make_host_buffer<float>(first_stream, device, d_expected);
  const auto h_actual   = c2h::make_host_buffer<float>(first_stream, device, d_actual);
  first_stream.sync();

  REQUIRE(std::equal(h_actual.begin(), h_actual.end(), h_expected.begin(), h_expected.end()));
}

CUB_TEST("c2h random generator isolates per-thread default streams", "[c2h][buffers][generators][streams]", CUB_SMALL)
{
  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));

  const auto device = cuda::device_ref{device_id};
  const cuda::stream copy_stream{device};

  constexpr std::size_t num_items = 256;
  const c2h::seed_t first_seed{1234};
  const c2h::seed_t second_seed{5678};

  auto d_expected = c2h::make_device_buffer<float>(copy_stream, device, num_items, cuda::no_init);
  auto d_actual   = c2h::make_device_buffer<float>(copy_stream, device, num_items, cuda::no_init);

  std::promise<cuda::event> first_distribution_captured_promise;
  auto first_distribution_captured = first_distribution_captured_promise.get_future();
  std::promise<cuda::event> second_generation_complete_promise;
  auto second_generation_complete = second_generation_complete_promise.get_future();

  auto first_task = std::async(std::launch::async, [&] {
    auto first_data = [&] {
      try
      {
        const cudaError_t status = cudaSetDevice(device_id);
        if (status != cudaSuccess)
        {
          throw cuda::cuda_error{status, "failed to set current device"};
        }

        const cuda::stream_ref stream{cudaStreamPerThread};
        auto data = c2h::detail::prepare_random_data(stream, first_seed, num_items);
        cuda::copy_bytes(stream, cuda::std::span<const float>{data.data(), num_items}, d_expected);
        auto first_distribution_captured_event = stream.record_event();
        first_distribution_captured_promise.set_value(std::move(first_distribution_captured_event));
        return data;
      }
      catch (...)
      {
        first_distribution_captured_promise.set_exception(std::current_exception());
        throw;
      }
    }();

    const cuda::stream_ref stream{cudaStreamPerThread};
    const auto second_generation_complete_event = second_generation_complete.get();
    stream.wait(second_generation_complete_event);
    cuda::copy_bytes(stream, cuda::std::span<const float>{first_data.data(), num_items}, d_actual);
    stream.sync();
  });

  auto second_task = std::async(std::launch::async, [&] {
    try
    {
      const cudaError_t status = cudaSetDevice(device_id);
      if (status != cudaSuccess)
      {
        throw cuda::cuda_error{status, "failed to set current device"};
      }

      const cuda::stream_ref stream{cudaStreamPerThread};
      const auto first_distribution_captured_event = first_distribution_captured.get();
      stream.wait(first_distribution_captured_event);
      const auto second_data                = c2h::detail::prepare_random_data(stream, second_seed, num_items);
      auto second_generation_complete_event = stream.record_event();
      stream.sync();
      second_generation_complete_promise.set_value(std::move(second_generation_complete_event));
    }
    catch (...)
    {
      second_generation_complete_promise.set_exception(std::current_exception());
      throw;
    }
  });

  second_task.get();
  first_task.get();

  const auto h_expected = c2h::make_host_buffer<float>(copy_stream, device, d_expected);
  const auto h_actual   = c2h::make_host_buffer<float>(copy_stream, device, d_actual);
  copy_stream.sync();

  REQUIRE(std::equal(h_actual.begin(), h_actual.end(), h_expected.begin(), h_expected.end()));
}

CUB_TEST("c2h random generator bounds cached stream states", "[c2h][buffers][generators][streams]", CUB_SMALL)
{
  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));

  const auto device = cuda::device_ref{device_id};
  std::vector<cuda::stream> streams;
  streams.reserve(c2h::detail::max_cached_generator_states + 1);

  streams.emplace_back(device);
  constexpr std::size_t num_items = 256;
  auto d_expected                 = c2h::make_device_buffer<float>(streams.front(), device, num_items, cuda::no_init);
  auto d_actual                   = c2h::make_device_buffer<float>(streams.front(), device, num_items, cuda::no_init);
  const auto first_data           = c2h::detail::prepare_random_data(streams.front(), c2h::seed_t{1234}, num_items);
  cuda::copy_bytes(streams.front(), cuda::std::span<const float>{first_data.data(), num_items}, d_expected);

  for (std::size_t index = 1; index <= c2h::detail::max_cached_generator_states; ++index)
  {
    streams.emplace_back(device);
    const auto random_data = c2h::detail::prepare_random_data(streams.back(), c2h::seed_t{static_cast<int>(index)}, 1);
    REQUIRE(random_data.data() != nullptr);
  }

  // The first state has been evicted from the cache, but its lease keeps the distribution alive for another consumer.
  cuda::copy_bytes(streams.front(), cuda::std::span<const float>{first_data.data(), num_items}, d_actual);
  const auto h_expected = c2h::make_host_buffer<float>(streams.front(), device, d_expected);
  const auto h_actual   = c2h::make_host_buffer<float>(streams.front(), device, d_actual);

  for (const auto& stream : streams)
  {
    stream.sync();
  }

  REQUIRE(c2h::detail::cached_generator_state_count() == c2h::detail::max_cached_generator_states);
  REQUIRE(std::equal(h_actual.begin(), h_actual.end(), h_expected.begin(), h_expected.end()));
}
