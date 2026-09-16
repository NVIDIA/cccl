// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/catch2_test_helper.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/custom_type.h>
#include <c2h/detail/env.cuh>
#include <c2h/detail/scoped_current_device.cuh>
#include <c2h/generator_common.h>
#include <c2h/vector.h>
#include <c2h/vector_generators.h>

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

CUB_TEST("c2h vector matcher rejects ranges with different sizes", "[c2h][comparison]", CUB_SMALL)
{
  const c2h::host_vector<std::int32_t> actual;
  const c2h::host_vector<std::int32_t> expected{42};

  REQUIRE_FALSE(Equals(expected).match(actual));
}

CUB_TEST("c2h checked memory rejects sizes that overflow padding", "[c2h][buffers][device_resource]", CUB_SMALL)
{
  REQUIRE(
    c2h::detail::check_free_device_memory((std::numeric_limits<std::size_t>::max)()) == cudaErrorMemoryAllocation);
}

CUB_TEST("c2h uniform offset size validation rejects invalid element counts", "[c2h][buffers][generators]", CUB_SMALL)
{
  REQUIRE(c2h::detail::checked_uniform_offsets_size(cuda::std::int32_t{0}) == 2);
  REQUIRE(c2h::detail::checked_uniform_offsets_size(cuda::std::int32_t{1}) == 3);

  REQUIRE_THROWS_AS(c2h::detail::checked_uniform_offsets_size(cuda::std::int32_t{-1}), std::invalid_argument);
  REQUIRE_THROWS_AS(c2h::detail::checked_uniform_offsets_size((cuda::std::numeric_limits<cuda::std::int32_t>::max)()),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
    c2h::detail::checked_uniform_offsets_size((cuda::std::numeric_limits<cuda::std::uint64_t>::max)() - 1),
    std::invalid_argument);
}

CUB_TEST("c2h uniform offset generators validate sizes before allocation", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto seed = c2h::seed_t{0};

  REQUIRE_THROWS_AS(
    c2h::gen_uniform_offsets(seed, cuda::std::int32_t{-1}, cuda::std::int32_t{0}, cuda::std::int32_t{1}),
    std::invalid_argument);

  const auto device = c2h::current_test_device();
  const cuda::stream stream{device};
  REQUIRE_THROWS_AS(c2h::gen_uniform_offsets_device_buffer(
                      stream, seed, cuda::std::int32_t{-1}, cuda::std::int32_t{0}, cuda::std::int32_t{1}),
                    std::invalid_argument);
}

CUB_TEST("c2h detail uniform offset generator validates destination size", "[c2h][buffers][generators]", CUB_SMALL)
{
  REQUIRE_THROWS_AS(
    c2h::detail::gen_uniform_offsets(
      c2h::seed_t{0},
      cuda::std::span<cuda::std::int32_t>{},
      cuda::std::int32_t{1},
      cuda::std::int32_t{0},
      cuda::std::int32_t{1}),
    std::invalid_argument);
}

CUB_TEST("c2h checked device memory resource creates device buffers", "[c2h][buffers][device_resource]", CUB_SMALL)
{
  STATIC_REQUIRE(cuda::mr::synchronous_resource_with<c2h::checked_device_memory_resource, cuda::mr::device_accessible>);

  const auto device = c2h::current_test_device();
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

  const auto device = c2h::current_test_device();
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

  constexpr std::size_t aligned_bytes = 1;
  constexpr std::size_t alignment     = cuda::mr::default_cuda_malloc_alignment * 2;
  void* const aligned_ptr             = resource.allocate_sync(aligned_bytes, alignment);
  REQUIRE(aligned_ptr != nullptr);
  REQUIRE(reinterpret_cast<std::uintptr_t>(aligned_ptr) % alignment == 0);
  resource.deallocate_sync(aligned_ptr, aligned_bytes, alignment);

  REQUIRE_THROWS_AS(resource.allocate_sync(1, 0), std::bad_alloc);
}

CUB_TEST("c2h buffer generator handles zero items", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto device = c2h::current_test_device();
  const cuda::stream stream{device};
  const auto d_items = c2h::gen_device_buffer<std::int32_t>(stream, c2h::seed_t{1234}, 0);
  REQUIRE(d_items.empty());
  REQUIRE(d_items.data() == nullptr);
}

CUB_TEST("c2h buffer generators populate checked CUDA buffers", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto device = c2h::current_test_device();
  const cuda::stream stream{device};

  constexpr std::size_t num_items = 256;
  constexpr std::int32_t expected = 42;
  const auto buffers = c2h::gen_buffers<std::int32_t>(stream, c2h::seed_t{1234}, num_items, expected, expected);

  REQUIRE(buffers.size == num_items);
  REQUIRE(buffers.d_items.size() == num_items);
  REQUIRE(buffers.h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(buffers.h_items.begin(), buffers.h_items.end(), expected)) == num_items);

  constexpr std::int32_t host_expected = -17;
  const auto h_items =
    c2h::gen_host_buffer<std::int32_t>(stream, c2h::seed_t{5678}, num_items, host_expected, host_expected);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(h_items.begin(), h_items.end(), host_expected)) == num_items);
}

CUB_TEST("c2h stream generators select the stream device", "[c2h][buffers][generators]", CUB_SMALL)
{
  int device_count{};
  REQUIRE(cudaSuccess == cudaGetDeviceCount(&device_count));
  if (device_count < 2)
  {
    SKIP("This test requires at least two CUDA devices.");
  }

  const auto device      = c2h::current_test_device();
  const int other_device = (device.get() + 1) % device_count;
  const cuda::stream stream{device};
  constexpr std::size_t size = 1;

  auto d_scalar = c2h::make_device_buffer<std::int32_t>(stream, device, size, cuda::no_init);
  auto d_vector = c2h::make_device_buffer<int2>(stream, device, size, cuda::no_init);

  using custom_t = c2h::custom_type_t<c2h::equal_comparable_t>;
  auto d_custom  = c2h::make_device_buffer<custom_t>(stream, device, size, cuda::no_init);

  constexpr std::int32_t total_elements   = 8;
  constexpr std::int32_t min_segment_size = 1;
  constexpr std::int32_t max_segment_size = 2;
  const auto offsets_size                 = c2h::detail::checked_uniform_offsets_size(total_elements);
  auto d_offsets = c2h::make_device_buffer<std::int32_t>(stream, device, offsets_size, cuda::no_init);

  constexpr std::int32_t scalar_value = 42;
  const int2 vector_value{42, 42};
  const auto custom_value = [] {
    custom_t value{};
    value.key = 42;
    value.val = 42;
    return value;
  }();

  {
    const c2h::detail::scoped_current_device other_device_scope{other_device};

    c2h::detail::gen_into_device_buffer(c2h::seed_t{1234}, d_scalar, scalar_value, scalar_value);
    c2h::detail::gen_into_device_buffer(c2h::seed_t{1234}, d_vector, vector_value, vector_value);
    c2h::detail::gen_into_device_buffer(c2h::seed_t{1234}, d_custom, custom_value, custom_value);
    const auto num_offsets = c2h::detail::gen_uniform_offsets(
      stream, c2h::seed_t{1234}, d_offsets.first(d_offsets.size()), total_elements, min_segment_size, max_segment_size);
    REQUIRE(num_offsets >= 2);
    REQUIRE(num_offsets <= offsets_size);

    int current_device{};
    REQUIRE(cudaSuccess == cudaGetDevice(&current_device));
    REQUIRE(current_device == other_device);
  }

  const auto h_scalar  = c2h::make_host_buffer<std::int32_t>(stream, device, d_scalar);
  const auto h_vector  = c2h::make_host_buffer<int2>(stream, device, d_vector);
  const auto h_custom  = c2h::make_host_buffer<custom_t>(stream, device, d_custom);
  const auto h_offsets = c2h::make_host_buffer<std::int32_t>(stream, device, d_offsets);
  stream.sync();

  REQUIRE(h_scalar.front() == scalar_value);
  REQUIRE(h_vector.front().x == vector_value.x);
  REQUIRE(h_vector.front().y == vector_value.y);
  REQUIRE(h_custom.front() == custom_value);
  REQUIRE(h_offsets.front() == 0);
  REQUIRE(std::find(h_offsets.begin(), h_offsets.end(), total_elements) != h_offsets.end());
}
