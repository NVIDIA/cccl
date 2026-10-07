// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/custom_type.h>
#include <c2h/detail/env.cuh>
#include <c2h/device_and_stream.h>

namespace
{
struct nontrivial_default_constructible_t
{
  constexpr nontrivial_default_constructible_t() noexcept
      : value(-1)
  {}

  constexpr explicit nontrivial_default_constructible_t(std::int32_t value_) noexcept
      : value(value_)
  {}

  friend constexpr bool operator==(const nontrivial_default_constructible_t& lhs,
                                   const nontrivial_default_constructible_t& rhs) noexcept
  {
    return lhs.value == rhs.value;
  }

  const std::int32_t value;
};

static_assert(cuda::std::is_trivially_copyable_v<nontrivial_default_constructible_t>);
static_assert(!cuda::std::is_trivially_default_constructible_v<nontrivial_default_constructible_t>);
static_assert(!cuda::std::is_copy_assignable_v<nontrivial_default_constructible_t>);
static_assert(nontrivial_default_constructible_t{}.value == -1);

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

  const auto device = c2h::current_device();
  const cuda::stream stream{device};

  REQUIRE_THROWS_AS(c2h::make_device_buffer<char>(stream, get_alloc_bytes(), cuda::no_init), std::bad_alloc);

  constexpr std::size_t num_items = 256;
  const auto d_items              = c2h::make_device_buffer<std::int32_t>(stream, num_items, cuda::no_init);
  REQUIRE(d_items.size() == num_items);
  REQUIRE(d_items.data() != nullptr);

  const auto empty = c2h::make_device_buffer<std::int32_t>(stream, std::size_t{0}, cuda::no_init);
  REQUIRE(empty.empty());
  REQUIRE(empty.data() == nullptr);

  constexpr std::array<std::int32_t, 4> expected{1, 2, 3, 4};
  const auto d_initialized = c2h::make_device_buffer<std::int32_t>(stream, {1, 2, 3, 4});
  const auto h_initialized = c2h::make_host_buffer<std::int32_t>(stream, d_initialized);
  stream.sync();
  REQUIRE(std::equal(h_initialized.begin(), h_initialized.end(), expected.begin(), expected.end()));

  auto resource                    = c2h::checked_device_memory_resource{device};
  constexpr auto invalid_alignment = cuda::mr::default_cuda_malloc_alignment - 1;
  REQUIRE_THROWS_AS(resource.allocate_sync(1, invalid_alignment), std::bad_alloc);
}

CUB_TEST("c2h checked memory resources support the legacy default stream", "[c2h][buffers]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = cuda::stream_ref{cudaStream_t{}};

  constexpr std::size_t num_items = 1;
  const auto d_items              = c2h::make_device_buffer<std::int32_t>(stream, device, num_items, cuda::no_init);
  auto h_items                    = c2h::make_host_buffer<std::int32_t>(stream, device, num_items, cuda::no_init);
  const auto d_initialized        = c2h::make_device_buffer<std::int32_t>(stream, device, {42});
  const auto h_initialized        = c2h::make_host_buffer<std::int32_t>(stream, device, {42});

  REQUIRE(d_items.size() == num_items);
  REQUIRE(d_items.data() != nullptr);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(h_items.data() != nullptr);
  REQUIRE(d_initialized.size() == num_items);
  REQUIRE(h_initialized.size() == num_items);
  REQUIRE(h_initialized.front() == 42);

  std::int32_t initialized_value{};
  REQUIRE(cudaSuccess
          == cudaMemcpy(&initialized_value, d_initialized.data(), sizeof(initialized_value), cudaMemcpyDeviceToHost));
  REQUIRE(initialized_value == 42);

  h_items.front() = 42;
  REQUIRE(h_items.front() == 42);
}

CUB_TEST("c2h checked host memory resource creates writable host buffers", "[c2h][buffers][host_resource]", CUB_SMALL)
{
  STATIC_REQUIRE(
    cuda::mr::synchronous_resource_with<c2h::checked_host_buffer_memory_resource, cuda::mr::host_accessible>);

  const auto device = c2h::current_device();
  const cuda::stream stream{device};

  constexpr std::size_t num_items = 256;
  auto h_items                    = c2h::make_host_buffer<std::int32_t>(stream, num_items, cuda::no_init);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(h_items.data() != nullptr);

  h_items.front() = 42;
  REQUIRE(h_items.front() == 42);

  const auto empty = c2h::make_host_buffer<std::int32_t>(stream, std::size_t{0}, cuda::no_init);
  REQUIRE(empty.empty());
  REQUIRE(empty.data() == nullptr);

  constexpr std::array<std::int32_t, 4> expected{1, 2, 3, 4};
  const auto initialized = c2h::make_host_buffer<std::int32_t>(stream, {1, 2, 3, 4});
  REQUIRE(std::equal(initialized.begin(), initialized.end(), expected.begin(), expected.end()));

  auto resource = c2h::checked_host_buffer_memory_resource{device};

  constexpr std::size_t aligned_bytes = 1;
  constexpr std::size_t alignment     = cuda::mr::default_cuda_malloc_alignment * 2;
  void* const aligned_ptr             = resource.allocate_sync(aligned_bytes, alignment);
  REQUIRE(aligned_ptr != nullptr);
  REQUIRE(reinterpret_cast<std::uintptr_t>(aligned_ptr) % alignment == 0);
  resource.deallocate_sync(aligned_ptr, aligned_bytes, alignment);

  REQUIRE_THROWS_AS(resource.allocate_sync(1, 0), std::bad_alloc);
}

CUB_TEST("c2h host buffer initializer list constructs elements", "[c2h][buffers][host_resource]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const cuda::stream stream{device};

  using value_type = nontrivial_default_constructible_t;
  constexpr std::array<value_type, 4> expected{value_type{1}, value_type{2}, value_type{3}, value_type{4}};
  const auto initialized =
    c2h::make_host_buffer<value_type>(stream, {value_type{1}, value_type{2}, value_type{3}, value_type{4}});

  REQUIRE(std::equal(initialized.begin(), initialized.end(), expected.begin(), expected.end()));
}

CUB_TEST("c2h buffer generator handles zero items", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const cuda::stream stream{device};
  auto d_items = c2h::make_device_buffer<std::int32_t>(stream, std::size_t{0}, cuda::no_init);
  c2h::gen(c2h::seed_t{1234}, d_items);

  REQUIRE(d_items.empty());
  REQUIRE(d_items.data() == nullptr);
}

CUB_TEST("c2h buffer generator honors the requested range", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  constexpr std::size_t num_items  = 256;
  constexpr std::int32_t min_value = -100;
  constexpr std::int32_t max_value = 100;
  const auto buffers = c2h::gen_buffers<std::int32_t>(stream, c2h::seed_t{1234}, num_items, min_value, max_value);

  const bool values_are_in_range =
    std::all_of(buffers.h_items.begin(), buffers.h_items.end(), [](const std::int32_t value) {
      return min_value <= value && value <= max_value;
    });
  REQUIRE(values_are_in_range);

  const auto first_value = buffers.h_items.front();
  const bool has_distinct_values =
    std::any_of(buffers.h_items.begin(), buffers.h_items.end(), [first_value](const std::int32_t value) {
      return value != first_value;
    });
  REQUIRE(has_distinct_values);
}

CUB_TEST("c2h buffer generators populate checked CUDA buffers", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  constexpr std::size_t num_items = 256;
  constexpr std::int32_t expected = 42;
  const auto buffers = c2h::gen_buffers<std::int32_t>(stream, c2h::seed_t{1234}, num_items, expected, expected);

  REQUIRE(buffers.size() == num_items);
  REQUIRE(buffers.d_items.size() == num_items);
  REQUIRE(buffers.h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(buffers.h_items.begin(), buffers.h_items.end(), expected)) == num_items);

  constexpr std::int32_t host_expected = -17;
  const auto h_items =
    c2h::gen_host_buffer<std::int32_t>(stream, c2h::seed_t{5678}, num_items, host_expected, host_expected);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(h_items.begin(), h_items.end(), host_expected)) == num_items);

  using custom_type = c2h::custom_type_t<c2h::equal_comparable_t>;
  custom_type custom_expected{};
  custom_expected.key = 13;
  custom_expected.val = 37;

  auto d_custom = c2h::make_device_buffer<custom_type>(stream, num_items, cuda::no_init);
  c2h::gen(c2h::seed_t{9012}, d_custom, custom_expected, custom_expected);
  const auto h_custom = c2h::make_host_buffer<custom_type>(stream, d_custom);
  stream.sync();

  const bool custom_values_match = std::all_of(h_custom.begin(), h_custom.end(), [&](const custom_type& value) {
    return value == custom_expected;
  });
  REQUIRE(custom_values_match);
}
