// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/__algorithm/fill.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/buffer_utils.cuh>
#include <c2h/catch2_test_helper.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/custom_type.h>
#include <c2h/detail/env.cuh>
#include <c2h/device_and_stream.h>

namespace
{
[[nodiscard]] constexpr cuda::std::int32_t repeat_byte(cuda::std::uint8_t value) noexcept
{
  return cuda::std::int32_t{value} * 0x01010101;
}

struct nontrivial_default_constructible_t
{
  constexpr nontrivial_default_constructible_t() noexcept
      : value(-1)
  {}

  constexpr explicit nontrivial_default_constructible_t(cuda::std::int32_t value_) noexcept
      : value(value_)
  {}

  friend constexpr bool operator==(const nontrivial_default_constructible_t& lhs,
                                   const nontrivial_default_constructible_t& rhs) noexcept
  {
    return lhs.value == rhs.value;
  }

  const cuda::std::int32_t value;
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
  REQUIRE_THROWS_AS(c2h::detail::checked_host_allocate(0, 0, 3), std::bad_alloc);
}

CUB_TEST("c2h vector matcher rejects ranges with different sizes", "[c2h][comparison]", CUB_SMALL)
{
  const c2h::host_vector<cuda::std::int32_t> actual;
  const c2h::host_vector<cuda::std::int32_t> expected{42};

  REQUIRE_FALSE(Equals(expected).match(actual));
}

CUB_TEST("c2h buffer matcher accepts host containers", "[c2h][buffers][comparison]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();
  const auto actual = c2h::make_device_buffer<int>(stream, {1, 2, 3});

  const std::array<int, 3> expected_array{1, 2, 3};
  const std::vector<int> expected_vector{1, 2, 3};
  const cuda::std::array<int, 3> expected_cuda_array{1, 2, 3};
  const auto expected_buffer = c2h::make_host_buffer<int>(stream, {1, 2, 3});

  REQUIRE_THAT(actual, c2h::Equals(stream, expected_array));
  REQUIRE_THAT(actual, c2h::Equals(stream, expected_vector));
  REQUIRE_THAT(actual, c2h::Equals(stream, expected_cuda_array));
  REQUIRE_THAT(actual, c2h::Equals(stream, expected_buffer));

  const std::array<int, 3> different_values{1, 4, 3};
  const std::vector<int> different_size{1, 2};
  REQUIRE_FALSE(c2h::Equals(stream, different_values).match(actual));
  REQUIRE_FALSE(c2h::Equals(stream, different_size).match(actual));

  const auto empty = c2h::make_device_buffer<int>(stream, 0, cuda::no_init);
  const std::vector<int> empty_expected;
  REQUIRE_THAT(empty, c2h::Equals(stream, empty_expected));
  REQUIRE_FALSE(c2h::Equals(stream, expected_array).match(empty));
  REQUIRE_FALSE(c2h::Equals(stream, empty_expected).match(actual));
}

CUB_TEST("c2h comparison retains bounded mismatch details in order", "[c2h][comparison]", CUB_SMALL)
{
  const auto mismatch_count = GENERATE(std::size_t{0}, 1, 5, 6, 10, 11, 15, 16);
  const auto first_mismatch = GENERATE(std::size_t{0}, 2, 4);
  CAPTURE(mismatch_count, first_mismatch);

  // Leave matching values between mismatches so element indices differ from mismatch ordinals.
  const auto num_items = first_mismatch + 2 * mismatch_count + 1;
  std::vector<int> expected(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    expected[i] = static_cast<int>(i + 1);
  }
  auto actual = expected;
  for (std::size_t i = 0; i < mismatch_count; ++i)
  {
    const auto index = first_mismatch + 2 * i;
    actual[index]    = -expected[index];
  }

  const auto result = c2h::detail::compare_host_ranges(actual, expected);
  REQUIRE(result.actual_size == num_items);
  REQUIRE(result.expected_size == num_items);
  REQUIRE(result.total_mismatches == mismatch_count);

  const auto first_count = (std::min) (mismatch_count, std::size_t{5});
  REQUIRE(result.first_mismatches.size() == first_count);
  for (std::size_t i = 0; i < first_count; ++i)
  {
    const auto index = first_mismatch + 2 * i;
    CAPTURE(i, index);
    CHECK(result.first_mismatches[i].index == index);
    CHECK(result.first_mismatches[i].actual == actual[index]);
    CHECK(result.first_mismatches[i].expected == expected[index]);
  }

  REQUIRE(result.last_mismatches.has_value() == (mismatch_count > 5));
  if (result.last_mismatches)
  {
    const auto last_count = (std::min) (mismatch_count - first_count, std::size_t{5});
    REQUIRE(result.last_mismatches->size() == last_count);
    for (std::size_t i = 0; i < last_count; ++i)
    {
      const auto index = first_mismatch + 2 * (mismatch_count - last_count + i);
      CAPTURE(i, index);
      CHECK((*result.last_mismatches)[i].index == index);
      CHECK((*result.last_mismatches)[i].actual == actual[index]);
      CHECK((*result.last_mismatches)[i].expected == expected[index]);
    }
  }

  const auto good_count = mismatch_count == 0 ? std::size_t{0} : (std::min) (first_mismatch, std::size_t{3});
  REQUIRE(result.good_values.size() == good_count);
  for (std::size_t i = 0; i < good_count; ++i)
  {
    const auto index = first_mismatch - good_count + i;
    CAPTURE(i, index);
    CHECK(result.good_values[i].index == index);
    CHECK(result.good_values[i].value == expected[index]);
  }
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

  const auto device = c2h::current_device();
  const cuda::stream stream{device};
  REQUIRE_THROWS_AS(c2h::gen_uniform_offsets_device_buffer(
                      stream, seed, cuda::std::int32_t{-1}, cuda::std::int32_t{0}, cuda::std::int32_t{1}),
                    std::invalid_argument);
}

namespace
{
// Use a named callable instead of a local generic lambda to work around
// an internal compiler error in GCC 7.
template <typename T, std::size_t N>
struct check_device_offsets
{
  std::size_t expected_size;
  std::size_t expected_capacity;
  std::array<T, N> expected;

  void operator()(cuda::stream_ref stream,
                  const cuda::__resizable_buffer<T, cuda::mr::device_accessible>& offsets) const
  {
    REQUIRE(offsets.size() == expected_size);
    REQUIRE(offsets.capacity() == expected_capacity);

    const auto host = c2h::make_host_buffer<T>(stream, offsets.first(offsets.size()));
    stream.sync();

    REQUIRE(std::equal(host.begin(), host.end(), expected.begin(), expected.begin() + expected_size));
  }
};
} // namespace

CUB_TEST("c2h offset buffers retain capacity after shrinking their logical size",
         "[c2h][buffers][generators]",
         CUB_SMALL)
{
  using offset_t          = cuda::std::int32_t;
  constexpr std::size_t N = 5;

  const auto stream         = c2h::make_current_device_stream();
  const auto total_elements = GENERATE(offset_t{0}, 8);
  const std::array<offset_t, N> expected{0, 2, 4, 6, 8};
  const auto num_elements      = static_cast<std::size_t>(total_elements);
  const auto expected_size     = num_elements / 2 + 1;
  const auto expected_capacity = num_elements + 2;

  REQUIRE(1 <= expected_size);
  REQUIRE(expected_size <= N);
  REQUIRE(total_elements == expected[expected_size - 1]);

  const check_device_offsets<offset_t, N> verify_device_offsets{expected_size, expected_capacity, expected};

  SECTION("device buffer")
  {
    const auto offsets = c2h::gen_uniform_offsets_device_buffer(stream, c2h::seed_t{1234}, total_elements, 2, 2);
    verify_device_offsets(stream, offsets);
  }

  SECTION("device and host buffers")
  {
    const auto buffers = c2h::gen_uniform_offsets_buffers(stream, c2h::seed_t{1234}, total_elements, 2, 2);
    verify_device_offsets(stream, buffers.d_items);
    REQUIRE(buffers.size() == expected_size);
    REQUIRE(
      std::equal(buffers.h_items.begin(), buffers.h_items.end(), expected.begin(), expected.begin() + expected_size));
  }
}

CUB_TEST("c2h detail uniform offset generators validate counts and destination size",
         "[c2h][buffers][generators]",
         CUB_SMALL)
{
  using offset_t = cuda::std::int32_t;

  const auto stream         = c2h::make_current_device_stream();
  const auto total_elements = GENERATE(offset_t{-1}, 0, 1, (std::numeric_limits<offset_t>::max)());

  SECTION("default policy")
  {
    REQUIRE_THROWS_AS(c2h::detail::gen_uniform_offsets(
                        c2h::seed_t{0}, cuda::std::span<offset_t>{}, total_elements, offset_t{0}, offset_t{1}),
                      std::invalid_argument);
  }

  SECTION("explicit stream")
  {
    REQUIRE_THROWS_AS(c2h::detail::gen_uniform_offsets(
                        stream, c2h::seed_t{0}, cuda::std::span<offset_t>{}, total_elements, offset_t{0}, offset_t{1}),
                      std::invalid_argument);
  }
}

using uniform_offset_types =
  c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t, cuda::std::int64_t, cuda::std::uint64_t>;

CUB_TEST("c2h uniform offset generators agree across execution streams",
         "[c2h][buffers][generators]",
         CUB_SMALL,
         uniform_offset_types)
{
  using offset_t = c2h::get<0, TestType>;

  const auto total_elements = GENERATE(offset_t{0}, offset_t{1}, offset_t{31});
  const auto seed           = c2h::seed_t{1234};
  const auto stream         = c2h::make_current_device_stream();

  const auto expected_device                = c2h::gen_uniform_offsets(seed, total_elements, offset_t{0}, offset_t{5});
  const c2h::host_vector<offset_t> expected = expected_device;
  const auto actual = c2h::gen_uniform_offsets_buffers(stream, seed, total_elements, offset_t{0}, offset_t{5});
  stream.sync();

  REQUIRE(actual.size() == expected.size());
  REQUIRE(std::equal(actual.h_items.begin(), actual.h_items.end(), expected.begin(), expected.end()));
}

CUB_TEST("c2h checked device memory resource creates device buffers", "[c2h][buffers][device_resource]", CUB_SMALL)
{
  STATIC_REQUIRE(cuda::mr::synchronous_resource_with<c2h::checked_device_memory_resource, cuda::mr::device_accessible>);

  const auto device = c2h::current_device();
  const cuda::stream stream{device};

  REQUIRE_THROWS_AS(c2h::make_device_buffer<char>(stream, get_alloc_bytes(), cuda::no_init), std::bad_alloc);

  constexpr std::size_t num_items = 256;
  const auto d_items              = c2h::make_device_buffer<cuda::std::int32_t>(stream, num_items, cuda::no_init);
  REQUIRE(d_items.size() == num_items);
  REQUIRE(d_items.data() != nullptr);

  const auto empty = c2h::make_device_buffer<cuda::std::int32_t>(stream, std::size_t{0}, cuda::no_init);
  REQUIRE(empty.empty());
  REQUIRE(empty.data() == nullptr);

  constexpr std::array<cuda::std::int32_t, 4> expected{1, 2, 3, 4};
  const auto d_initialized = c2h::make_device_buffer<cuda::std::int32_t>(stream, {1, 2, 3, 4});
  const auto h_initialized = c2h::make_host_buffer<cuda::std::int32_t>(stream, d_initialized);
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
  const auto d_items       = c2h::make_device_buffer<cuda::std::int32_t>(stream, device, num_items, cuda::no_init);
  auto h_items             = c2h::make_host_buffer<cuda::std::int32_t>(stream, device, num_items, cuda::no_init);
  const auto d_initialized = c2h::make_device_buffer<cuda::std::int32_t>(stream, device, {42});
  const auto h_initialized = c2h::make_host_buffer<cuda::std::int32_t>(stream, device, {42});

  REQUIRE(d_items.size() == num_items);
  REQUIRE(d_items.data() != nullptr);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(h_items.data() != nullptr);
  REQUIRE(d_initialized.size() == num_items);
  REQUIRE(h_initialized.size() == num_items);
  REQUIRE(h_initialized.front() == 42);

  cuda::std::int32_t initialized_value{};
  REQUIRE(cudaSuccess
          == cudaMemcpy(&initialized_value, d_initialized.data(), sizeof(initialized_value), cudaMemcpyDeviceToHost));
  REQUIRE(initialized_value == 42);

  h_items.front() = 42;
  REQUIRE(h_items.front() == 42);
}

CUB_TEST("c2h default stream reference is constexpr and preserves the current device",
         "[c2h][buffers][streams]",
         CUB_SMALL)
{
  const auto device     = c2h::current_device();
  constexpr auto stream = c2h::default_stream_ref();
  STATIC_REQUIRE(stream.get() == cudaStream_t{});
  STATIC_REQUIRE(noexcept(c2h::default_stream_ref()));
  REQUIRE(c2h::current_device() == device);
}

CUB_TEST("c2h copy to host waits for completion", "[c2h][buffers][copy]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const cuda::stream owning_stream{device};
  const bool use_default_stream = GENERATE(false, true);
  const std::size_t num_items   = GENERATE(std::size_t{0}, std::size_t{1}, std::size_t{257});
  const auto stream             = use_default_stream ? c2h::default_stream_ref() : cuda::stream_ref{owning_stream};
  CAPTURE(use_default_stream, num_items);

  auto d_items = c2h::make_device_buffer<cuda::std::int32_t>(owning_stream, device, num_items, cuda::no_init);
  constexpr cuda::std::uint8_t canary = 0x2a;
  constexpr auto expected             = repeat_byte(canary);
  SECTION("checked host buffer")
  {
    auto h_items = c2h::make_host_buffer<cuda::std::int32_t>(owning_stream, device, num_items, cuda::no_init);
    if (num_items != 0)
    {
      // Keep the write after setup in both sections; checked host allocation can synchronize the device.
      cuda::fill_bytes(owning_stream, d_items, canary);
    }
    c2h::copy_to_host(stream, d_items, h_items);
    REQUIRE(owning_stream.is_done());
    REQUIRE(std::all_of(h_items.begin(), h_items.end(), [=](auto value) {
      return value == expected;
    }));
  }
  SECTION("standard vector")
  {
    std::vector<cuda::std::int32_t> h_items(num_items, -1);
    if (num_items != 0)
    {
      // Keep the write after setup in both sections; checked host allocation can synchronize the device.
      cuda::fill_bytes(owning_stream, d_items, canary);
    }
    c2h::copy_to_host(stream, d_items, h_items);
    REQUIRE(owning_stream.is_done());
    REQUIRE(std::all_of(h_items.begin(), h_items.end(), [=](auto value) {
      return value == expected;
    }));
  }
}

CUB_TEST("c2h read single waits for completion", "[c2h][buffers][copy]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const cuda::stream owning_stream{device};
  const bool use_default_stream = GENERATE(false, true);
  const auto stream             = use_default_stream ? c2h::default_stream_ref() : cuda::stream_ref{owning_stream};
  CAPTURE(use_default_stream);

  auto d_item = c2h::make_device_buffer<cuda::std::int32_t>(owning_stream, device, 1, cuda::no_init);
  constexpr cuda::std::uint8_t first_canary = 0x2a;
  constexpr auto first_expected             = repeat_byte(first_canary);
  cuda::fill_bytes(owning_stream, d_item, first_canary);
  REQUIRE(c2h::read_single(stream, d_item) == first_expected);
  REQUIRE(owning_stream.is_done());

  constexpr cuda::std::uint8_t second_canary = 0x17;
  constexpr auto second_expected             = repeat_byte(second_canary);
  cuda::fill_bytes(owning_stream, d_item, second_canary);
  REQUIRE(c2h::read_single(stream, d_item) == second_expected);
  REQUIRE(owning_stream.is_done());
}

CUB_TEST("c2h checked host memory resource creates writable host buffers", "[c2h][buffers][host_resource]", CUB_SMALL)
{
  STATIC_REQUIRE(
    cuda::mr::synchronous_resource_with<c2h::checked_host_buffer_memory_resource, cuda::mr::host_accessible>);

  const auto device = c2h::current_device();
  const cuda::stream stream{device};

  constexpr std::size_t num_items = 256;
  auto h_items                    = c2h::make_host_buffer<cuda::std::int32_t>(stream, num_items, cuda::no_init);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(h_items.data() != nullptr);

  h_items.front() = 42;
  REQUIRE(h_items.front() == 42);

  const auto empty = c2h::make_host_buffer<cuda::std::int32_t>(stream, std::size_t{0}, cuda::no_init);
  REQUIRE(empty.empty());
  REQUIRE(empty.data() == nullptr);

  constexpr std::array<cuda::std::int32_t, 4> expected{1, 2, 3, 4};
  const auto initialized = c2h::make_host_buffer<cuda::std::int32_t>(stream, {1, 2, 3, 4});
  REQUIRE(std::equal(initialized.begin(), initialized.end(), expected.begin(), expected.end()));

  auto resource = c2h::checked_host_buffer_memory_resource{device};

  constexpr std::size_t aligned_bytes = 1;
  constexpr std::size_t alignment     = cuda::mr::default_cuda_malloc_alignment * 2;
  void* const aligned_ptr             = resource.allocate_sync(aligned_bytes, alignment);
  REQUIRE(aligned_ptr != nullptr);
  REQUIRE(reinterpret_cast<cuda::std::uintptr_t>(aligned_ptr) % alignment == 0);
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
  auto d_items = c2h::make_device_buffer<cuda::std::int32_t>(stream, std::size_t{0}, cuda::no_init);
  c2h::gen(c2h::seed_t{1234}, d_items);

  REQUIRE(d_items.empty());
  REQUIRE(d_items.data() == nullptr);
}

CUB_TEST("c2h buffer generator honors the requested range", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  constexpr std::size_t num_items        = 256;
  constexpr cuda::std::int32_t min_value = -100;
  constexpr cuda::std::int32_t max_value = 100;
  const auto buffers = c2h::gen_buffers<cuda::std::int32_t>(stream, c2h::seed_t{1234}, num_items, min_value, max_value);

  const bool values_are_in_range =
    std::all_of(buffers.h_items.begin(), buffers.h_items.end(), [=](const cuda::std::int32_t value) {
      return min_value <= value && value <= max_value;
    });
  REQUIRE(values_are_in_range);

  const auto first_value = buffers.h_items.front();
  const bool has_distinct_values =
    std::any_of(buffers.h_items.begin(), buffers.h_items.end(), [first_value](const cuda::std::int32_t value) {
      return value != first_value;
    });
  REQUIRE(has_distinct_values);
}

CUB_TEST("c2h buffer generators populate checked CUDA buffers", "[c2h][buffers][generators]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  constexpr std::size_t num_items       = 256;
  constexpr cuda::std::int32_t expected = 42;
  const auto buffers = c2h::gen_buffers<cuda::std::int32_t>(stream, c2h::seed_t{1234}, num_items, expected, expected);

  REQUIRE(buffers.size() == num_items);
  REQUIRE(buffers.d_items.size() == num_items);
  REQUIRE(buffers.h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(buffers.h_items.begin(), buffers.h_items.end(), expected)) == num_items);

  constexpr cuda::std::int32_t host_expected = -17;
  const auto h_items =
    c2h::gen_host_buffer<cuda::std::int32_t>(stream, c2h::seed_t{5678}, num_items, host_expected, host_expected);
  REQUIRE(h_items.size() == num_items);
  REQUIRE(static_cast<std::size_t>(std::count(h_items.begin(), h_items.end(), host_expected)) == num_items);

  using custom_type = c2h::custom_type_t<c2h::equal_comparable_t>;
  custom_type custom_expected{};
  custom_expected.key = 13;
  custom_expected.val = 37;

  auto d_custom = c2h::make_device_buffer<custom_type>(stream, num_items, cuda::no_init);
  // The generator writes only key and val. Whole-object copies also read policy members and padding bytes.
  cuda::fill_bytes(stream, d_custom, 0);
  c2h::gen(c2h::seed_t{9012}, d_custom, custom_expected, custom_expected);
  const auto h_custom = c2h::make_host_buffer<custom_type>(stream, d_custom);
  stream.sync();

  const bool custom_values_match = std::all_of(h_custom.begin(), h_custom.end(), [&](const custom_type& value) {
    return value == custom_expected;
  });
  REQUIRE(custom_values_match);

  const auto custom_buffers =
    c2h::gen_buffers<custom_type>(stream, c2h::seed_t{9012}, num_items, custom_expected, custom_expected);
  REQUIRE_THAT(custom_buffers.d_items, c2h::Equals(stream, h_custom));
}
