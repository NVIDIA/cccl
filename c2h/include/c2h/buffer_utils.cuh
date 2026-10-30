// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/__algorithm/copy.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <sstream>
#include <string>

#include <cuda_runtime_api.h>

#include <c2h/catch2_test_helper.h>
#include <c2h/catch2_test_macros.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/detail/current_device.cuh>
#include <c2h/device_and_stream.h>

namespace c2h
{
namespace detail
{
[[nodiscard]] inline bool is_default_stream(cuda::stream_ref stream) noexcept
{
  return stream.get() == cudaStream_t{};
}

// Default stream handles do not identify a device; infer it from the CUDA
// allocation before creating a temporary copy stream.
template <typename T>
[[nodiscard]] cuda::device_ref get_allocation_device(const T* ptr)
{
  int device{};
  const auto status = cuda::__driver::__pointerGetAttributeNoThrow<CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(device, ptr);
  REQUIRE(status == cudaSuccess);
  return cuda::device_ref{device};
}
} // namespace detail

// Copies device items to an equally sized host range and waits for completion.
// Requires: writes to device_items are ordered on stream or have already completed.
// Note: The legacy default stream is supported.
template <typename DeviceItems, typename HostItems>
void copy_to_host(cuda::stream_ref stream, const DeviceItems& device_items, HostItems& host_items)
{
  static_assert(cuda::std::is_same_v<typename DeviceItems::value_type, typename HostItems::value_type>,
                "copy_to_host requires matching element types");
  REQUIRE(device_items.size() == host_items.size());
  if (device_items.size() == 0)
  {
    return;
  }

  using value_t = typename HostItems::value_type;

  if (detail::is_default_stream(stream))
  {
    const auto device = detail::get_allocation_device(device_items.data());
    detail::assert_current_device(device.get());
    stream.sync();
    auto copy_stream = cuda::stream{device};
    cuda::copy_bytes(copy_stream, device_items, cuda::std::span<value_t>{host_items.data(), host_items.size()});
    copy_stream.sync();
  }
  else
  {
    cuda::copy_bytes(stream, device_items, cuda::std::span<value_t>{host_items.data(), host_items.size()});
    stream.sync();
  }
}

// Reads the only element of a device buffer and waits for the copy to complete.
// Requires: buffer contains exactly one element.
template <typename T, cuda::std::size_t Extent>
[[nodiscard]] cuda::std::remove_cv_t<T> read_single(cuda::stream_ref stream, cuda::std::span<T, Extent> device_items)
{
  static_assert(Extent == cuda::std::dynamic_extent || Extent == 1, "read single requires exactly one element");
  REQUIRE(device_items.size() == 1);

  using value_t = cuda::std::remove_cv_t<T>;
  alignas(value_t) cuda::std::byte storage[sizeof(value_t)];
  auto host_items = cuda::std::span<value_t>{reinterpret_cast<value_t*>(storage), 1};

  c2h::copy_to_host(stream, device_items, host_items);
  return host_items[0];
}

template <typename T>
[[nodiscard]] T read_single(cuda::stream_ref stream, const cuda::device_buffer<T>& buffer)
{
  return c2h::read_single(stream, buffer.subspan(0));
}

namespace detail
{
// The expected host range must provide value_type, size(), and indexed access,
// and must remain alive while the matcher is used.
template <typename HostItems>
struct device_buffer_equals_matcher : Catch::Matchers::MatcherGenericBase
{
  using T = typename HostItems::value_type;

  device_buffer_equals_matcher(cuda::stream_ref stream, const HostItems& expected)
      : stream{stream}
      , expected{expected}
  {}

  bool match(const cuda::device_buffer<T>& actual) const
  {
    const auto device = actual.size() != 0 ? detail::get_allocation_device(actual.data()) : c2h::current_device();
    auto actual_host  = c2h::make_host_buffer<T>(stream, device, actual.size(), cuda::no_init);
    c2h::copy_to_host(stream, actual, actual_host);

    comparison_result = detail::compare_host_ranges(actual_host, expected);
    return comparison_result.actual_size == comparison_result.expected_size && comparison_result.total_mismatches == 0;
  }

protected:
  std::string describe() const override
  {
    std::stringstream ss;
    detail::print_comparison(comparison_result, ss);
    return ss.str();
  }

private:
  cuda::stream_ref stream;
  const HostItems& expected;
  mutable detail::vector_compare_result_t<T> comparison_result{};
};
} // namespace detail

template <typename HostItems>
[[nodiscard]] auto Equals(cuda::stream_ref stream, const HostItems& expected)
{
  return detail::device_buffer_equals_matcher<HostItems>{stream, expected};
}
} // namespace c2h
