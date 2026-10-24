// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/__algorithm/copy.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <sstream>
#include <string>

#include <cuda_runtime_api.h>

#include <c2h/catch2_test_helper.h>
#include <c2h/catch2_test_macros.h>
#include <c2h/checked_memory_resource.cuh>
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
[[nodiscard]] cuda::device_ref pointer_device(const T* ptr)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaSuccess == cudaPointerGetAttributes(&attributes, ptr));
  return cuda::device_ref{attributes.device};
}

inline void synchronize_device(cuda::device_ref device)
{
  const int target_device = device.get();
  int previous_device     = 0;
  REQUIRE(cudaSuccess == cudaGetDevice(&previous_device));
  if (previous_device != target_device)
  {
    REQUIRE(cudaSuccess == cudaSetDevice(target_device));
  }

  const cudaError_t sync_status    = cudaDeviceSynchronize();
  const cudaError_t restore_status = previous_device != target_device ? cudaSetDevice(previous_device) : cudaSuccess;

  REQUIRE(sync_status == cudaSuccess);
  REQUIRE(restore_status == cudaSuccess);
}
} // namespace detail

// Copies device items to an equally sized host range and waits for completion.
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

  if (::c2h::detail::is_default_stream(stream))
  {
    const auto device = ::c2h::detail::pointer_device(device_items.data());
    ::c2h::detail::synchronize_device(device);
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
template <typename T>
[[nodiscard]] T read_single(cuda::stream_ref stream, const cuda::device_buffer<T>& buffer)
{
  REQUIRE(buffer.size() == 1);

  const auto device = ::c2h::detail::pointer_device(buffer.data());
  auto value        = ::c2h::make_host_buffer<T>(stream, device, 1, cuda::no_init);
  ::c2h::copy_to_host(stream, buffer, value);
  return value[0];
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
    const auto device = actual.size() != 0 ? ::c2h::detail::pointer_device(actual.data()) : ::c2h::current_device();
    auto actual_host  = c2h::make_host_buffer<T>(stream, device, actual.size(), cuda::no_init);
    c2h::copy_to_host(stream, actual, actual_host);

    comparison_result = ::c2h::detail::compare_host_ranges(actual_host, expected);
    return comparison_result.actual_size == comparison_result.expected_size && comparison_result.total_mismatches == 0;
  }

protected:
  std::string describe() const override
  {
    std::stringstream ss;
    ::c2h::detail::print_comparison(comparison_result, ss);
    return ss.str();
  }

private:
  cuda::stream_ref stream;
  const HostItems& expected;
  mutable ::c2h::detail::vector_compare_result_t<T> comparison_result{};
};
} // namespace detail

template <typename HostItems>
[[nodiscard]] auto Equals(cuda::stream_ref stream, const HostItems& expected)
{
  return detail::device_buffer_equals_matcher<HostItems>{stream, expected};
}
} // namespace c2h
