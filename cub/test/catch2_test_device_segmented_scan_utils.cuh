// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/initializer_list>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cub_test_macros_lightweight.h"
#include <c2h/checked_memory_resource.cuh>

template <typename InputT, typename OutputT = InputT>
struct type_pair
{
  using input_t  = InputT;
  using output_t = OutputT;
};

template <typename InputIt, typename OutputIt, typename InitValueT, typename BinaryOp>
void compute_exclusive_scan_reference(InputIt first, InputIt last, OutputIt result, InitValueT init, BinaryOp op)
{
  using value_t  = cub::detail::it_value_t<InputIt>;
  using accum_t  = ::cuda::std::__accumulator_t<BinaryOp, value_t, InitValueT>;
  using output_t = cub::detail::it_value_t<OutputIt>;
  accum_t acc    = static_cast<accum_t>(init);
  for (; first != last; ++first)
  {
    const auto v = *first;
    *result++    = static_cast<output_t>(acc);
    acc          = op(acc, v);
  }
}

template <typename InputIt, typename OutputIt, typename BinaryOp, typename InitValueT>
void compute_inclusive_scan_reference(InputIt first, InputIt last, OutputIt result, BinaryOp op, InitValueT init)
{
  using value_t  = cub::detail::it_value_t<InputIt>;
  using accum_t  = ::cuda::std::__accumulator_t<BinaryOp, value_t, InitValueT>;
  using output_t = cub::detail::it_value_t<OutputIt>;
  accum_t acc    = static_cast<accum_t>(init);
  for (; first != last; ++first)
  {
    acc       = op(acc, *first);
    *result++ = static_cast<output_t>(acc);
  }
}

namespace segmented_scan_test
{
[[nodiscard]] inline cuda::device_ref current_device()
{
  int device = 0;
  REQUIRE(cudaSuccess == cudaGetDevice(&device));
  return cuda::device_ref{device};
}

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

template <typename T, typename... Args>
[[nodiscard]] auto make_host_buffer(cuda::stream_ref stream, cuda::device_ref device, Args&&... args)
{
  return c2h::make_host_buffer<T>(stream, device, cuda::std::forward<Args>(args)...);
}

template <typename T>
[[nodiscard]] auto
make_host_buffer(cuda::stream_ref stream, cuda::device_ref device, std::size_t num_items, const T& value)
{
  auto result = c2h::make_host_buffer<T>(stream, device, num_items, cuda::no_init);
  for (std::size_t i = 0; i < result.size(); ++i)
  {
    result[i] = value;
  }

  return result;
}

template <typename T>
[[nodiscard]] auto
make_host_buffer(cuda::stream_ref stream, cuda::device_ref device, cuda::std::initializer_list<T> values)
{
  auto result = c2h::make_host_buffer<T>(stream, device, values.size(), cuda::no_init);
  auto out    = result.begin();
  for (const auto& value : values)
  {
    *out = value;
    ++out;
  }

  return result;
}

template <typename DeviceItems, typename HostItems>
void copy_to_host(cuda::stream_ref stream, const DeviceItems& device_items, HostItems& host_items)
{
  REQUIRE(device_items.size() == host_items.size());
  if (device_items.size() == 0)
  {
    return;
  }

  using value_t = typename HostItems::value_type;

  if (is_default_stream(stream))
  {
    const auto device = pointer_device(device_items.data());
    synchronize_device(device);
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

// Enqueues a host-to-device copy. For non-default streams, the copy may still be
// pending when this returns; callers must keep the host range alive and unchanged
// until the stream has synchronized or otherwise completed the copy.
template <typename T, typename HostItems>
void enqueue_copy_to_device(cuda::stream_ref stream, const HostItems& host_items, cuda::device_buffer<T>& device_items)
{
  REQUIRE(host_items.size() == device_items.size());
  if (device_items.size() == 0)
  {
    return;
  }

  const auto host_span = cuda::std::span<const T>{host_items.data(), host_items.size()};

  if (is_default_stream(stream))
  {
    const auto device = pointer_device(device_items.data());
    synchronize_device(device);
    auto copy_stream = cuda::stream{device};
    cuda::copy_bytes(copy_stream, host_span, device_items);
    copy_stream.sync();
  }
  else
  {
    cuda::copy_bytes(stream, host_span, device_items);
  }
}

template <typename T>
void enqueue_copy_to_device(
  cuda::stream_ref stream, cuda::std::span<const T> host_items, cuda::device_buffer<T>& device_items)
{
  REQUIRE(host_items.size() == device_items.size());
  if (device_items.size() == 0)
  {
    return;
  }

  if (is_default_stream(stream))
  {
    const auto device = pointer_device(device_items.data());
    synchronize_device(device);
    auto copy_stream = cuda::stream{device};
    cuda::copy_bytes(copy_stream, host_items, device_items);
    copy_stream.sync();
  }
  else
  {
    cuda::copy_bytes(stream, host_items, device_items);
  }
}

template <typename T, typename Expected>
void require_equal(cuda::stream_ref stream, const cuda::device_buffer<T>& actual, const Expected& expected)
{
  REQUIRE(actual.size() == expected.size());

  const auto device = actual.size() != 0 ? pointer_device(actual.data()) : current_device();
  auto h_actual     = make_host_buffer<T>(stream, device, actual.size(), cuda::no_init);
  copy_to_host(stream, actual, h_actual);

  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    REQUIRE(h_actual[i] == expected[i]);
  }
}

template <typename T>
[[nodiscard]] T read_single(cuda::stream_ref stream, const cuda::device_buffer<T>& buffer)
{
  REQUIRE(buffer.size() == 1);

  const auto device = pointer_device(buffer.data());
  auto value        = make_host_buffer<T>(stream, device, 1, cuda::no_init);
  copy_to_host(stream, buffer, value);
  return value[0];
}

template <typename HostItems>
[[nodiscard]] auto
make_device_buffer_from_host(cuda::stream_ref stream, cuda::device_ref device, const HostItems& host_items)
{
  using T           = typename HostItems::value_type;
  auto device_items = c2h::make_device_buffer<T>(stream, device, host_items.size(), cuda::no_init);
  enqueue_copy_to_device(stream, host_items, device_items);
  return device_items;
}

template <typename HostItems>
void make_device_buffer_from_host(cuda::stream_ref, cuda::device_ref, const HostItems&&) = delete;

template <typename T, typename GeneratorT>
[[nodiscard]] auto make_tabulated_host_buffer(
  cuda::stream_ref stream, cuda::device_ref device, std::size_t num_items, GeneratorT generator)
{
  auto result = make_host_buffer<T>(stream, device, num_items, cuda::no_init);
  for (std::size_t i = 0; i < result.size(); ++i)
  {
    result[i] = static_cast<T>(generator(i));
  }
  return result;
}

template <typename Actual, typename Expected>
void require_ranges_equal(const Actual& actual, const Expected& expected)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    REQUIRE(actual[i] == expected[i]);
  }
}
} // namespace segmented_scan_test
