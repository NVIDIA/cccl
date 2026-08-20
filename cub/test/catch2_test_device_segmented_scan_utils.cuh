// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/span>
#include <cuda/stream>

#include <cstddef>
#include <vector>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/checked_allocator.cuh>

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

template <typename T, typename DeviceItems>
void copy_to_host(cuda::stream_ref stream, const DeviceItems& device_items, std::vector<T>& host_items)
{
  REQUIRE(device_items.size() == host_items.size());

  if (is_default_stream(stream))
  {
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    auto copy_stream = cuda::stream{current_device()};
    cuda::copy_bytes(copy_stream, device_items, cuda::std::span<T>{host_items.data(), host_items.size()});
    copy_stream.sync();
  }
  else
  {
    cuda::copy_bytes(stream, device_items, cuda::std::span<T>{host_items.data(), host_items.size()});
    stream.sync();
  }
}

template <typename T>
void copy_to_device(cuda::stream_ref stream, cuda::std::span<const T> host_items, cuda::device_buffer<T>& device_items)
{
  REQUIRE(host_items.size() == device_items.size());

  if (is_default_stream(stream))
  {
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    auto copy_stream = cuda::stream{current_device()};
    cuda::copy_bytes(copy_stream, host_items, device_items);
    copy_stream.sync();
  }
  else
  {
    cuda::copy_bytes(stream, host_items, device_items);
  }
}

template <typename T>
void copy_to_device(cuda::stream_ref stream, const std::vector<T>& host_items, cuda::device_buffer<T>& device_items)
{
  copy_to_device(stream, cuda::std::span<const T>{host_items.data(), host_items.size()}, device_items);
}

template <typename T, typename Expected>
void require_equal(cuda::stream_ref stream, const cuda::device_buffer<T>& actual, const Expected& expected)
{
  REQUIRE(actual.size() == expected.size());

  std::vector<T> h_actual(actual.size());
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

  std::vector<T> value(1);
  copy_to_host(stream, buffer, value);
  return value[0];
}

template <typename T>
[[nodiscard]] cuda::device_buffer<T>
make_device_buffer_from_host(cuda::stream_ref stream, cuda::device_ref device, const std::vector<T>& host_items)
{
  auto device_items = c2h::make_device_buffer<T>(stream, device, host_items.size(), cuda::no_init);
  copy_to_device(stream, host_items, device_items);
  return device_items;
}

template <typename T, typename GeneratorT>
[[nodiscard]] std::vector<T> make_tabulated_vector(std::size_t num_items, GeneratorT generator)
{
  std::vector<T> result(num_items);
  for (std::size_t i = 0; i < result.size(); ++i)
  {
    result[i] = static_cast<T>(generator(i));
  }
  return result;
}
} // namespace segmented_scan_test
