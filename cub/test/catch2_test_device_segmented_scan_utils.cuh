// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstddef>

#include <cuda_runtime_api.h>

#include "catch2_test_scan_reference.cuh"
#include "cub_test_macros.h"
#include <c2h/buffer_utils.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/device_and_stream.h>

template <typename InputT, typename OutputT = InputT>
struct type_pair
{
  using input_t  = InputT;
  using output_t = OutputT;
};

namespace segmented_scan_test
{
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

  if (::c2h::detail::is_default_stream(stream))
  {
    const auto device = ::c2h::detail::pointer_device(device_items.data());
    ::c2h::detail::synchronize_device(device);
    auto copy_stream = cuda::stream{device};
    cuda::copy_bytes(copy_stream, host_span, device_items);
    copy_stream.sync();
  }
  else
  {
    cuda::copy_bytes(stream, host_span, device_items);
  }
}

template <typename T, typename GeneratorT>
[[nodiscard]] auto make_tabulated_host_buffer(
  cuda::stream_ref stream, cuda::device_ref device, std::size_t num_items, GeneratorT generator)
{
  auto result = c2h::make_host_buffer<T>(stream, device, num_items, cuda::no_init);
  for (std::size_t i = 0; i < result.size(); ++i)
  {
    result[i] = static_cast<T>(generator(i));
  }
  return result;
}
} // namespace segmented_scan_test
