// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/buffer>
#include <cuda/stream>

#include <cstddef>

#include <c2h/checked_memory_resource.cuh>

namespace segmented_scan_test
{
// Requires: querying the stream's device must succeed.
template <typename T, typename GeneratorT>
[[nodiscard]] auto make_tabulated_host_buffer(cuda::stream_ref stream, std::size_t num_items, GeneratorT generator)
{
  auto result = c2h::make_host_buffer<T>(stream, num_items, cuda::no_init);
  for (std::size_t i = 0; i < result.size(); ++i)
  {
    result[i] = static_cast<T>(generator(i));
  }
  return result;
}
} // namespace segmented_scan_test
