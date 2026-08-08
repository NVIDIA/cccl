//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cuda/experimental/__cuco/detail/utility/memcpy_async.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

namespace
{
// Does a round trip copy from host to device and back to host on the given
// stream.  Returns the number of mismatches from the round trip.
[[nodiscard]] ::cuda::std::size_t round_trip_mismatches(::cuda::stream_ref stream)
{
  constexpr ::cuda::std::size_t num_items = 1024;
  constexpr ::cuda::std::size_t num_bytes = sizeof(int) * num_items;

  auto device_mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});
  ::cuda::mr::legacy_pinned_memory_resource host_mr{};

  ::cuda::host_buffer<int> src{stream, host_mr, num_items, ::cuda::no_init};
  ::cuda::device_buffer<int> device{stream, device_mr, num_items, ::cuda::no_init};
  ::cuda::host_buffer<int> dst{stream, host_mr, num_items, ::cuda::no_init};

  // Seeding the src data and marking all destination data
  // with -1 which should not be present when we check the
  // host data again.
  for (::cuda::std::size_t i = 0; i < num_items; ++i)
  {
    src.data()[i] = static_cast<int>(i);
    dst.data()[i] = -1;
  }

  // Executing the round trip.
  cudax::cuco::detail::__memcpy_async(device.data(), src.data(), num_bytes, stream);
  cudax::cuco::detail::__memcpy_async(dst.data(), device.data(), num_bytes, stream);
  stream.sync();

  // Calculating the mismatches.  A -1 still sitting in dst means the copy
  // never landed at all.
  ::cuda::std::size_t mismatches = 0;
  for (::cuda::std::size_t i = 0; i < num_items; ++i)
  {
    mismatches += dst.data()[i] != static_cast<int>(i);
  }
  return mismatches;
}
} // namespace

// Tests a normal non-null stream.  The legacy NULL stream is not covered:
// cuda::copy_bytes forwards it to cuMemcpyBatchAsync on CTK 13.0+, which
// rejects it, and no cuco call site passes it.
C2H_TEST("cuco memcpy_async round-trips on an explicit stream", "[memcpy_async]")
{
  ::cuda::stream stream{::cuda::device_ref{0}};
  REQUIRE(round_trip_mismatches(stream) == 0);
}
