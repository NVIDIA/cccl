// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_adjacent_difference.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/functional>

#include <cstdint>

#include <c2h/catch2_test_helper.h>

// Guard: the legacy memory-size query call with all defaults (no explicit difference_op,
// no explicit stream) must resolve unambiguously to the legacy temp-storage overload
// when the env passthrough overload is also visible. If the env overload's SFINAE
// is too loose, this becomes "ambiguous overload" or silently dispatches to env.

C2H_TEST("DeviceAdjacentDifference::SubtractLeftCopy legacy size-query is unambiguous", "[adjacent_difference][device]")
{
  int* d_in    = nullptr;
  int* d_out   = nullptr;
  size_t bytes = 0;
  int n        = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractLeftCopy(nullptr, bytes, d_in, d_out, n));
}

C2H_TEST("DeviceAdjacentDifference::SubtractLeft legacy size-query is unambiguous", "[adjacent_difference][device]")
{
  int* d_in    = nullptr;
  size_t bytes = 0;
  int n        = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractLeft(nullptr, bytes, d_in, n));
}

C2H_TEST("DeviceAdjacentDifference::SubtractRightCopy legacy size-query is unambiguous",
         "[adjacent_difference][device]")
{
  int* d_in    = nullptr;
  int* d_out   = nullptr;
  size_t bytes = 0;
  int n        = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractRightCopy(nullptr, bytes, d_in, d_out, n));
}

C2H_TEST("DeviceAdjacentDifference::SubtractRight legacy size-query is unambiguous", "[adjacent_difference][device]")
{
  int* d_in    = nullptr;
  size_t bytes = 0;
  int n        = 0;

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractRight(nullptr, bytes, d_in, n));
}

C2H_TEST("DeviceAdjacentDifference::SubtractLeftCopy accepts cuda::device_buffer input",
         "[adjacent_difference][device]")
{
  using type = std::int32_t;

  cuda::stream stream{cuda::devices[0]};
  auto input = cuda::make_device_buffer<type>(stream, cuda::devices[0], {2, 5, 9, 14, 20});
  c2h::device_vector<type> output(input.size());
  auto output_it = thrust::raw_pointer_cast(output.data());

  void* temporary_storage     = nullptr;
  std::size_t temporary_bytes = 0;
  REQUIRE(
    cudaSuccess
    == cub::DeviceAdjacentDifference::SubtractLeftCopy(
      temporary_storage, temporary_bytes, input.begin(), output_it, input.size(), cuda::std::minus<>{}, stream.get()));

  c2h::device_vector<std::uint8_t> temporary_buffer(temporary_bytes);
  temporary_storage = thrust::raw_pointer_cast(temporary_buffer.data());
  REQUIRE(
    cudaSuccess
    == cub::DeviceAdjacentDifference::SubtractLeftCopy(
      temporary_storage, temporary_bytes, input.begin(), output_it, input.size(), cuda::std::minus<>{}, stream.get()));
  stream.sync();

  const c2h::host_vector<type> expected{2, 3, 4, 5, 6};
  REQUIRE(output == expected);
}

// todo(giannis): extract examples from the docs to literalinclude extracts here
