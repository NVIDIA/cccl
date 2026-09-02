//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/mdspan>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cuda/experimental/fill_bytes.cuh>

#include <cstring>

#include "testing.cuh"

TEST_CASE("fill_bytes mdspan documentation example", "[fill_bytes][example]")
{
  // example-begin fill-bytes-mdspan
  using extents_t = cuda::std::dims<2>;
  extents_t extents{2, 3};

  thrust::device_vector<int> device_data(extents.extent(0) * extents.extent(1));
  int* dst_ptr = thrust::raw_pointer_cast(device_data.data());
  cuda::device_mdspan<int, extents_t> dst(dst_ptr, extents);

  cuda::stream stream{cuda::device_ref{0}};
  cuda::experimental::fill_bytes(dst, uint32_t{0xFF00FF00}, stream);
  // example-end fill-bytes-mdspan

  stream.sync();

  const thrust::host_vector<int> actual(device_data);
  for (const int value : actual)
  {
    uint32_t value_bits{};
    std::memcpy(&value_bits, &value, sizeof(value_bits));
    REQUIRE(value_bits == uint32_t{0xFF00FF00});
  }
}
