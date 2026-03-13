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

#include <cuda/std/mdspan>
#include <cuda/stream>

#include <cuda/experimental/__copy/mdspan_d2d.cuh>

#include "testing.cuh"

static const cuda::stream stream{cuda::device_ref{0}};

TEST_CASE("copy d2d 2D layout_right", "[copy][d2d][2d]")
{
  constexpr int M     = 4;
  constexpr int N     = 8;
  constexpr int total = M * N;
  thrust::host_vector<int> host_data(total);
  for (int i = 0; i < total; ++i)
  {
    host_data[i] = i;
  }
  thrust::device_vector<int> d_src = host_data;
  thrust::device_vector<int> d_dst(total, 0);

  using extents = cuda::std::extents<int, M, N>;
  cuda::device_mdspan<int, extents> src(thrust::raw_pointer_cast(d_src.data()));
  cuda::device_mdspan<int, extents> dst(thrust::raw_pointer_cast(d_dst.data()));

  cuda::experimental::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<int> result(d_dst);
  CUDAX_REQUIRE(result == host_data);
}
