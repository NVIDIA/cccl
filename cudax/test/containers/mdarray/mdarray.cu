//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/mdspan>

#include <cuda/experimental/__container/mdarray_device.cuh>

#include "testing.cuh"

C2H_CCCLRT_TEST("cudax::mdarray", "[container][mdarray]")
{
  cuda::device_memory_pool pool{cuda::device_ref{0}};
  using extents_type = cuda::std::dims<2>;
  using mdarray_type = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  mdarray_type mdarray{extents_type{2, 3}};

  CUDAX_REQUIRE(mdarray.size() == 6);
  CUDAX_REQUIRE(mdarray.extent(0) == 2);
  CUDAX_REQUIRE(mdarray.extent(1) == 3);
}
