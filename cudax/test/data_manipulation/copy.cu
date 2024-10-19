//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>

#include <cuda/experimental/buffer.cuh>
#include <cuda/experimental/data_manipulation.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <catch2/catch.hpp>
#include <utility.cuh>

TEST_CASE("Copy", "[copy]")
{
  cudax::stream s;
  cuda::mr::pinned_memory_resource res_host;

  cudax::mr::async_memory_resource res;
  const cudax::uninitialized_buffer<int, cuda::mr::device_accessible> b1(res, 42);
  cudax::uninitialized_buffer<int, cuda::mr::device_accessible> b2(res, 42);
  cudax::copy_bytes(s, b1, b2);

  std::vector<int> vec1(42, 1);
  std::vector<int> vec2(42, 1);

  cudax::copy_bytes(s, b1, vec1);
  cudax::copy_bytes(s, vec2, b2);

  cudax::copy_bytes(s, vec1, vec2);

  cudax::uninitialized_buffer<int, cuda::mr::host_accessible> b3(res_host, 42);
  cudax::copy_bytes(s, b1, b3);
  cudax::copy_bytes(s, std::move(b1), b3);
}
