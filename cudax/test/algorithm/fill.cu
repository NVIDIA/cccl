//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "common.cuh"

TEST_CASE("Fill", "[data_manipulation]")
{
  cudax::stream _stream;
  SECTION("Host resource")
  {
    cudax::mr::pinned_memory_resource host_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(host_resource, buffer_size);

    cudax::fill_bytes(_stream, buffer, fill_byte);

    check_result_and_erase(_stream, cuda::std::span(buffer));
  }

  SECTION("Device resource")
  {
    cudax::mr::device_memory_resource device_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(device_resource, buffer_size);
    cudax::fill_bytes(_stream, buffer, fill_byte);

    std::vector<int> host_vector(42);
    CUDART(cudaMemcpyAsync(
      host_vector.data(), buffer.data(), buffer.size() * sizeof(int), cudaMemcpyDefault, _stream.get()));

    check_result_and_erase(_stream, host_vector);
  }
  SECTION("Launch transform")
  {
    cudax::mr::pinned_memory_resource host_resource;
    cudax::weird_buffer buffer(host_resource, buffer_size);

    cudax::fill_bytes(_stream, buffer, fill_byte);
    check_result_and_erase(_stream, cuda::std::span(buffer.data, buffer.size));
  }
}
