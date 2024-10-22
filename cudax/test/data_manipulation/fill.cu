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
  cudax::stream stream;
  SECTION("Host resource")
  {
    cuda::mr::pinned_memory_resource host_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(host_resource, buffer_size);

    cudax::fill_bytes(stream, buffer, fill_byte);

    check_result_and_erase(stream, cuda::std::span(buffer));
  }

  SECTION("Device resource")
  {
    cuda::mr::device_memory_resource device_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(device_resource, buffer_size);
    cudax::fill_bytes(stream, buffer, fill_byte);

    std::vector<int> host_vector(42);
    CUDART(
      cudaMemcpyAsync(host_vector.data(), buffer.data(), buffer.size() * sizeof(int), cudaMemcpyDefault, stream.get()));

    check_result_and_erase(stream, host_vector);
  }
  SECTION("Launch transform")
  {
    cuda::mr::pinned_memory_resource host_resource;
    cudax::weird_buffer buffer(host_resource, buffer_size);

    cudax::fill_bytes(stream, buffer, fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data, buffer.size));
  }
}
