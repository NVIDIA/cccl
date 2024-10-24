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

TEST_CASE("Copy", "[data_manipulation]")
{
  cudax::stream _stream;

  SECTION("Device resource")
  {
    cudax::mr::async_memory_resource device_resource;
    std::vector<int> host_vector(buffer_size);

    {
      cudax::uninitialized_async_buffer<int, cuda::mr::device_accessible> buffer(device_resource, _stream, buffer_size);
      cudax::fill_bytes(_stream, buffer, fill_byte);

      cudax::copy_bytes(_stream, buffer, host_vector);
      check_result_and_erase(_stream, host_vector);

      cudax::copy_bytes(_stream, std::move(buffer), host_vector);
      check_result_and_erase(_stream, host_vector);
    }
    {
      cudax::uninitialized_async_buffer<int, cuda::mr::device_accessible> not_yet_const_buffer(
        device_resource, _stream, buffer_size);
      cudax::fill_bytes(_stream, not_yet_const_buffer, fill_byte);

      const auto& const_buffer = not_yet_const_buffer;

      cudax::copy_bytes(_stream, const_buffer, host_vector);
      check_result_and_erase(_stream, host_vector);

      cudax::copy_bytes(_stream, const_buffer, cuda::std::span(host_vector));
      check_result_and_erase(_stream, host_vector);
    }
  }

  SECTION("Host and managed resource")
  {
    cuda::mr::managed_memory_resource managed_resource;
    cuda::mr::pinned_memory_resource host_resource;

    {
      cudax::uninitialized_buffer<int, cuda::mr::host_accessible> host_buffer(host_resource, buffer_size);
      cudax::uninitialized_buffer<int, cuda::mr::device_accessible> device_buffer(managed_resource, buffer_size);

      cudax::fill_bytes(_stream, host_buffer, fill_byte);

      cudax::copy_bytes(_stream, host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);

      cudax::copy_bytes(_stream, cuda::std::span(host_buffer), device_buffer);
      check_result_and_erase(_stream, device_buffer);
    }

    {
      cudax::uninitialized_buffer<int, cuda::mr::host_accessible> not_yet_const_host_buffer(host_resource, buffer_size);
      cudax::uninitialized_buffer<int, cuda::mr::device_accessible> device_buffer(managed_resource, buffer_size);
      cudax::fill_bytes(_stream, not_yet_const_host_buffer, fill_byte);

      const auto& const_host_buffer = not_yet_const_host_buffer;

      cudax::copy_bytes(_stream, const_host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);

      cudax::copy_bytes(_stream, cuda::std::span(const_host_buffer), device_buffer);
      check_result_and_erase(_stream, device_buffer);
    }
  }
  SECTION("Launch transform")
  {
    cuda::mr::pinned_memory_resource host_resource;
    cudax::weird_buffer input(host_resource, buffer_size);
    cudax::weird_buffer output(host_resource, buffer_size);

    memset(input.data, fill_byte, input.size * sizeof(int));

    cudax::copy_bytes(_stream, input, output);
    check_result_and_erase(_stream, cuda::std::span(output.data, output.size));
  }

  SECTION("Asymetric size")
  {
    cuda::mr::pinned_memory_resource host_resource;
    cudax::uninitialized_buffer<int, cuda::mr::host_accessible> host_buffer(host_resource, 1);
    cudax::fill_bytes(_stream, host_buffer, fill_byte);

    ::std::vector<int> vec(buffer_size, 0xdeadbeef);

    cudax::copy_bytes(_stream, host_buffer, vec);
    _stream.wait();

    CUDAX_REQUIRE(vec[0] == get_expected_value(fill_byte));
    CUDAX_REQUIRE(vec[1] == 0xdeadbeef);
  }
}
