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
    cudax::pinned_memory_resource host_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(host_resource, buffer_size);

    cudax::fill_bytes(_stream, buffer, fill_byte);

    check_result_and_erase(_stream, cuda::std::span(buffer));
  }

  SECTION("Device resource")
  {
    cudax::device_memory_resource device_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(device_resource, buffer_size);
    cudax::fill_bytes(_stream, buffer, fill_byte);

    std::vector<int> host_vector(42);
    CUDART(cudaMemcpyAsync(
      host_vector.data(), buffer.data(), buffer.size() * sizeof(int), cudaMemcpyDefault, _stream.get()));

    check_result_and_erase(_stream, host_vector);
  }
  SECTION("Launch transform")
  {
    cudax::pinned_memory_resource host_resource;
    cudax::weird_buffer buffer(host_resource, buffer_size);

    cudax::fill_bytes(_stream, buffer, fill_byte);
    check_result_and_erase(_stream, cuda::std::span(buffer.data, buffer.size));
  }
}

TEST_CASE("Mdspan Fill", "[data_manipulation]")
{
  cudax::stream stream;
  {
    cuda::std::dextents<size_t, 3> dynamic_extents{1, 2, 3};
    auto buffer = make_buffer_for_mdspan(dynamic_extents, 0);
    cuda::std::mdspan<int, decltype(dynamic_extents)> dynamic_mdspan(buffer.data(), dynamic_extents);

    cudax::fill_bytes(stream, dynamic_mdspan, fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data(), buffer.size()));
  }
  {
    cuda::std::extents<size_t, 2, cuda::std::dynamic_extent, 4> mixed_extents{1};
    auto buffer = make_buffer_for_mdspan(mixed_extents, 0);
    cuda::std::mdspan<int, decltype(mixed_extents)> mixed_mdspan(buffer.data(), mixed_extents);

    cudax::fill_bytes(stream, cuda::std::move(mixed_mdspan), fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data(), buffer.size()));
  }
  {
    using static_extents = cuda::std::extents<size_t, 2, 3, 4>;
    auto size            = cuda::std::layout_left::mapping<static_extents>().required_span_size();
    cudax::weird_buffer<cuda::std::mdspan<int, static_extents>> buffer(cudax::pinned_memory_resource{}, size);

    cudax::fill_bytes(stream, buffer, fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data, buffer.size));
  }
}
