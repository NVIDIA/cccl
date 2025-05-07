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

C2H_TEST("Fill", "[data_manipulation]")
{
  cudax::stream _stream;
  SECTION("Host resource")
  {
    cudax::legacy_pinned_memory_resource host_resource;
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
    cudax::legacy_pinned_memory_resource host_resource;
    cudax::weird_buffer buffer(host_resource, buffer_size);

    cudax::fill_bytes(_stream, buffer, fill_byte);
    check_result_and_erase(_stream, cuda::std::span(buffer.data, buffer.size));
  }
}

C2H_TEST("Mdspan Fill", "[data_manipulation]")
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
    cudax::legacy_pinned_memory_resource host_resource;
    using static_extents = cuda::std::extents<size_t, 2, 3, 4>;
    auto size            = cuda::std::layout_left::mapping<static_extents>().required_span_size();
    cudax::weird_buffer<cuda::std::mdspan<int, static_extents>> buffer(host_resource, size);

    cudax::fill_bytes(stream, buffer, fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data, buffer.size));
  }
}

C2H_TEST("Non exhaustive mdspan fill_bytes", "[data_manipulation]")
{
  cudax::stream stream;
  {
    auto fake_strided_mdspan = create_fake_strided_mdspan();

    try
    {
      cudax::fill_bytes(stream, fake_strided_mdspan, fill_byte);
    }
    catch (const ::std::invalid_argument& e)
    {
      CHECK(e.what() == ::std::string("fill_bytes supports only exhaustive mdspans"));
    }
  }
}
