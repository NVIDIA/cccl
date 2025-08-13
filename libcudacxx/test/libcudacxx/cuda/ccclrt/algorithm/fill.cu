//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "common.cuh"

C2H_CCCLRT_TEST("Fill", "[algorithm]")
{
  cuda::stream _stream{cuda::device_ref{0}};
  SECTION("Host memory")
  {
    test_buffer<int> buffer(test_buffer_type::pinned, buffer_size);

    cuda::fill_bytes(_stream, buffer, fill_byte);

    check_result_and_erase(_stream, cuda::std::span(buffer));
  }

  SECTION("Device memory")
  {
    test_buffer<int> buffer(test_buffer_type::device, buffer_size);
    cuda::fill_bytes(_stream, buffer, fill_byte);

    std::vector<int> host_vector(42);
    {
      cuda::__ensure_current_context ctx_setter{cuda::device_ref{0}};
      CUDART(cudaMemcpyAsync(
        host_vector.data(), buffer.data(), buffer.size() * sizeof(int), cudaMemcpyDefault, _stream.get()));
    }

    check_result_and_erase(_stream, host_vector);

    cuda::std::span<int> span(buffer.data(), 0);
    cuda::fill_bytes(_stream, span, fill_byte);
    printf("0 sized span: %p\n", span.data());
  }
}

C2H_CCCLRT_TEST("Mdspan Fill", "[algorithm]")
{
  cuda::stream stream{cuda::device_ref{0}};
  {
    cuda::std::dextents<size_t, 3> dynamic_extents{1, 2, 3};
    auto buffer = make_buffer_for_mdspan(dynamic_extents, 0);
    cuda::std::mdspan<int, decltype(dynamic_extents)> dynamic_mdspan(buffer.data(), dynamic_extents);

    cuda::fill_bytes(stream, dynamic_mdspan, fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data(), buffer.size()));
  }
  {
    cuda::std::extents<size_t, 2, cuda::std::dynamic_extent, 4> mixed_extents{1};
    auto buffer = make_buffer_for_mdspan(mixed_extents, 0);
    cuda::std::mdspan<int, decltype(mixed_extents)> mixed_mdspan(buffer.data(), mixed_extents);

    cuda::fill_bytes(stream, cuda::std::move(mixed_mdspan), fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data(), buffer.size()));
  }
}

C2H_CCCLRT_TEST("Non exhaustive mdspan fill_bytes", "[data_manipulation]")
{
  cuda::stream stream{cuda::device_ref{0}};
  {
    auto fake_strided_mdspan = create_fake_strided_mdspan();

    try
    {
      cuda::fill_bytes(stream, fake_strided_mdspan, fill_byte);
    }
    catch (const ::std::invalid_argument& e)
    {
      CHECK(e.what() == ::std::string("fill_bytes supports only exhaustive mdspans"));
    }
  }
}
