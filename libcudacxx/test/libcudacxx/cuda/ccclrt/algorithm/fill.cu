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
    auto buffer = make_pinned_memory_buffer<int>(_stream, buffer_size);

    cuda::fill_bytes(_stream, buffer, fill_byte);

    check_result_and_erase(_stream, buffer);
  }

  SECTION("Device memory")
  {
    auto buffer = cuda::make_device_buffer<int>(_stream, cuda::device_ref{0}, buffer_size, cuda::no_init);
    cuda::fill_bytes(_stream, buffer, fill_byte);

    auto host_vector = make_pinned_memory_buffer<int>(_stream, buffer_size);
    cuda::copy_bytes(_stream, buffer, host_vector);

    check_result_and_erase(_stream, host_vector);

    auto span = buffer.first(0);
    cuda::fill_bytes(_stream, span, fill_byte);
  }
}

template <typename T>
void check_fill_pattern(cuda::stream_ref stream, const test_buffer<T>& device_buf, T expected)
{
  std::vector<T> host_vec(device_buf.size());
  {
    cuda::__ensure_current_context ctx_setter{cuda::device_ref{0}};
    CUDART(
      cudaMemcpyAsync(host_vec.data(), device_buf.data(), device_buf.size_bytes(), cudaMemcpyDefault, stream.get()));
  }
  stream.sync();
  for (const auto& val : host_vec)
  {
    CCCLRT_REQUIRE(val == expected);
  }
}

C2H_CCCLRT_TEST("Fill uint16", "[algorithm]")
{
  cuda::stream stream{cuda::device_ref{0}};
  constexpr ::cuda::std::uint16_t pattern = 0xABCD;

  SECTION("Device memory")
  {
    test_buffer<::cuda::std::uint16_t> buffer(test_buffer_type::device, buffer_size);
    cuda::fill_bytes(stream, buffer, pattern);
    check_fill_pattern(stream, buffer, pattern);
  }

  SECTION("Pinned memory")
  {
    test_buffer<::cuda::std::uint16_t> buffer(test_buffer_type::pinned, buffer_size);
    cuda::fill_bytes(stream, buffer, pattern);
    check_fill_pattern(stream, buffer, pattern);
  }
}

C2H_CCCLRT_TEST("Fill uint32", "[algorithm]")
{
  cuda::stream stream{cuda::device_ref{0}};
  constexpr ::cuda::std::uint32_t pattern = 0xDEADBEEF;

  SECTION("Device memory")
  {
    test_buffer<::cuda::std::uint32_t> buffer(test_buffer_type::device, buffer_size);
    cuda::fill_bytes(stream, buffer, pattern);
    check_fill_pattern(stream, buffer, pattern);
  }

  SECTION("Pinned memory")
  {
    test_buffer<::cuda::std::uint32_t> buffer(test_buffer_type::pinned, buffer_size);
    cuda::fill_bytes(stream, buffer, pattern);
    check_fill_pattern(stream, buffer, pattern);
  }
}

C2H_CCCLRT_TEST("Mdspan Fill", "[algorithm]")
{
  cuda::stream stream{cuda::device_ref{0}};
  {
    cuda::std::dextents<size_t, 3> dynamic_extents{1, 2, 3};
    auto buffer = make_buffer_for_mdspan(stream, dynamic_extents, 0);
    cuda::std::mdspan<int, decltype(dynamic_extents)> dynamic_mdspan(buffer.data(), dynamic_extents);

    cuda::fill_bytes(stream, dynamic_mdspan, fill_byte);
    check_result_and_erase(stream, buffer);
  }
  {
    cuda::std::extents<size_t, 2, cuda::std::dynamic_extent, 4> mixed_extents{1};
    auto buffer = make_buffer_for_mdspan(stream, mixed_extents, 0);
    cuda::std::mdspan<int, decltype(mixed_extents)> mixed_mdspan(buffer.data(), mixed_extents);

    cuda::fill_bytes(stream, cuda::std::move(mixed_mdspan), fill_byte);
    check_result_and_erase(stream, buffer);
  }
}

C2H_CCCLRT_TEST("Mdspan Fill uint32", "[algorithm]")
{
  cuda::stream stream{cuda::device_ref{0}};
  constexpr ::cuda::std::uint32_t pattern = 0xCAFEBABE;

  cuda::std::dextents<size_t, 2> extents{4, 8};
  test_buffer<::cuda::std::uint32_t> buffer(test_buffer_type::pinned, 4 * 8);
  cuda::std::mdspan<::cuda::std::uint32_t, decltype(extents)> mdspan(buffer.data(), extents);

  cuda::fill_bytes(stream, mdspan, pattern);
  check_fill_pattern(stream, buffer, pattern);
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
