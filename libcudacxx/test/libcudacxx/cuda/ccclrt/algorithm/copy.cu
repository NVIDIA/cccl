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

C2H_CCCLRT_TEST("1d Copy", "[algorithm]")
{
  cuda::stream _stream{cuda::device_ref{0}};

  SECTION("Device resource")
  {
    std::vector<int> host_vector(buffer_size);

    {
      test_buffer<int> buffer(test_buffer_type::device, buffer_size);
      cuda::fill_bytes(_stream, buffer, fill_byte);

      cuda::copy_bytes(_stream, buffer, host_vector);
      check_result_and_erase(_stream, host_vector);

      cuda::copy_bytes(_stream, std::move(buffer), host_vector);
      check_result_and_erase(_stream, host_vector);
    }
    {
      test_buffer<int> not_yet_const_buffer(test_buffer_type::device, buffer_size);
      cuda::fill_bytes(_stream, not_yet_const_buffer, fill_byte);

      const auto& const_buffer = not_yet_const_buffer;

      cuda::copy_bytes(_stream, const_buffer, host_vector);
      check_result_and_erase(_stream, host_vector);

      cuda::copy_bytes(_stream, const_buffer, cuda::std::span(host_vector));
      check_result_and_erase(_stream, host_vector);

      ::cuda::std::span<int> span(const_buffer.data(), 0);
      cuda::copy_bytes(_stream, span, host_vector);
      printf("0 sized span: %p\n", span.data());
    }
  }

  SECTION("Host and managed resource")
  {
    {
      test_buffer<int> host_buffer(test_buffer_type::pinned, buffer_size);
      test_buffer<int> device_buffer(test_buffer_type::managed, buffer_size);

      cuda::fill_bytes(_stream, host_buffer, fill_byte);

      cuda::copy_bytes(_stream, host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);

      cuda::copy_bytes(_stream, cuda::std::span(host_buffer), device_buffer);
      check_result_and_erase(_stream, device_buffer);
    }

    {
      test_buffer<int> not_yet_const_host_buffer(test_buffer_type::pinned, buffer_size);
      test_buffer<int> device_buffer(test_buffer_type::managed, buffer_size);
      cuda::fill_bytes(_stream, not_yet_const_host_buffer, fill_byte);

      const auto& const_host_buffer = not_yet_const_host_buffer;

      cuda::copy_bytes(_stream, const_host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);

      cuda::copy_bytes(_stream, cuda::std::span(const_host_buffer), device_buffer);
      check_result_and_erase(_stream, device_buffer);
    }
  }

  SECTION("Asymmetric size")
  {
    test_buffer<int> host_buffer(test_buffer_type::pinned, 1);
    cuda::fill_bytes(_stream, host_buffer, fill_byte);

    ::std::vector<int> vec(buffer_size, 0xbeef);

    cuda::copy_bytes(_stream, host_buffer, vec);
    _stream.sync();

    CCCLRT_REQUIRE(vec[0] == get_expected_value(fill_byte));
    CCCLRT_REQUIRE(vec[1] == 0xbeef);
  }
}

template <typename SrcLayout = cuda::std::layout_right,
          typename DstLayout = SrcLayout,
          typename SrcExtents,
          typename DstExtents>
void test_mdspan_copy_bytes(
  cuda::stream_ref stream, SrcExtents src_extents = SrcExtents(), DstExtents dst_extents = DstExtents())
{
  auto src_buffer = make_buffer_for_mdspan<SrcLayout>(src_extents, 1);
  auto dst_buffer = make_buffer_for_mdspan<DstLayout>(dst_extents, 0);

  cuda::std::mdspan<int, SrcExtents, SrcLayout> src(src_buffer.data(), src_extents);
  cuda::std::mdspan<int, DstExtents, DstLayout> dst(dst_buffer.data(), dst_extents);

  for (int i = 0; i < static_cast<int>(src.extent(1)); i++)
  {
    src(0, i) = i;
  }

  cuda::copy_bytes(stream, std::move(src), dst);
  stream.sync();

  for (int i = 0; i < static_cast<int>(dst.extent(1)); i++)
  {
    CCCLRT_REQUIRE(dst(0, i) == i);
  }
}

C2H_CCCLRT_TEST("Mdspan copy", "[algorithm]")
{
  cuda::stream stream{cuda::device_ref{0}};

  SECTION("Different extents")
  {
    auto static_extents = cuda::std::extents<size_t, 3, 4>();
    test_mdspan_copy_bytes(stream, static_extents, static_extents);
    test_mdspan_copy_bytes<cuda::std::layout_left>(stream, static_extents, static_extents);

    auto dynamic_extents = cuda::std::dextents<size_t, 2>(3, 4);
    test_mdspan_copy_bytes(stream, dynamic_extents, dynamic_extents);
    test_mdspan_copy_bytes(stream, static_extents, dynamic_extents);
    test_mdspan_copy_bytes<cuda::std::layout_left>(stream, static_extents, dynamic_extents);

    auto mixed_extents = cuda::std::extents<int, cuda::std::dynamic_extent, 4>(3);
    test_mdspan_copy_bytes(stream, dynamic_extents, mixed_extents);
    test_mdspan_copy_bytes(stream, mixed_extents, static_extents);
    test_mdspan_copy_bytes<cuda::std::layout_left>(stream, mixed_extents, static_extents);
  }
}

C2H_CCCLRT_TEST("Non exhaustive mdspan copy_bytes", "[algorithm]")
{
  cuda::stream stream{cuda::device_ref{0}};
  {
    auto fake_strided_mdspan = create_fake_strided_mdspan();

    try
    {
      cuda::copy_bytes(stream, fake_strided_mdspan, fake_strided_mdspan);
    }
    catch (const ::std::invalid_argument& e)
    {
      CHECK(e.what() == ::std::string("copy_bytes supports only exhaustive mdspans"));
    }
  }
}
