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

C2H_TEST("1d Copy", "[data_manipulation]")
{
  cudax::stream _stream{cuda::device_ref{0}};

  SECTION("Device resource")
  {
    cudax::device_memory_resource device_resource{cuda::device_ref{0}};
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
    cudax::legacy_managed_memory_resource managed_resource;
    cudax::legacy_pinned_memory_resource host_resource;

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
    cudax::legacy_pinned_memory_resource host_resource;
    cudax::weird_buffer input(host_resource, buffer_size);
    cudax::weird_buffer output(host_resource, buffer_size);

    memset(input.data, fill_byte, input.size * sizeof(int));

    cudax::copy_bytes(_stream, input, output);
    check_result_and_erase(_stream, cuda::std::span(output.data, output.size));
  }

  SECTION("Asymmetric size")
  {
    cudax::legacy_pinned_memory_resource host_resource;
    cudax::uninitialized_buffer<int, cuda::mr::host_accessible> host_buffer(host_resource, 1);
    cudax::fill_bytes(_stream, host_buffer, fill_byte);

    ::std::vector<int> vec(buffer_size, 0xbeef);

    cudax::copy_bytes(_stream, host_buffer, vec);
    _stream.sync();

    CUDAX_REQUIRE(vec[0] == get_expected_value(fill_byte));
    CUDAX_REQUIRE(vec[1] == 0xbeef);
  }
}

template <typename SrcLayout = cuda::std::layout_right,
          typename DstLayout = SrcLayout,
          typename SrcExtents,
          typename DstExtents>
void test_mdspan_copy_bytes(
  cudax::stream_ref stream, SrcExtents src_extents = SrcExtents(), DstExtents dst_extents = DstExtents())
{
  auto src_buffer = make_buffer_for_mdspan<SrcLayout>(src_extents, 1);
  auto dst_buffer = make_buffer_for_mdspan<DstLayout>(dst_extents, 0);

  cuda::std::mdspan<int, SrcExtents, SrcLayout> src(src_buffer.data(), src_extents);
  cuda::std::mdspan<int, DstExtents, DstLayout> dst(dst_buffer.data(), dst_extents);

  for (int i = 0; i < static_cast<int>(src.extent(1)); i++)
  {
    src(0, i) = i;
  }

  cudax::copy_bytes(stream, std::move(src), dst);
  stream.sync();

  for (int i = 0; i < static_cast<int>(dst.extent(1)); i++)
  {
    CUDAX_CHECK(dst(0, i) == i);
  }
}

C2H_TEST("Mdspan copy", "[data_manipulation]")
{
  cudax::stream stream{cuda::device_ref{0}};

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

  SECTION("Launch transform")
  {
    auto host_resource = cudax::legacy_pinned_memory_resource{};
    auto mixed_extents =
      cuda::std::extents<size_t, 1024, cuda::std::dynamic_extent, 2, cuda::std::dynamic_extent>(1024, 2);
    [[maybe_unused]] auto static_extents = cuda::std::extents<size_t, 1024, 1024, 2, 2>();
    auto mdspan_buffer                   = make_buffer_for_mdspan(mixed_extents, 1);
    cuda::std::mdspan<int, decltype(mixed_extents)> mdspan(mdspan_buffer.data(), mixed_extents);
    cudax::weird_buffer<cuda::std::mdspan<int, decltype(static_extents)>> buffer{
      host_resource, mdspan.mapping().required_span_size()};

    cudax::copy_bytes(stream, mdspan, buffer);
    stream.sync();
    CUDAX_REQUIRE(!memcmp(mdspan_buffer.data(), buffer.data, mdspan_buffer.size()));
  }
}

C2H_TEST("Non exhaustive mdspan copy_bytes", "[data_manipulation]")
{
  cudax::stream stream{cuda::device_ref{0}};
  {
    auto fake_strided_mdspan = create_fake_strided_mdspan();

    try
    {
      cudax::copy_bytes(stream, fake_strided_mdspan, fake_strided_mdspan);
    }
    catch (const ::std::invalid_argument& e)
    {
      CHECK(e.what() == ::std::string("copy_bytes supports only exhaustive mdspans"));
    }
  }
}
