//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>

#include "common.cuh"
#include "cuda/__algorithm/copy.h"

C2H_CCCLRT_TEST("1d Copy", "[algorithm]")
{
  cuda::stream _stream{cuda::device_ref{0}};

  SECTION("Device resource")
  {
    std::vector<int> host_vector(buffer_size);

    {
      auto buffer = cuda::make_device_buffer<int>(_stream, cuda::device_ref{0}, buffer_size, cuda::no_init);
      cuda::fill_bytes(_stream, buffer, fill_byte);

      cuda::copy_bytes(_stream, buffer, host_vector);
      check_result_and_erase(_stream, host_vector);

      cuda::copy_bytes(_stream, buffer, host_vector);
      check_result_and_erase(_stream, host_vector);
    }
    {
      auto not_yet_const_buffer =
        cuda::make_device_buffer<int>(_stream, cuda::device_ref{0}, buffer_size, cuda::no_init);
      cuda::fill_bytes(_stream, not_yet_const_buffer, fill_byte);

      const auto& const_buffer = not_yet_const_buffer;

      cuda::copy_bytes(_stream, const_buffer, host_vector);
      check_result_and_erase(_stream, host_vector);

      cuda::copy_bytes(_stream, const_buffer, cuda::std::span(host_vector));
      check_result_and_erase(_stream, host_vector);

      cuda::copy_configuration config;
      config.src_location_hint = cuda::device_ref{0};
#if _CCCL_CTK_AT_LEAST(13, 0)
      config.src_access_order = cuda::source_access_order::stream;
#else
      config.src_access_order = cuda::source_access_order::any;
#endif
      cuda::copy_bytes(_stream, const_buffer, host_vector, config);
      check_result_and_erase(_stream, host_vector);

      cuda::copy_bytes(_stream, const_buffer.first(0), host_vector);
    }
  }

  SECTION("Host and managed resource")
  {
    {
      auto host_buffer   = make_pinned_memory_buffer<int>(_stream, buffer_size);
      auto device_buffer = make_managed_memory_buffer<int>(_stream, buffer_size);

      cuda::fill_bytes(_stream, host_buffer, fill_byte);

      cuda::copy_bytes(_stream, host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);

      cuda::copy_bytes(_stream, host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);
    }

    {
      auto not_yet_const_host_buffer = make_pinned_memory_buffer<int>(_stream, buffer_size);
      auto device_buffer             = make_managed_memory_buffer<int>(_stream, buffer_size);
      cuda::fill_bytes(_stream, not_yet_const_host_buffer, fill_byte);

      const auto& const_host_buffer = not_yet_const_host_buffer;

      cuda::copy_bytes(_stream, const_host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);

      cuda::copy_bytes(_stream, const_host_buffer, device_buffer);
      check_result_and_erase(_stream, device_buffer);
    }
  }

  SECTION("Asymmetric size")
  {
    auto host_buffer = make_pinned_memory_buffer<int>(_stream, 1);
    cuda::fill_bytes(_stream, host_buffer, fill_byte);

    ::std::vector<int> vec(buffer_size, 0xbeef);

    cuda::copy_bytes(_stream, host_buffer, vec);
    _stream.sync();

    CCCLRT_REQUIRE(vec[0] == get_expected_value(fill_byte));
    CCCLRT_REQUIRE(vec[1] == 0xbeef);
  }
}

C2H_CCCLRT_TEST("copy_bytes uses the stream device when current device differs", "[algorithm][multi_gpu]")
{
  if (cuda::devices.size() < 2)
  {
    return;
  }

#if _CCCL_CTK_AT_LEAST(13, 0)
  cuda::device_ref current_device{0};
  cuda::device_ref explicit_device{1};
  if (!explicit_device.attribute(cuda::device_attributes::memory_pools_supported))
  {
    return;
  }

  cuda::stream stream{explicit_device};

  int expected = get_expected_value(fill_byte);
  int result{};

  {
    auto host_src = make_pinned_memory_buffer<int>(stream, 1, explicit_device);
    auto host_dst = make_pinned_memory_buffer<int>(stream, 1, explicit_device);
    auto src      = cuda::make_device_buffer<int>(stream, explicit_device, 1, cuda::no_init);
    auto dst      = cuda::make_device_buffer<int>(stream, explicit_device, 1, cuda::no_init);

    host_src.data()[0] = expected;
    host_dst.data()[0] = 0;

    {
      cuda::__ensure_current_context guard(explicit_device);
      cuda::copy_bytes(stream, host_src, src);
    }

    {
      cuda::__ensure_current_context guard(current_device);
      cuda::copy_bytes(stream, src, dst);
    }

    {
      cuda::__ensure_current_context guard(explicit_device);
      cuda::copy_bytes(stream, dst, host_dst);
    }

    stream.sync();
    result = host_dst.data()[0];
  }
  stream.sync();

  CCCLRT_REQUIRE(result == expected);
#endif // _CCCL_CTK_AT_LEAST(13, 0)
}

template <typename SrcLayout = cuda::std::layout_right,
          typename DstLayout = SrcLayout,
          typename SrcExtents,
          typename DstExtents>
void test_mdspan_copy_bytes(
  cuda::stream_ref stream, SrcExtents src_extents = SrcExtents(), DstExtents dst_extents = DstExtents())
{
  auto src_buffer = make_buffer_for_mdspan<SrcLayout>(stream, src_extents, 1);
  auto tmp_buffer = make_buffer_for_mdspan<SrcLayout>(stream, src_extents, 0);
  auto dst_buffer = make_buffer_for_mdspan<DstLayout>(stream, dst_extents, 0);

  cuda::std::mdspan<int, SrcExtents, SrcLayout> src(src_buffer.data(), src_extents);
  cuda::std::mdspan<int, SrcExtents, SrcLayout> tmp(tmp_buffer.data(), src_extents);
  cuda::std::mdspan<int, DstExtents, DstLayout> dst(dst_buffer.data(), dst_extents);

  for (int i = 0; i < static_cast<int>(src.extent(1)); i++)
  {
    src(0, i) = i;
  }

  cuda::copy_bytes(stream, std::move(src), tmp);

  cuda::copy_configuration config;
#if _CCCL_CTK_AT_LEAST(13, 0)
  config.src_access_order = cuda::source_access_order::stream;
#else
  config.src_access_order = cuda::source_access_order::any;
#endif
  cuda::copy_bytes(stream, tmp, dst, config);
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
