//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda/experimental/fill_bytes.cuh>

#include <cstring>
#include <stdexcept>

#include "testing.cuh"

static const cuda::stream stream{cuda::device_ref{0}};

using host_vector_bytes_t = thrust::host_vector<cuda::std::byte>;
using span_bytes_t        = cuda::std::span<const cuda::std::byte>;

template <typename Value>
void fill_expected_bytes(host_vector_bytes_t& expected, size_t byte_offset, size_t num_bytes, const Value& value)
{
  cuda::std::byte pattern[sizeof(Value)];
  std::memcpy(pattern, &value, sizeof(Value));

  for (size_t i = 0; i < num_bytes; ++i)
  {
    expected[byte_offset + i] = pattern[i % sizeof(Value)];
  }
}

// create a host vector of bytes with a repeated pattern
template <typename Value>
host_vector_bytes_t repeated_bytes_vector(size_t num_bytes, const Value& value)
{
  host_vector_bytes_t expected(num_bytes);
  fill_expected_bytes(expected, 0, num_bytes, value);
  return expected;
}

template <typename Tp>
span_bytes_t to_byte_span(const thrust::host_vector<Tp>& data)
{
  return cuda::std::as_bytes(cuda::std::span<const Tp>{data.data(), data.size()});
}

// create a copy of the input as a host vector of bytes
template <typename Tp>
host_vector_bytes_t to_byte_vector(const thrust::host_vector<Tp>& data)
{
  auto data_bytes = to_byte_span(data);
  host_vector_bytes_t bytes(data_bytes.size());
  for (size_t i = 0; i < data_bytes.size(); ++i)
  {
    bytes[i] = data_bytes[i];
  }
  return bytes;
}

bool equal_bytes(span_bytes_t lhs, span_bytes_t rhs)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i)
  {
    if (lhs[i] != rhs[i])
    {
      return false;
    }
  }
  return true;
}

template <typename Tp>
thrust::host_vector<Tp> make_host_data(size_t size, uint8_t byte)
{
  thrust::host_vector<Tp> data(size);
  ::std::memset(data.data(), byte, data.size() * sizeof(Tp));
  return data;
}

template <typename Tp, typename Value>
host_vector_bytes_t contiguous_fill_bytes(size_t num_elements, Value value)
{
  return repeated_bytes_vector(num_elements * sizeof(Tp), value);
}

template <typename Tp, typename Value>
void fill_expected_element(host_vector_bytes_t& expected, size_t element_offset, Value value)
{
  fill_expected_bytes(expected, element_offset * sizeof(Tp), sizeof(Tp), value);
}

// contiguous layout tests
template <typename _Layout = cuda::std::layout_right, typename Tp, typename Value, typename Index, size_t... Extents>
void test_impl(const thrust::host_vector<Tp>& input,
               const host_vector_bytes_t& expected,
               cuda::std::extents<Index, Extents...> extents,
               Value value)
{
  using extents_t = cuda::std::extents<Index, Extents...>;
  using mapping_t = typename _Layout::template mapping<extents_t>;
  thrust::device_vector<Tp> device_data(input.begin(), input.end());
  cuda::device_mdspan<Tp, extents_t, _Layout> dst(thrust::raw_pointer_cast(device_data.data()), mapping_t(extents));

  cuda::experimental::fill_bytes(dst, value, stream);
  stream.sync();

  const thrust::host_vector<Tp> actual(device_data);
  REQUIRE(equal_bytes(to_byte_span(actual), to_byte_span(expected)));
}

// layout_stride tests
template <typename Tp, typename Value, typename Index, size_t... Extents>
void test_impl_stride(
  const thrust::host_vector<Tp>& input,
  const host_vector_bytes_t& expected,
  const cuda::std::extents<Index, Extents...>& extents,
  const cuda::std::array<Index, sizeof...(Extents)>& strides,
  Value value,
  size_t offset = 0)
{
  using extents_t = cuda::std::extents<Index, Extents...>;
  using mapping_t = cuda::std::layout_stride::mapping<extents_t>;

  thrust::device_vector<Tp> device_data(input.begin(), input.end());
  cuda::device_mdspan<Tp, extents_t, cuda::std::layout_stride> dst(
    thrust::raw_pointer_cast(device_data.data()) + offset, mapping_t(extents, strides));

  cuda::experimental::fill_bytes(dst, value, stream);
  stream.sync();

  const thrust::host_vector<Tp> actual(device_data);
  REQUIRE(equal_bytes(to_byte_span(actual), to_byte_span(expected)));
}

// layout_stride_relaxed tests
template <typename Tp, typename Value, typename Index, size_t... Extents>
void test_impl_relaxed(
  const thrust::host_vector<Tp>& input,
  const host_vector_bytes_t& expected,
  const cuda::std::extents<Index, Extents...>& extents,
  const cuda::dstrides<Index, sizeof...(Extents)>& strides,
  Index offset,
  Value value)
{
  using extents_t = cuda::std::extents<Index, Extents...>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;

  thrust::device_vector<Tp> device_data(input.begin(), input.end());
  cuda::device_mdspan<Tp, extents_t, cuda::layout_stride_relaxed> dst(
    thrust::raw_pointer_cast(device_data.data()), mapping_t(extents, strides, offset));

  cuda::experimental::fill_bytes(dst, value, stream);
  stream.sync();

  const thrust::host_vector<Tp> actual(device_data);
  REQUIRE(equal_bytes(to_byte_span(actual), to_byte_span(expected)));
}

/***********************************************************************************************************************
 * 1D Tests
 **********************************************************************************************************************/

enum class pattern32 : uint32_t
{
  value = 0xff00ff00u,
};

TEST_CASE("fill_bytes device mdspan accepts generic fill values", "[fill_bytes][device]")
{
  constexpr int n = 16427;
  using extents_t = cuda::std::extents<int, n>;
  // int16_t
  {
    auto input              = make_host_data<int16_t>(n, 0xCD);
    constexpr int16_t value = -0x1234;
    test_impl(input, contiguous_fill_bytes<int16_t>(n, value), extents_t{}, value);
  }
  // std::byte
  {
    auto input           = make_host_data<uint8_t>(n, 0xCD);
    constexpr auto value = cuda::std::byte{0xAB};
    test_impl(input, contiguous_fill_bytes<uint8_t>(n, value), extents_t{}, value);
  }
  // int
  {
    auto input           = make_host_data<int>(n, 0xCD);
    constexpr auto value = pattern32::value;
    test_impl(input, contiguous_fill_bytes<int>(n, value), extents_t{}, value);
  }
  // float
  {
    auto input          = make_host_data<float>(n, 0xCD);
    constexpr int value = 0xAB;
    test_impl(input, contiguous_fill_bytes<float>(n, value), extents_t{}, value);
  }
}

TEST_CASE("fill_bytes handles layout_stride_relaxed negative strides and offsets", "[fill_bytes][relaxed]")
{
  constexpr int n = 8455;
  using extents_t = cuda::std::extents<int, n>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;

  // int8_t
  {
    auto input = make_host_data<uint8_t>(n, 0xCD);
    const mapping_t mapping(extents_t{}, cuda::dstrides<int, 1>(-1), n - 1);
    const uint8_t value = 0x5A;
    test_impl_relaxed(input, contiguous_fill_bytes<uint8_t>(n, value), extents_t{}, mapping.strides(), n - 1, value);
  }
  // uint32_t pattern into int64_t elements
  {
    auto input = make_host_data<int64_t>(n, 0xCD);
    const mapping_t mapping(extents_t{}, cuda::dstrides<int, 1>(-1), n - 1);
    const uint32_t value = 0x12345678;
    test_impl_relaxed(input, contiguous_fill_bytes<int64_t>(n, value), extents_t{}, mapping.strides(), n - 1, value);
  }
}

/***********************************************************************************************************************
 * 2D Tests
 **********************************************************************************************************************/

TEST_CASE("fill_bytes handles layout_left device mdspan", "[fill_bytes][layout_left]")
{
  constexpr int rows = 265;
  constexpr int cols = 456;
  using extents_t    = cuda::std::extents<int, rows, cols>;

  auto input           = make_host_data<uint16_t>(rows * cols, 0xCD);
  const uint16_t value = 0x1234;
  test_impl<cuda::std::layout_left>(input, contiguous_fill_bytes<uint16_t>(rows * cols, value), extents_t{}, value);
}

TEST_CASE("fill_bytes handles singleton dimensions", "[fill_bytes][singleton]")
{
  constexpr int rows = 2;
  constexpr int cols = 4;
  using extents_t    = cuda::std::extents<int, rows, 1, cols>;

  auto input           = make_host_data<uint16_t>(rows * cols, 0xCD);
  constexpr auto value = uint16_t{0x1234};
  test_impl(input, contiguous_fill_bytes<uint16_t>(rows * cols, value), extents_t{}, value);
  test_impl<cuda::std::layout_left>(input, contiguous_fill_bytes<uint16_t>(rows * cols, value), extents_t{}, value);
}

TEST_CASE("fill_bytes preserves padding bytes in strided row-major destination layouts", "[fill_bytes][stride]")
{
  constexpr int rows = 367;
  constexpr int cols = 456;
  constexpr int ld   = 657;
  using extents_t    = cuda::std::extents<int, rows, cols>;
  using mapping_t    = cuda::std::layout_stride::mapping<extents_t>;

  cuda::std::array<int, 2> strides{ld, 1};
  mapping_t mapping(extents_t{}, strides);
  auto input     = make_host_data<uint16_t>(mapping.required_span_size(), 0xCD);
  uint16_t value = 0x1234;

  auto expected = to_byte_vector(input);
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      fill_expected_element<uint16_t>(expected, row * ld + col, value);
    }
  }
  test_impl_stride(input, expected, extents_t{}, strides, value);
}

TEST_CASE("fill_bytes preserves padding bytes in strided column-major destination layouts", "[fill_bytes][stride]")
{
  constexpr int rows = 127;
  constexpr int cols = 79;
  constexpr int ld   = 191;
  using extents_t    = cuda::std::extents<int, rows, cols>;
  using mapping_t    = cuda::std::layout_stride::mapping<extents_t>;

  cuda::std::array<int, 2> strides{1, ld};
  mapping_t mapping(extents_t{}, strides);
  auto input     = make_host_data<uint16_t>(mapping.required_span_size(), 0xCD);
  uint16_t value = 0x1234;

  auto expected = to_byte_vector(input);
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      fill_expected_element<uint16_t>(expected, row + col * ld, value);
    }
  }
  test_impl_stride(input, expected, extents_t{}, strides, value);
}

/***********************************************************************************************************************
 * 3D Tests
 **********************************************************************************************************************/

TEST_CASE("fill_bytes handles 3D mdspans", "[fill_bytes][3d]")
{
  constexpr int dim0  = 2;
  constexpr int dim1  = 3;
  constexpr int dim2  = 4;
  constexpr int total = dim0 * dim1 * dim2;
  using extents_t     = cuda::std::extents<int, dim0, dim1, dim2>;

  auto input           = make_host_data<uint32_t>(total, 0xCD);
  constexpr auto value = pattern32::value;
  test_impl(input, contiguous_fill_bytes<uint32_t>(total, value), extents_t{}, value);
  test_impl<cuda::std::layout_left>(input, contiguous_fill_bytes<uint32_t>(total, value), extents_t{}, value);
}

TEST_CASE("fill_bytes handles 3D strided permutation layouts", "[fill_bytes][3d][stride][permutation]")
{
  constexpr int dim0 = 25;
  constexpr int dim1 = 37;
  constexpr int dim2 = 41;
  using extents_t    = cuda::std::extents<int, dim0, dim1, dim2>;
  constexpr int span = (dim0 - 1) + (dim1 - 1) * dim2 * dim0 + (dim2 - 1) * dim0 + 1;

  cuda::std::array<int, 3> strides{1, dim2 * dim0, dim0};
  auto input           = make_host_data<uint32_t>(span, 0xCD);
  constexpr auto value = pattern32::value;

  auto expected = to_byte_vector(input);
  for (int i = 0; i < dim0; ++i)
  {
    for (int j = 0; j < dim1; ++j)
    {
      for (int k = 0; k < dim2; ++k)
      {
        int offset = i + j * dim2 * dim0 + k * dim0;
        fill_expected_element<uint32_t>(expected, offset, value);
      }
    }
  }
  test_impl_stride(input, expected, extents_t{}, strides, value);
}

TEST_CASE("fill_bytes handles 3D strided tile_size greater than one", "[fill_bytes][3d][stride][tile]")
{
  constexpr int dim0      = 27;
  constexpr int dim1      = 39;
  constexpr int dim2      = 47;
  using extents_t         = cuda::std::extents<int, dim0, dim1, dim2>;
  constexpr int stride0   = 64;
  constexpr int stride1   = 2048;
  constexpr int span_size = (dim0 - 1) * stride0 + (dim1 - 1) * stride1 + dim2;

  cuda::std::array<int, 3> strides{stride0, stride1, 1};
  auto input           = make_host_data<uint32_t>(span_size, 0xCD);
  constexpr auto value = pattern32::value;

  auto expected = to_byte_vector(input);
  for (int i = 0; i < dim0; ++i)
  {
    for (int j = 0; j < dim1; ++j)
    {
      for (int k = 0; k < dim2; ++k)
      {
        int offset = i * stride0 + j * stride1 + k;
        fill_expected_element<uint32_t>(expected, offset, value);
      }
    }
  }
  test_impl_stride(input, expected, extents_t{}, strides, value);
}

TEST_CASE("fill_bytes preserves surrounding bytes in strided subviews with offsets", "[fill_bytes][stride][offset]")
{
  constexpr int rows   = 2;
  constexpr int cols   = 3;
  constexpr int ld     = 4;
  constexpr int offset = 3;
  constexpr int alloc  = 16;
  using extents_t      = cuda::std::extents<int, rows, cols>;

  cuda::std::array<int, 2> strides{1, ld};
  auto input           = make_host_data<uint32_t>(alloc, 0xCD);
  constexpr auto value = uint32_t{0x12345678};

  auto expected = to_byte_vector(input);
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      fill_expected_element<uint32_t>(expected, offset + row + col * ld, value);
    }
  }
  test_impl_stride(input, expected, extents_t{}, strides, value, offset);
}

/***********************************************************************************************************************
 * Edge Cases
 **********************************************************************************************************************/

TEST_CASE("fill_bytes handles rank-zero and zero-size mdspans", "[fill_bytes][edge]")
{
  // rank-zero
  {
    using extents_t = cuda::std::extents<int>;
    auto input      = make_host_data<uint32_t>(1, 0xCD);
    auto value      = pattern32::value;
    test_impl(input, contiguous_fill_bytes<uint32_t>(1, value), extents_t{}, value);
  }
  // zero-size
  {
    auto input = make_host_data<uint32_t>(1, 0xCD);
    test_impl(input, to_byte_vector(input), cuda::std::dims<1>{0}, uint32_t{0xdeadbeef});
  }
}

TEST_CASE("fill_bytes rejects interleaved layout", "[fill_bytes][throw]")
{
  using extents_t = cuda::std::extents<int, 2, 2>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
  auto input      = make_host_data<uint32_t>(6, 0xCD);

  thrust::device_vector<uint32_t> device_data(input.begin(), input.end());
  cuda::device_mdspan<uint32_t, extents_t, cuda::layout_stride_relaxed> dst(
    thrust::raw_pointer_cast(device_data.data()), mapping_t(extents_t{}, cuda::dstrides<int, 2>(2, 3)));

  REQUIRE_THROWS_AS(cuda::experimental::fill_bytes(dst, uint32_t{0x12345678}, stream), std::invalid_argument);
}
