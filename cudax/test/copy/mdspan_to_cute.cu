//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <cuda/experimental/__copy/mdspan_to_cute.cuh>
#include <cuda/experimental/__copy/mdspan_d2h_h2d.cuh>


#include "testing.cuh"
#include <cute/tensor.hpp>

C2H_TEST("to_cute with 0D mdspan (scalar)", "[mdspan][cute]")
{
  float value = 42.0f;
  cuda::std::mdspan<float, cuda::std::extents<size_t>> src(&value);

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == &value);
  CUDAX_REQUIRE(cute::size(tensor) == 1);
  CUDAX_REQUIRE(*tensor.data() == 42.0f);
}

C2H_TEST("to_cute with 1D mdspan layout_right", "[mdspan][cute]")
{
  constexpr int N = 16;
  cuda::std::array<float, N> data{};
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::mdspan<float, cuda::std::dims<1>> src(data.data(), N);

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == data.data());
  CUDAX_REQUIRE(cute::size<0>(tensor) == N);
  CUDAX_REQUIRE(cute::size(tensor) == N);
  CUDAX_REQUIRE(cute::stride<0>(tensor) == 1);
  for (int i = 0; i < N; ++i)
  {
    CUDAX_REQUIRE(tensor(i) == src(i));
  }
}

C2H_TEST("to_cute with 2D mdspan layout_right (row-major)", "[mdspan][cute]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  cuda::std::array<float, M * N> data{};
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::mdspan<float, cuda::std::dims<2>> src(data.data(), M, N);

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == data.data());
  CUDAX_REQUIRE(cute::size<0>(tensor) == M);
  CUDAX_REQUIRE(cute::size<1>(tensor) == N);
  CUDAX_REQUIRE(cute::size(tensor) == M * N);
  CUDAX_REQUIRE(cute::stride<0>(tensor) == N);
  CUDAX_REQUIRE(cute::stride<1>(tensor) == 1);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      CUDAX_REQUIRE(tensor(i, j) == src(i, j));
    }
  }
}

C2H_TEST("to_cute with 2D mdspan layout_left (column-major)", "[mdspan][cute]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  cuda::std::array<float, M * N> data{};
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::mdspan<float, cuda::std::dims<2>, cuda::std::layout_left> src(data.data(), M, N);

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == data.data());
  CUDAX_REQUIRE(cute::size<0>(tensor) == M);
  CUDAX_REQUIRE(cute::size<1>(tensor) == N);
  CUDAX_REQUIRE(cute::size(tensor) == M * N);
  CUDAX_REQUIRE(cute::stride<0>(tensor) == 1);
  CUDAX_REQUIRE(cute::stride<1>(tensor) == M);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      CUDAX_REQUIRE(tensor(i, j) == src(i, j));
    }
  }
}

C2H_TEST("to_cute with 2D mdspan layout_stride", "[mdspan][cute]")
{
  constexpr int M         = 4;
  constexpr int N         = 8;
  constexpr int stride0   = 16; // Custom stride for first dimension
  constexpr int stride1   = 2; // Custom stride for second dimension
  constexpr int data_size = stride0 * (M - 1) + stride1 * (N - 1) + 1;
  cuda::std::array<float, data_size> data{};
  for (int i = 0; i < data_size; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::array<size_t, 2> strides{stride0, stride1};
  cuda::std::layout_stride::mapping<cuda::std::dims<2>> mapping(cuda::std::dims<2>(M, N), strides);
  cuda::std::mdspan<float, cuda::std::dims<2>, cuda::std::layout_stride> src(data.data(), mapping);

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == data.data());
  CUDAX_REQUIRE(cute::size<0>(tensor) == M);
  CUDAX_REQUIRE(cute::size<1>(tensor) == N);
  CUDAX_REQUIRE(cute::stride<0>(tensor) == stride0);
  CUDAX_REQUIRE(cute::stride<1>(tensor) == stride1);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      CUDAX_REQUIRE(tensor(i, j) == src(i, j));
    }
  }
}

C2H_TEST("to_cute with const element type", "[mdspan][cute]")
{
  constexpr int N = 16;
  cuda::std::array<float, N> data{};
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::mdspan<const float, cuda::std::dims<1>, cuda::std::layout_right> src(data.data(), N);
  auto tensor = cuda::experimental::to_cute(src);
  static_assert(cuda::std::is_same_v<decltype(tensor.data()), const float*&>);
}

C2H_TEST("to_cute with different index types", "[mdspan][cute]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  cuda::std::array<double, M * N> data{};
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = static_cast<double>(i);
  }
  cuda::std::mdspan<double, cuda::std::dims<2, int>> src(data.data(), M, N);
  auto tensor = cuda::experimental::to_cute(src);
  static_assert(cuda::std::is_same_v<decltype(cute::size<0>(tensor)), int>);
  CUDAX_REQUIRE(tensor.data() == data.data());
}

C2H_TEST("to_cute with static extents layout_right", "[mdspan][cute]")
{
  constexpr size_t M = 4;
  constexpr size_t N = 8;
  cuda::std::array<float, M * N> data{};
  for (size_t i = 0; i < M * N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::mdspan<float, cuda::std::extents<size_t, M, N>> src(data.data());

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == data.data());
  static_assert(cuda::std::is_same_v<decltype(cute::size<0>(tensor)), cute::C<M>>);
  static_assert(cuda::std::is_same_v<decltype(cute::size<1>(tensor)), cute::C<N>>);
  static_assert(cuda::std::is_same_v<decltype(cute::stride<0>(tensor)), cute::C<N>>);
  static_assert(cuda::std::is_same_v<decltype(cute::stride<1>(tensor)), cute::C<size_t{1}>>);
  static_assert(cute::size<0>(tensor) == M);
  static_assert(cute::size<1>(tensor) == N);
  static_assert(cute::stride<0>(tensor) == N);
  static_assert(cute::stride<1>(tensor) == 1);
}

C2H_TEST("to_cute with static extents layout_left", "[mdspan][cute]")
{
  constexpr size_t M = 4;
  constexpr size_t N = 8;
  cuda::std::array<float, M * N> data{};
  for (size_t i = 0; i < M * N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::mdspan<float, cuda::std::extents<size_t, M, N>, cuda::std::layout_left> src(data.data());

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == data.data());
  static_assert(cuda::std::is_same_v<decltype(cute::size<0>(tensor)), cute::C<M>>);
  static_assert(cuda::std::is_same_v<decltype(cute::size<1>(tensor)), cute::C<N>>);
  static_assert(cuda::std::is_same_v<decltype(cute::stride<0>(tensor)), cute::C<size_t{1}>>);
  static_assert(cuda::std::is_same_v<decltype(cute::stride<1>(tensor)), cute::C<M>>);
  static_assert(cute::size<0>(tensor) == M);
  static_assert(cute::size<1>(tensor) == N);
  static_assert(cute::stride<0>(tensor) == 1);
  static_assert(cute::stride<1>(tensor) == M);
}

C2H_TEST("to_cute with mixed static and dynamic extents", "[mdspan][cute]")
{
  constexpr size_t M = 4;
  constexpr int N    = 8;
  cuda::std::array<float, M * N> data{};
  for (size_t i = 0; i < M * N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  cuda::std::mdspan<float, cuda::std::extents<size_t, M, cuda::std::dynamic_extent>> src(data.data(), N);

  auto tensor = cuda::experimental::to_cute(src);
  CUDAX_REQUIRE(tensor.data() == data.data());
  static_assert(cuda::std::is_same_v<decltype(cute::size<0>(tensor)), cute::C<M>>);
  static_assert(cuda::std::is_same_v<decltype(cute::size<1>(tensor)), size_t>);
  static_assert(cuda::std::is_same_v<decltype(cute::stride<0>(tensor)), size_t>);
  static_assert(cuda::std::is_same_v<decltype(cute::stride<1>(tensor)), size_t>);
  static_assert(cute::size<0>(tensor) == M);
  CUDAX_REQUIRE(cute::size<1>(tensor) == N);
  CUDAX_REQUIRE(cute::stride<0>(tensor) == N);
  CUDAX_REQUIRE(cute::stride<1>(tensor) == 1);
}
