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

#include <cuda/stream>

#include <cuda/experimental/__copy/mdspan_d2h_h2d.cuh>

#include "testing.cuh"

/***********************************************************************************************************************
 * Utils
 **********************************************************************************************************************/

template <typename Layout = cuda::std::layout_right, typename T, typename... IndexType>
auto to_mdspan_dyn(thrust::device_vector<T>& d_input, IndexType... indices)
{
  using extents = cuda::std::dims<sizeof...(IndexType)>;
  return cuda::device_mdspan<T, extents, Layout>(thrust::raw_pointer_cast(d_input.data()), extents(indices...));
}

template <typename Layout = cuda::std::layout_right, typename T, typename... IndexType>
auto to_mdspan_dyn(thrust::host_vector<T>& h_input, IndexType... indices)
{
  using extents = cuda::std::dims<sizeof...(IndexType)>;
  return cuda::host_mdspan<T, extents, Layout>(h_input.data(), extents(indices...));
}

template <typename Layout = cuda::std::layout_right, typename T, typename... IndexType>
auto to_const_mdspan(thrust::host_vector<T>& h_input, IndexType... indices)
{
  using extents = cuda::std::dims<sizeof...(IndexType)>;
  return cuda::host_mdspan<const T, extents, Layout>(h_input.data(), extents(indices...));
}

//----------------------------------------------------------------------------------------------------------------------

template <typename Layout = cuda::std::layout_right, typename T, typename I, size_t... Extents>
auto to_mdspan(thrust::host_vector<T>& h_input, cuda::std::extents<I, Extents...> extents)
{
  using extents_t = cuda::std::extents<I, Extents...>;
  using mapping_t = typename Layout::template mapping<extents_t>;
  return cuda::host_mdspan<T, extents_t, Layout>(h_input.data(), mapping_t(extents));
}

template <typename Layout = cuda::std::layout_right, typename T, typename I, size_t... Extents>
auto to_mdspan(thrust::device_vector<T>& d_input, cuda::std::extents<I, Extents...> extents)
{
  using extents_t = cuda::std::extents<I, Extents...>;
  using mapping_t = typename Layout::template mapping<extents_t>;
  return cuda::device_mdspan<T, extents_t, Layout>(thrust::raw_pointer_cast(d_input.data()), mapping_t(extents));
}

template <typename Layout = cuda::std::layout_right, typename I, typename T, auto... Extents>
auto to_const_mdspan(thrust::host_vector<T>& h_input, cuda::std::extents<I, Extents...> extents)
{
  using extents_t = cuda::std::extents<I, Extents...>;
  using mapping_t = typename Layout::template mapping<extents_t>;
  return cuda::host_mdspan<const T, extents_t, Layout>(h_input.data(), mapping_t(extents));
}

/***********************************************************************************************************************
 * 1D Tests
 **********************************************************************************************************************/

static const cuda::stream stream{cuda::device_ref{0}};

C2H_TEST("copy H2D 1D layout_right int", "[copy][h2d]")
{
  constexpr int N = 16;
  thrust::host_vector<int> host_data(N);
  thrust::device_vector<int> device_data(N, 0);
  for (int i = 0; i < N; ++i)
  {
    host_data[i] = i;
  }
  auto src = to_mdspan_dyn(host_data, N);
  auto dst = to_mdspan_dyn(device_data, N);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == host_data);
}

C2H_TEST("copy H2D 1D const source static extents", "[copy][h2d]")
{
  constexpr size_t N = 16;
  thrust::host_vector<int> host_data(N);
  thrust::device_vector<int> device_data(N, 0);
  for (size_t i = 0; i < N; ++i)
  {
    host_data[i] = static_cast<int>(i);
  }
  auto src = to_const_mdspan<cuda::std::layout_right>(host_data, cuda::std::extents<size_t, N>());
  auto dst = to_mdspan_dyn(device_data, N);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == host_data);
}

/***********************************************************************************************************************
 * 2D Tests
 **********************************************************************************************************************/

C2H_TEST("copy H2D 2D same layout", "[copy][h2d]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> host_data(M * N);
  thrust::device_vector<int> device_data(M * N, 0);
  for (int i = 0; i < M * N; ++i)
  {
    host_data[i] = i;
  }
  auto src = to_mdspan_dyn(host_data, M, N);
  auto dst = to_mdspan_dyn(device_data, M, N);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == host_data);

  auto src_col_major = to_mdspan_dyn<cuda::std::layout_left>(host_data, M, N);
  auto dst_col_major = to_mdspan_dyn<cuda::std::layout_left>(device_data, M, N);

  cuda::experimental::copy(src_col_major, dst_col_major, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == host_data);
}

C2H_TEST("copy H2D 2D different layout", "[copy][h2d]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> host_data(M * N);
  thrust::device_vector<int> device_data(M * N, 0);
  for (int i = 0; i < M * N; ++i)
  {
    host_data[i] = i;
  }
  auto src = to_mdspan_dyn(host_data, M, N);
  auto dst = to_mdspan_dyn<cuda::std::layout_left>(device_data, M, N);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
  // dst(i,j) = src(i,j) = i*N + j, stored at position j*M + i
  thrust::host_vector<int> expected(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[j * M + i] = i * N + j;
    }
  }
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == expected);
}

C2H_TEST("copy H2D 2D same layout swapped extents (common sublayout)", "[copy][h2d]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> host_data(M * N);
  thrust::device_vector<int> device_data(M * N, 0);
  for (int i = 0; i < M * N; ++i)
  {
    host_data[i] = i;
  }
  auto src = to_mdspan_dyn(host_data, M, N);
  auto dst = to_mdspan_dyn(device_data, N, M); // swapped

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == host_data);
}

C2H_TEST("copy H2D 2D mixed rank extents", "[copy][h2d]")
{
  constexpr size_t M = 4;
  constexpr int N    = 8;
  thrust::host_vector<int> host_data(M * N);
  thrust::device_vector<int> device_data(M * N, 0);
  for (int i = 0; i < static_cast<int>(M * N); ++i)
  {
    host_data[i] = i;
  }
  using src_extents = cuda::std::extents<size_t, M, cuda::std::dynamic_extent>;
  auto src          = to_mdspan(host_data, src_extents(M, N));
  auto dst          = to_mdspan_dyn(device_data, M * N);

  cuda::experimental::copy(src, dst, stream);
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == host_data);
}

/***********************************************************************************************************************
 * 3D Tests
 **********************************************************************************************************************/

C2H_TEST("copy H2D 3D layout_right", "[copy][h2d]")
{
  constexpr int D0    = 2;
  constexpr int D1    = 3;
  constexpr int D2    = 4;
  constexpr int total = D0 * D1 * D2;
  thrust::host_vector<int> host_data(total);
  thrust::device_vector<int> device_data(total, 0);
  for (int i = 0; i < total; ++i)
  {
    host_data[i] = i;
  }
  auto src = to_mdspan_dyn(host_data, D0, D1, D2);
  auto dst = to_mdspan_dyn<cuda::std::layout_left>(device_data, D0, D1, D2);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<int> expected(total);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        expected[i * D1 * D2 + j * D2 + k] = i * D1 * D2 + j * D2 + k;
      }
    }
  }
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == expected);
}

C2H_TEST("copy H2D 2D large", "[copy][h2d]")
{
  constexpr int M = 1280;
  constexpr int N = 2564;
  thrust::host_vector<int> host_data(M * N);
  thrust::device_vector<int> device_data(M * N, 0);
  for (int i = 0; i < M * N; ++i)
  {
    host_data[i] = i;
  }
  auto src = to_mdspan_dyn(host_data, M, N);
  auto dst = to_mdspan_dyn(device_data, M, N);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<int>(device_data) == host_data);
}

/***********************************************************************************************************************
 * Edge Cases
 **********************************************************************************************************************/

C2H_TEST("copy H2D rank 0", "[copy][h2d]")
{
  thrust::host_vector<int> host_data(1, 42);
  thrust::device_vector<int> device_data(1, 0);
  auto src = to_mdspan_dyn(host_data);
  auto dst = to_mdspan_dyn(device_data);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
  CUDAX_REQUIRE(device_data[0] == 42);
}

C2H_TEST("copy H2D size 0", "[copy][h2d]")
{
  int value = 42;
  cuda::host_mdspan<int, cuda::std::dims<0>> src(&value);
  cuda::device_mdspan<int, cuda::std::dims<0>> dst(&value);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();
}

C2H_TEST("copy H2D size mismatch throws", "[copy][h2d]")
{
  constexpr int N = 16;
  thrust::host_vector<int> host_data(N, 0);
  thrust::device_vector<int> device_data(N / 2, 0);
  auto src = to_mdspan_dyn(host_data, N);
  auto dst = to_mdspan_dyn(device_data, N / 2);

  REQUIRE_THROWS_AS(cuda::experimental::copy(src, dst, stream), std::invalid_argument);
}
