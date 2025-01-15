//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/mdspan>

#include <cuda/experimental/accessor.cuh>

#include <tuple>

#include "cuda/__barrier/aligned_size.h"
#include <testing.cuh>

template <typename T, typename E, typename L, typename A>
__global__ void mdspan_accessor_kernel(cuda::std::mdspan<T, E, L, A> md)
{
  CUDAX_REQUIRE(md(threadIdx.x) == threadIdx.x);
}

using namespace cuda::experimental;

using TypeLists = std::tuple<std::tuple<eviction_none_t, cuda::aligned_size_t<4>, no_prefetch_t, ptr_no_aliasing_t>,
                             std::tuple<eviction_normal_t, cuda::aligned_size_t<8>, prefetch_64B_t, ptr_no_aliasing_t>>;

TEMPLATE_LIST_TEST_CASE("Accessor", "[device]", TypeLists)
{
  using EvictionPolicy = std::tuple_element_t<0, TestType>;
  using Alignment      = std::tuple_element_t<1, TestType>;
  using Prefetch       = std::tuple_element_t<2, TestType>;
  using Restrict       = std::tuple_element_t<3, TestType>;

  thrust::host_vector<int> h_vector(32);
  std::iota(h_vector.begin(), h_vector.end(), 0);
  thrust::device_vector<int> d_vector = h_vector;

  auto md            = cuda::std::mdspan(thrust::raw_pointer_cast(d_vector.data()), d_vector.size());
  auto md_with_props = cuda::experimental::add_properties(md, cuda::experimental::eviction_no_alloc);
  mdspan_accessor_kernel<<<1, 32>>>(md_with_props);
  cudaDeviceSynchronize();
  CUDAX_REQUIRE(cudaGetLastError() == cudaSuccess);
}
