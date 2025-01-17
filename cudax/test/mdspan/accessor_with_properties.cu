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

#include <testing.cuh>

template <typename T, typename E, typename L, typename A>
__global__ void mdspan_accessor_kernel(cuda::std::mdspan<T, E, L, A> md)
{
  CUDAX_REQUIRE(md(threadIdx.x) == threadIdx.x);
  md(threadIdx.x) = threadIdx.x * 2;
  __syncthreads();
  CUDAX_REQUIRE(md(threadIdx.x) == threadIdx.x * 2);
}

using namespace cuda::experimental;

template <typename... Ts>
using type_list = ::cuda::std::__type_list<Ts...>;

template <typename... Ts>
using __type_cartesian_product = ::cuda::std::__type_cartesian_product<Ts...>;

using memory_behavior_list = type_list<read_only_t, read_write_t>;

using eviction_list =
  type_list<eviction_none_t,
            eviction_normal_t,
            eviction_first_t, //
            eviction_last_t,
            eviction_last_use_t,
            eviction_no_alloc_t>;

using alignment_list = type_list<alignment_t<4>, alignment_t<8>>;

using prefetch_list = type_list<no_prefetch_spatial_t, prefetch_64B_t, prefetch_128B_t, prefetch_256B_t>;

using aliasing_list = type_list<ptr_may_alias_t, ptr_no_aliasing_t>;

using TypeLists = ::cuda::std::
  __type_cartesian_product<memory_behavior_list, eviction_list, alignment_list, prefetch_list, aliasing_list>;

TEMPLATE_LIST_TEST_CASE("Accessor", "[device]", TypeLists)
{
  using MemoryBehavior = ::cuda::std::__type_at_c<0, TestType>;
  using EvictionPolicy = ::cuda::std::__type_at_c<1, TestType>;
  using Alignment      = ::cuda::std::__type_at_c<2, TestType>;
  using Prefetch       = ::cuda::std::__type_at_c<3, TestType>;
  using Aliasing       = ::cuda::std::__type_at_c<4, TestType>;

  thrust::host_vector<int> h_vector(32);
  std::iota(h_vector.begin(), h_vector.end(), 0);
  thrust::device_vector<int> d_vector = h_vector;

  auto md = cuda::std::mdspan(thrust::raw_pointer_cast(d_vector.data()), d_vector.size());
  auto md_with_props =
    cuda::experimental::add_properties(md, MemoryBehavior{}, EvictionPolicy{}, Alignment{}, Prefetch{}, Aliasing{});
  mdspan_accessor_kernel<<<1, 32>>>(md_with_props);
  cudaDeviceSynchronize();
  CUDAX_REQUIRE(cudaGetLastError() == cudaSuccess);
}
