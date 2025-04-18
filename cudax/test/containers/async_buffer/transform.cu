//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/device/device_transform.cuh>

#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/memory_resource>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/container.cuh>

#include <algorithm>

#include "helper.h"
#include "types.h"

using cub::detail::transform::Algorithm;

template <Algorithm Alg>
struct policy_hub_for_alg
{
  struct max_policy : cub::ChainedPolicy<300, max_policy, max_policy>
  {
    static constexpr int min_bif         = 64 * 1024;
    static constexpr Algorithm algorithm = Alg;
    using algo_policy =
      ::cuda::std::_If<Alg == Algorithm::prefetch,
                       cub::detail::transform::prefetch_policy_t<256>,
                       cub::detail::transform::async_copy_policy_t<256>>;
  };
};

template <Algorithm Alg,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp>
CUB_RUNTIME_FUNCTION static cudaError_t transform_many_with_alg(
  ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
  RandomAccessIteratorOut output,
  Offset num_items,
  TransformOp transform_op,
  cudaStream_t stream = nullptr)
{
  return cub::detail::transform::dispatch_t<cub::detail::transform::requires_stable_address::no,
                                            Offset,
                                            ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                            RandomAccessIteratorOut,
                                            TransformOp,
                                            policy_hub_for_alg<Alg>>{}
    .dispatch(inputs, output, num_items, transform_op, stream);
}

using algorithms =
  c2h::type_list<::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
                 ,
                 ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>
#endif // _CUB_HAS_TRANSFORM_UBLKCP
                 >;

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
#  define FILTER_UBLKCP                                \
    if (alg == Algorithm::ublkcp && ptx_version < 900) \
    {                                                  \
      return;                                          \
    }
#else // _CUB_HAS_TRANSFORM_UBLKCP
#  define FILTER_UBLKCP
#endif // _CUB_HAS_TRANSFORM_UBLKCP

#define FILTER_UNSUPPORTED_ALGS                                           \
  int ptx_version = 0;                                                    \
  REQUIRE(cub::PtxVersion(ptx_version) == cudaSuccess);                   \
  _CCCL_DIAG_PUSH                                                         \
  _CCCL_DIAG_SUPPRESS_MSVC(4127) /* conditional expression is constant */ \
  FILTER_UBLKCP                                                           \
  _CCCL_DIAG_POP

C2H_TEST("DeviceTransform::Transform cudax::async_device_buffer", "[device][device_transform]", algorithms)
{
  using type         = int;
  constexpr auto alg = c2h::get<0, TestType>::value;
  FILTER_UNSUPPORTED_ALGS
  const int num_items = 1 << 24;

  cudax::stream stream{};
  cudax::env_t<cuda::mr::device_accessible> env{cudax::device_memory_resource{}, stream};

  cudax::async_device_buffer<type> a{env, num_items, cudax::uninit};
  cudax::async_device_buffer<type> b{env, num_items, cudax::uninit};
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), a.begin(), a.end());
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), b.begin(), b.end());

  cudax::async_device_buffer<type> result{env, num_items, cudax::uninit};

  transform_many_with_alg<alg>(
    ::cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, ::cuda::std::plus<type>{});

  // copy back to host
  thrust::host_vector<type> a_h(num_items);
  thrust::host_vector<type> b_h(num_items);
  thrust::host_vector<type> result_h(num_items);
  REQUIRE(cudaMemcpyAsync(a_h.data(), a.data(), num_items * sizeof(type), cudaMemcpyDeviceToHost, stream.get())
          == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(b_h.data(), b.data(), num_items * sizeof(type), cudaMemcpyDeviceToHost, stream.get())
          == cudaSuccess);
  REQUIRE(
    cudaMemcpyAsync(result_h.data(), result.data(), num_items * sizeof(type), cudaMemcpyDeviceToHost, stream.get())
    == cudaSuccess);
  stream.sync();

  // compute reference and verify
  thrust::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result_h);
}
