// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/stream>

#include <iostream>

#include "cub_test_macros.h"

#if _CCCL_STD_VER >= 2020

// nvcc turns the `.member = value,` C++ syntax into GNU's `member: value,` when clang (14 - 21) is used
_CCCL_DIAG_PUSH
#  if _CCCL_COMPILER(CLANG)
_CCCL_DIAG_SUPPRESS_CLANG("-Wgnu-designator")
#  endif // _CCCL_COMPILER(CLANG)

// example-begin transform-policy-selector
struct TransformPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability /*cc*/) const -> cub::TransformPolicy
  {
    return {.min_bytes_in_flight = 64 * 1024,
            .algorithm           = cub::TransformAlgorithm::prefetch,
            .prefetch            = {.threads_per_block = 256},
            .vectorized          = {}, // unused because algorithm is prefetch
            .async_copy          = {}}; // unused because algorithm is prefetch
  }
};
// example-end transform-policy-selector

_CCCL_DIAG_POP

CUB_TEST("cub::DeviceTransform::Transform accepts a custom policy selector", "[transform][env]", CUB_SMALL)
{
  // example-begin transform-tuning
  auto d_input  = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7};
  auto d_output = thrust::device_vector<int>(7, thrust::no_init);

  const auto error = cub::DeviceTransform::Transform(
    d_input.data(),
    d_output.data(),
    d_input.size(),
    cuda::std::negate{},
    cuda::execution::tune(TransformPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTransform::Transform failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{-1, -2, -3, -4, -5, -6, -7};
  // example-end transform-tuning

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_output == expected);
}

CUB_TEST("cub::DeviceTransform::Transform accepts a runtime min_bytes_in_flight override",
         "[transform][env]",
         CUB_SMALL)
{
  // example-begin transform-min-bytes-in-flight
  auto d_input  = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7};
  auto d_output = thrust::device_vector<int>(7, thrust::no_init);

  // Ask the dispatch to keep at least 192 KiB per SM in flight when sizing the tiles. Unlike a policy selector, this
  // is a runtime value: it changes the grid configuration, not the kernel.
  const auto error = cub::DeviceTransform::Transform(
    d_input.data(),
    d_output.data(),
    d_input.size(),
    cuda::std::negate{},
    cuda::std::execution::env{cuda::get_stream(cuda::stream_ref{cudaStream_t{}}),
                              cuda::execution::min_bytes_in_flight(192 * 1024)});

  thrust::device_vector<int> expected{-1, -2, -3, -4, -5, -6, -7};
  // example-end transform-min-bytes-in-flight

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_output == expected);
}

// A launcher factory that records the grid of the last launch, so we can observe the tile size the dispatch picked.
struct recording_launcher_factory_t : cub::detail::TripleChevronFactory
{
  static unsigned int& last_grid_dim_x()
  {
    static unsigned int value = 0;
    return value;
  }

  CUB_RUNTIME_FUNCTION THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron operator()(
    dim3 grid,
    dim3 block,
    ::cuda::std::size_t shared_mem,
    ::cudaStream_t stream,
    bool dependent_launch = false,
    dim3 cluster_dim      = dim3{0, 0, 0}) const
  {
    NV_IF_TARGET(NV_IS_HOST, (last_grid_dim_x() = grid.x;));
    return cub::detail::TripleChevronFactory::operator()(
      grid, block, shared_mem, stream, dependent_launch, cluster_dim);
  }
};

// Negates n ints through the dispatch with the given override and returns the grid size that was launched.
inline unsigned int grid_for_min_bytes_in_flight(int min_bytes_in_flight, cuda::std::int64_t n)
{
  auto d_input  = thrust::device_vector<int>(n, 3);
  auto d_output = thrust::device_vector<int>(n, thrust::no_init);

  using policy_selector_t =
    cub::detail::transform::policy_selector_from_types<false, true, cuda::std::tuple<int*>, int*>;

  recording_launcher_factory_t::last_grid_dim_x() = 0;
  const auto error = cub::detail::transform::dispatch<cub::detail::transform::requires_stable_address::no>(
    cuda::std::make_tuple(thrust::raw_pointer_cast(d_input.data())),
    thrust::raw_pointer_cast(d_output.data()),
    n,
    cuda::always_true{},
    cuda::std::negate{},
    cudaStream_t{},
    policy_selector_t{},
    {},
    recording_launcher_factory_t{},
    min_bytes_in_flight);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(thrust::count(d_output.begin(), d_output.end(), -3) == n);
  return recording_launcher_factory_t::last_grid_dim_x();
}

CUB_TEST("cub::DeviceTransform min_bytes_in_flight override changes the tile size",
         "[transform][env]",
         CUB_SMALL)
{
  const cuda::std::int64_t n = cuda::std::int64_t{1} << 22;

  const unsigned int grid_default = grid_for_min_bytes_in_flight(0, n);
  const unsigned int grid_tiny    = grid_for_min_bytes_in_flight(1, n); // one item per thread reaches 1 byte
  const unsigned int grid_huge    = grid_for_min_bytes_in_flight(1 << 30, n); // runs into max_items_per_thread
  REQUIRE(grid_default > 0);
  REQUIRE(grid_tiny >= grid_default); // smaller tiles, at least as many blocks
  REQUIRE(grid_huge <= grid_default); // larger tiles, at most as many blocks
  REQUIRE(grid_tiny > grid_huge);
  REQUIRE(grid_for_min_bytes_in_flight(0, n) == grid_default); // the override does not leak into later calls
}

CUB_TEST("cub::DeviceTransform does not reuse the configuration of a different min_bytes_in_flight",
         "[transform][env]",
         CUB_SMALL)
{
  // The dispatch caches the (occupancy, items per thread) configuration per kernel. Two calls with different
  // targets must each get their own configuration, whatever the order of the calls.
  const cuda::std::int64_t n = cuda::std::int64_t{1} << 22;
  const int small_target     = 1;
  const int large_target     = 1 << 30;

  const unsigned int grid_small_first = grid_for_min_bytes_in_flight(small_target, n);
  const unsigned int grid_large_first = grid_for_min_bytes_in_flight(large_target, n);
  REQUIRE(grid_small_first != grid_large_first);

  // same values again, in both orders: each call must reproduce the grid of its own target
  REQUIRE(grid_for_min_bytes_in_flight(large_target, n) == grid_large_first);
  REQUIRE(grid_for_min_bytes_in_flight(small_target, n) == grid_small_first);
  REQUIRE(grid_for_min_bytes_in_flight(small_target, n) == grid_small_first);
  REQUIRE(grid_for_min_bytes_in_flight(large_target, n) == grid_large_first);

  // a third, intermediate value gets its own configuration too and does not disturb the first two
  const unsigned int grid_mid = grid_for_min_bytes_in_flight(64 * 1024, n);
  REQUIRE(grid_mid <= grid_small_first);
  REQUIRE(grid_mid >= grid_large_first);
  REQUIRE(grid_for_min_bytes_in_flight(small_target, n) == grid_small_first);
  REQUIRE(grid_for_min_bytes_in_flight(large_target, n) == grid_large_first);
}

#else // _CCCL_STD_VER >= 2020

// we need a dummy test for C++17, otherwise the return code of the test executable is 2 (not 0)
CUB_TEST("cub::DeviceTransform::Transform dummy test", "[transform][env]", CUB_SMALL)
{
  SUCCEED();
}

#endif // _CCCL_STD_VER >= 2020
