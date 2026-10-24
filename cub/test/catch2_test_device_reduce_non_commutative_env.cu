// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <cuda/execution>
#include <cuda/iterator>
#include <cuda/std/execution>

#include <cstdint>

#include "block_size_extracting_helpers.h"
#include "catch2_test_device_reduce_non_commutative.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::ReduceNonCommutative, device_reduce_non_commutative);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

// Custom tuning that forces a specific block size, used to verify a tuning environment reaches the kernel.
template <int ThreadsPerBlock>
struct reduce_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ReducePolicy
  {
    const auto policy =
      cub::ReducePassPolicy{ThreadsPerBlock, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 32>, cuda::std::integral_constant<unsigned int, 64>>;

using requirements =
  c2h::type_list<cuda::execution::determinism::run_to_run_t, cuda::execution::determinism::not_guaranteed_t>;

// Large enough for the two-pass path, which needs temporary storage
constexpr std::int64_t num_items = 1 << 20;

[[nodiscard]] auto make_input()
{
  return cuda::make_transform_iterator(cuda::counting_iterator<std::int64_t>{0}, index_to_run_t{});
}

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device reduce non-commutative works with default environment", "[reduce][device]", CUB_SMALL)
{
  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  REQUIRE(cudaSuccess
          == cub::DeviceReduce::ReduceNonCommutative(
            make_input(), d_out.begin(), num_items, concatenate_runs_t{}, initial_run));
  REQUIRE(d_out[0] == expected_run(num_items));
}

CUB_TEST("Device reduce non-commutative uses custom stream", "[reduce][device]", CUB_SMALL)
{
  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  const cuda::stream stream = c2h::make_current_device_stream();

  SECTION("single-phase")
  {
    REQUIRE(cudaSuccess
            == cub::DeviceReduce::ReduceNonCommutative(
              make_input(), d_out.begin(), num_items, concatenate_runs_t{}, initial_run, stream));
  }

  SECTION("two-phase")
  {
    size_t storage_bytes{};
    REQUIRE(
      cudaSuccess
      == cub::DeviceReduce::ReduceNonCommutative(
        nullptr, storage_bytes, make_input(), d_out.begin(), num_items, concatenate_runs_t{}, initial_run, stream));
    c2h::device_vector<std::uint8_t> temp_storage(storage_bytes, thrust::no_init);
    REQUIRE(
      cudaSuccess
      == cub::DeviceReduce::ReduceNonCommutative(
        thrust::raw_pointer_cast(temp_storage.data()),
        storage_bytes,
        make_input(),
        d_out.begin(),
        num_items,
        concatenate_runs_t{},
        initial_run,
        stream));
  }

  stream.sync();
  REQUIRE(d_out[0] == expected_run(num_items));
}

#endif // TEST_LAUNCH == 0

CUB_TEST("Device reduce non-commutative uses environment", "[reduce][device]", CUB_SMALL)
{
  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ReduceNonCommutative(
      nullptr, expected_bytes_allocated, make_input(), d_out.begin(), num_items, concatenate_runs_t{}, initial_run));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_reduce_non_commutative(make_input(), d_out.begin(), num_items, concatenate_runs_t{}, initial_run, env);

  REQUIRE(d_out[0] == expected_run(num_items));
}

#if TEST_LAUNCH != 1

CUB_TEST("Device reduce non-commutative can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op<concatenate_runs_t> op{thrust::raw_pointer_cast(d_block_size.data())};
  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  device_reduce_non_commutative(make_input(), d_out.begin(), num_items, op, initial_run, env);

  REQUIRE(d_out[0] == expected_run(num_items));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device reduce non-commutative accepts determinism requirements", "[reduce][device]", CUB_SMALL, requirements)
{
  using determinism_t = c2h::get<0, TestType>;

  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  // Not guaranteed is still met by the in-order implementation
  auto env = stdexec::env{cuda::execution::require(determinism_t{})};

  device_reduce_non_commutative(make_input(), d_out.begin(), num_items, concatenate_runs_t{}, initial_run, env);

  REQUIRE(d_out[0] == expected_run(num_items));
}

#endif // TEST_LAUNCH != 1
