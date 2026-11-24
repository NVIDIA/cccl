// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/execution>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstddef>
#include <sstream>

#include <cuda_runtime_api.h>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include "stream_registry_factory.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/buffer_utils.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::ExclusiveScan, device_scan_exclusive);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::ExclusiveSum, device_scan_exclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveScan, device_scan_inclusive);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveSum, device_scan_inclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveScanInit, device_scan_inclusive_init);

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_scan_exclusive_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSum, device_scan_exclusive_sum_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_scan_inclusive_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSum, device_scan_inclusive_sum_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanInit, device_scan_inclusive_init_two_phase);

// %PARAM% TEST_LAUNCH lid 0:1:2

namespace stdexec = cuda::std::execution;

template <typename T, typename ValueT>
void check_counting_output(
  cuda::stream_ref stream, cuda::device_ref device, const cuda::device_buffer<T>& output, ValueT start)
{
  auto result = c2h::make_host_buffer<T>(stream, device, output.size(), cuda::no_init);
  c2h::copy_to_host(stream, output, result);
  REQUIRE(cuda::std::equal(result.begin(), result.end(), cuda::counting_iterator<ValueT>{start}));
}

using block_size_check_t = block_size_extracting_op<cuda::std::plus<>>;

// Launcher helper always passes an environment.
// We need a test of simple use to check if default environment works.
// ifdef it out not to spend time compiling and running it twice.
#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device scan exclusive scan works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  const num_items_t num_items = 2;
  auto d_in                   = cuda::constant_iterator(value_t{1});
  auto d_out                  = c2h::make_device_buffer<value_t>(stream, device, num_items, cuda::no_init);

  using selector_t = cub::detail::scan::
    policy_selector_from_types<decltype(d_in), decltype(d_out.data()), value_t, offset_t, block_size_check_t>;

  cuda::compute_capability cc;
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const unsigned int target_block_size = selector_t{}(cc).lookback.threads_per_block;

  auto d_block_size = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  auto init = value_t{42};
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.data(), block_size_check, init, num_items));
  REQUIRE(c2h::read_single(stream, d_out.subspan(0, 1)) == init);
  REQUIRE(c2h::read_single(stream, d_out.subspan(1, 1)) == (init + value_t{1}));

  // Make sure we use default tuning
  REQUIRE(c2h::read_single(stream, d_block_size) == static_cast<unsigned int>(target_block_size));
}

CUB_TEST_CASE("Device scan exclusive scan with FutureValue works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t = int;

  const num_items_t num_items = 4;

  auto d_in  = c2h::make_device_buffer<int>(stream, device, {1, 1, 1, 1});
  auto d_out = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);

  auto init_value_vec = c2h::make_device_buffer<int>(stream, device, {42});
  auto future_init    = cub::FutureValue<int>(init_value_vec.data());

  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(d_in.data(), d_out.data(), cuda::std::plus{}, future_init, num_items));

  auto expected = c2h::make_host_buffer<int>(stream, device, {42, 43, 44, 45});
  REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
}

CUB_TEST_CASE("Device scan exclusive sum works with default environment", "[sum][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t = int;
  using value_t     = int;

  const num_items_t num_items = 2;

  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = c2h::make_device_buffer<value_t>(stream, device, num_items, cuda::no_init);

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.data(), num_items));
  REQUIRE(c2h::read_single(stream, d_out.subspan(0, 1)) == value_t{});
  REQUIRE(c2h::read_single(stream, d_out.subspan(1, 1)) == value_t{} + d_in[0]);
}

CUB_TEST_CASE("Device scan inclusive-scan-init works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t = int;
  using value_t     = int;

  const num_items_t num_items = 3;
  auto d_in                   = cuda::constant_iterator(value_t{1});
  auto d_out                  = c2h::make_device_buffer<value_t>(stream, device, num_items, cuda::no_init);

  const value_t init{10};

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScanInit(d_in, d_out.data(), cuda::std::plus{}, init, num_items));

  check_counting_output(stream, device, d_out, init + 1);
}

#endif // TEST_LAUNCH == 0

template <int ThreadsPerBlock>
struct scan_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ScanPolicy
  {
    return {cub::ScanAlgorithm::lookback,
            {ThreadsPerBlock,
             1,
             cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE,
             cub::CacheLoadModifier::LOAD_DEFAULT,
             cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE,
             cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
             cub::detail::default_delay_constructor_policy(true)},
            {}};
  }
};

template <int ThreadsPerBlock>
struct scan_tuning_transform
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability, cub::ScanPolicy policy) const
    -> cub::ScanPolicy
  {
    policy.algorithm                  = cub::ScanAlgorithm::lookback;
    policy.lookback.threads_per_block = ThreadsPerBlock;
    return policy;
  }
};

// Lookback delays are only slept (via __nanosleep) if the grid has at least 500 blocks, and are replaced by a thread
// fence otherwise. __nanosleep sleeps at most 1 ms per call, so delays above INT_MAX (the largest an int can hold) are
// cheap to run, but they cannot be asserted to have elapsed (the sleep is only guaranteed to be in [0, 2 * t]).
template <cub::LookbackDelayAlgorithm DelayAlgorithm>
struct large_delay_scan_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ScanPolicy
  {
    return {cub::ScanAlgorithm::lookback,
            {32,
             1,
             cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE,
             cub::CacheLoadModifier::LOAD_DEFAULT,
             cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE,
             cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
             cub::LookbackDelayPolicy{DelayAlgorithm, 3'000'000'000u, 3'000'000'000u}},
            {}};
  }
};

struct unrelated_policy
{};

struct unrelated_tuning
{
  // should never be called
  auto operator()(cuda::compute_capability) const -> unrelated_policy
  {
    throw 1337;
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 32>, cuda::std::integral_constant<unsigned int, 64>>;

#if TEST_LAUNCH != 1

using large_delay_algorithms = c2h::type_list<
  cuda::std::integral_constant<cub::LookbackDelayAlgorithm, cub::LookbackDelayAlgorithm::fixed_delay>,
  cuda::std::integral_constant<cub::LookbackDelayAlgorithm, cub::LookbackDelayAlgorithm::exponential_backoff>>;

CUB_TEST("Device scan works with lookback delays above INT_MAX", "[scan][device]", CUB_SMALL, large_delay_algorithms)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr auto delay_algorithm = c2h::get<0, TestType>::value;

  // 32 items per tile, so that we get at least 500 tiles and the delays are actually slept
  constexpr int tile_size = 32;
  constexpr int num_items = 600 * tile_size;

  auto d_in  = cuda::constant_iterator(1);
  auto d_out = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);

  auto env = cuda::execution::tune(large_delay_scan_tuning<delay_algorithm>{});

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(d_in, d_out.data(), num_items, env));
  check_counting_output(stream, device, d_out, 1);
}

CUB_TEST("Device scan exclusive-scan can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);
  auto init      = 42;

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.data(), block_size_check, init, num_items, env));

  check_counting_output(stream, device, d_out, init);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device scan exclusive-scan tuning can be modified", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = 64; // must be different from any value we use in production
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);
  auto init      = 42;

  {
    stream.sync();
    device_scan_exclusive(d_in, d_out.data(), block_size_check, init, num_items, stdexec::env<>{});
    const unsigned int default_block_size = c2h::read_single(stream, d_block_size);
    REQUIRE(default_block_size != target_block_size);
    cuda::fill_bytes(stream, d_block_size, 0);
  }

  auto env = cuda::execution::tune(scan_tuning_transform<target_block_size>{});

  stream.sync();
  device_scan_exclusive(d_in, d_out.data(), block_size_check, init, num_items, env);

  check_counting_output(stream, device, d_out, init);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device scan exclusive-sum can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  auto num_items    = target_block_size;
  auto d_block_size = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  // use block_size_recording_iterator to embed blockDim info in the input type and query after
  // since ExclusiveSum can not take a custom scan_op
  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.data(), num_items, env));

  check_counting_output(stream, device, d_out, 0);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST_CASE("Device scan inclusive sum works with default environment", "[sum][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t = int;
  using value_t     = int;

  const num_items_t num_items = 3;

  auto d_in  = c2h::make_device_buffer<value_t>(stream, device, {1, 1, 1});
  auto d_out = c2h::make_device_buffer<value_t>(stream, device, num_items, cuda::no_init);

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(d_in.data(), d_out.data(), num_items));

  auto expected = c2h::make_host_buffer<value_t>(stream, device, {1, 2, 3});
  REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
}

CUB_TEST_CASE("Device scan inclusive-scan works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  const num_items_t num_items = 2;
  auto d_in                   = cuda::constant_iterator(value_t{1});
  auto d_out                  = c2h::make_device_buffer<value_t>(stream, device, num_items, cuda::no_init);

  using selector_t = cub::detail::scan::
    policy_selector_from_types<decltype(d_in), decltype(d_out.data()), value_t, offset_t, block_size_check_t>;

  cuda::compute_capability cc;
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const unsigned int target_block_size = selector_t{}(cc).lookback.threads_per_block;

  auto d_block_size = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScan(d_in, d_out.data(), block_size_check, num_items));
  REQUIRE(c2h::read_single(stream, d_out.subspan(0, 1)) == d_in[0]);
  REQUIRE(c2h::read_single(stream, d_out.subspan(1, 1)) == d_in[0] + d_in[1]);

  // Make sure we use default tuning
  REQUIRE(c2h::read_single(stream, d_block_size) == static_cast<unsigned int>(target_block_size));
}

CUB_TEST("Device scan inclusive-scan can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScan(d_in, d_out.data(), block_size_check, num_items, env));

  check_counting_output(stream, device, d_out, 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device scan inclusive-scan-init can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);

  const int init{10};

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(
    cudaSuccess == cub::DeviceScan::InclusiveScanInit(d_in, d_out.data(), block_size_check, init, num_items, env));

  check_counting_output(stream, device, d_out, init + 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST_CASE("Device scan exclusive sum in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});

  auto error = cub::DeviceScan::ExclusiveSum(d_data.data(), static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 3, 6});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST_CASE("Device scan exclusive scan in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});

  auto error = cub::DeviceScan::ExclusiveScan(d_data.data(), cuda::std::plus{}, 42, static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::make_host_buffer<int>(stream, device, {42, 43, 45, 48});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST_CASE("Device scan inclusive sum in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});

  auto error = cub::DeviceScan::InclusiveSum(d_data.data(), static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, 10});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST_CASE("Device scan inclusive scan in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});

  auto error = cub::DeviceScan::InclusiveScan(d_data.data(), cuda::std::plus{}, static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, 10});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST_CASE("Device scan exclusive scan with FutureValue in-place works with default environment",
              "[scan][device]",
              CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});

  auto init_value_vec = c2h::make_device_buffer<int>(stream, device, {42});
  auto future_init    = cub::FutureValue<int>(init_value_vec.data());

  auto error =
    cub::DeviceScan::ExclusiveScan(d_data.data(), cuda::std::plus{}, future_init, static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::make_host_buffer<int>(stream, device, {42, 43, 45, 48});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

#endif // TEST_LAUNCH != 1

CUB_TEST("Device scan exclusive-scan uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();
  const auto device = c2h::current_device();

  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1.0f);
  auto d_out                  = c2h::make_device_buffer<float>(stream, num_items, cuda::no_init);

  using init_value_t = float;

  const init_value_t init{42.0f};

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(
            nullptr, expected_bytes_allocated, d_in, d_out.data(), scan_op_t{}, init, num_items));

  auto env = stdexec::env{cuda::stream_ref{stream},
                          cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive(d_in, d_out.data(), scan_op_t{}, init, num_items, env);

  check_counting_output(stream, device, d_out, static_cast<int>(init));
}

CUB_TEST("Device scan exclusive-scan with FutureValue uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();
  const auto device = c2h::current_device();

  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1);
  auto d_out                  = c2h::make_device_buffer<int>(stream, num_items, cuda::no_init);

  auto init_value_vec = c2h::make_device_buffer<int>(stream, {42});
  auto future_init    = cub::FutureValue<int>(init_value_vec.data());

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(
            nullptr, expected_bytes_allocated, d_in, d_out.data(), scan_op_t{}, future_init, num_items));

  auto env = stdexec::env{cuda::stream_ref{stream},
                          cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive(d_in, d_out.data(), scan_op_t{}, future_init, num_items, env);

  check_counting_output(stream, device, d_out, 42);
}

CUB_TEST("Device scan exclusive-sum uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();
  const auto device = c2h::current_device();

  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1.0f);
  auto d_out                  = c2h::make_device_buffer<float>(stream, num_items, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess == cub::DeviceScan::ExclusiveSum(nullptr, expected_bytes_allocated, d_in, d_out.data(), num_items));

  auto env = stdexec::env{cuda::stream_ref{stream},
                          cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive_sum(d_in, d_out.data(), num_items, env);

  check_counting_output(stream, device, d_out, 0);
}

CUB_TEST("Device scan inclusive-sum uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using num_items_t = int;

  const num_items_t num_items = GENERATE(0, 1, 10);
  auto d_in                   = c2h::make_device_buffer<int>(stream, num_items, 3);
  auto d_out                  = c2h::make_device_buffer<int>(stream, num_items, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveSum(nullptr, expected_bytes_allocated, d_in.data(), d_out.data(), num_items));

  auto env = stdexec::env{cuda::stream_ref{stream},
                          cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_inclusive_sum(d_in.data(), d_out.data(), num_items, env);

  auto expected = c2h::make_host_buffer<int>(stream, num_items, cuda::no_init);
  for (int i = 0; i < num_items; i++)
  {
    expected[i] = 3 * (i + 1);
  }
  REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
}

CUB_TEST("Device scan inclusive-scan uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();
  const auto device = c2h::current_device();

  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1.0f);
  auto d_out                  = c2h::make_device_buffer<float>(stream, num_items, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(nullptr, expected_bytes_allocated, d_in, d_out.data(), scan_op_t{}, num_items));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)}; // temp storage
                                                                                                         // size

  device_scan_inclusive(d_in, d_out.data(), scan_op_t{}, num_items, env);

  check_counting_output(stream, device, d_out, 1);
}

CUB_TEST("Device scan inclusive-scan-init uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using num_items_t = int;

  const num_items_t num_items = 4;
  auto d_in                   = c2h::make_device_buffer<float>(stream, {1.0f, 2.0f, 3.0f, 4.0f});
  auto d_out                  = c2h::make_device_buffer<float>(stream, num_items, cuda::no_init);
  const float init            = 10.0f;

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScanInit(
            nullptr, expected_bytes_allocated, d_in.data(), d_out.data(), cuda::std::plus{}, init, num_items));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)}; // temp storage
                                                                                                         // size

  device_scan_inclusive_init(d_in.data(), d_out.data(), cuda::std::plus{}, init, num_items, env);

  auto expected = c2h::make_host_buffer<float>(stream, {11.0f, 13.0f, 16.0f, 20.0f});
  REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
}

CUB_TEST("Device scan exclusive-sum in-place uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto d_data = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4});

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            nullptr, expected_bytes_allocated, d_data.data(), d_data.data(), static_cast<int>(d_data.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive_sum(d_data.data(), static_cast<int>(d_data.size()), env);

  auto expected = c2h::make_host_buffer<int>(stream, {0, 1, 3, 6});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST("Device scan exclusive-scan in-place uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto d_data = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4});

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      nullptr,
      expected_bytes_allocated,
      d_data.data(),
      d_data.data(),
      cuda::std::plus{},
      42,
      static_cast<int>(d_data.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive(d_data.data(), cuda::std::plus{}, 42, static_cast<int>(d_data.size()), env);

  auto expected = c2h::make_host_buffer<int>(stream, {42, 43, 45, 48});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST("Device scan exclusive-scan with FutureValue in-place uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto d_data = c2h::make_device_buffer<int>(stream, {1, 1, 1, 1});

  auto init_value_vec = c2h::make_device_buffer<int>(stream, {42});
  auto future_init    = cub::FutureValue<int>(init_value_vec.data());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      nullptr,
      expected_bytes_allocated,
      d_data.data(),
      d_data.data(),
      cuda::std::plus{},
      future_init,
      static_cast<int>(d_data.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive(d_data.data(), cuda::std::plus{}, future_init, static_cast<int>(d_data.size()), env);

  auto expected = c2h::make_host_buffer<int>(stream, {42, 43, 44, 45});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST("Device scan inclusive-sum in-place uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto d_data = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4});

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveSum(
            nullptr, expected_bytes_allocated, d_data.data(), d_data.data(), static_cast<int>(d_data.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive_sum(d_data.data(), static_cast<int>(d_data.size()), env);

  auto expected = c2h::make_host_buffer<int>(stream, {1, 3, 6, 10});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

CUB_TEST("Device scan inclusive-scan in-place uses environment", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto d_data = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4});

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      nullptr,
      expected_bytes_allocated,
      d_data.data(),
      d_data.data(),
      cuda::std::plus{},
      static_cast<int>(d_data.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive(d_data.data(), cuda::std::plus{}, static_cast<int>(d_data.size()), env);

  auto expected = c2h::make_host_buffer<int>(stream, {1, 3, 6, 10});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

#if TEST_LAUNCH == 0

// The launch wrappers copy their arguments, which the non-copyable stream types of test_with_custom_streams do not
// allow, so the two-phase overloads are called directly with custom streams. Test them with host launch only.

// Runs the two-phase overload wrapped by two_phase: queries the temporary storage size, executes with the allocated
// storage, and checks that all kernels are launched on the stream of env.
template <class EnvT, class TwoPhaseFn>
void run_two_phase(const EnvT& env, TwoPhaseFn two_phase)
{
  const auto stream = cuda::__call_or(cuda::get_stream, c2h::default_stream_ref(), env);
  const stream_scope scope{stream.get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  auto temp_storage =
    c2h::make_device_buffer<cuda::std::uint8_t>(stream, c2h::current_device(), temp_storage_bytes, cuda::no_init);
  REQUIRE(cudaSuccess == two_phase(temp_storage.data(), temp_storage_bytes));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

CUB_TEST_CASE("Device scan exclusive-sum two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 3, 6});
    REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan exclusive-sum in-place two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  const int num_items{static_cast<int>(d_data.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_data.data(), num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 3, 6});
    REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan exclusive-scan two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), cuda::std::plus<>{}, 10, num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {10, 11, 13, 16});
    REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan exclusive-scan in-place two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  const int num_items{static_cast<int>(d_data.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage, temp_storage_bytes, d_data.data(), cuda::std::plus<>{}, 10, num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {10, 11, 13, 16});
    REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan exclusive-scan with FutureValue two-phase works with custom streams",
              "[scan][device]",
              CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  auto d_init     = c2h::make_device_buffer<int>(stream, device, {10});
  const int num_items{static_cast<int>(d_in.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_in.data(),
        d_out.data(),
        cuda::std::plus<>{},
        cub::FutureValue<int>(d_init.data()),
        num_items,
        env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {10, 11, 13, 16});
    REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan exclusive-scan with FutureValue in-place two-phase works with custom streams",
              "[scan][device]",
              CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_init = c2h::make_device_buffer<int>(stream, device, {10});
  const int num_items{static_cast<int>(d_data.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_data.data(),
        cuda::std::plus<>{},
        cub::FutureValue<int>(d_init.data()),
        num_items,
        env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {10, 11, 13, 16});
    REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan inclusive-sum two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, 10});
    REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan inclusive-sum in-place two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  const int num_items{static_cast<int>(d_data.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_data.data(), num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, 10});
    REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan inclusive-scan two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScan(
        d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), cuda::std::plus<>{}, num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, 10});
    REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan inclusive-scan in-place two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  const int num_items{static_cast<int>(d_data.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScan(
        d_temp_storage, temp_storage_bytes, d_data.data(), cuda::std::plus<>{}, num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, 10});
    REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan inclusive-scan-init two-phase works with custom streams", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScanInit(
        d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), cuda::std::plus<>{}, 10, num_items, env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {11, 13, 16, 20});
    REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
  });
}

CUB_TEST_CASE("Device scan inclusive-scan-init with deferred init two-phase works with custom streams",
              "[scan][device]",
              CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in   = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out        = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const auto d_init = c2h::make_device_buffer<int>(stream, device, {10});
  const int num_items{static_cast<int>(d_in.size())};

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScanInit(
        d_temp_storage,
        temp_storage_bytes,
        d_in.data(),
        d_out.data(),
        cuda::std::plus<>{},
        cuda::args::deferred(d_init.data()),
        num_items,
        env);
    });

    const auto expected = c2h::make_host_buffer<int>(stream, device, {11, 13, 16, 20});
    REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
  });
}

// The tuning and determinism environments are empty classes, which CTK 12.x cannot pass as kernel parameters, so these
// tests cannot use the device launch (TEST_LAUNCH == 1).
CUB_TEST("Device scan exclusive-scan two-phase can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  const int num_items = 3;
  auto d_in           = cuda::constant_iterator(1);
  auto d_out          = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);
  const int init{42};

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  stream.sync();
  device_scan_exclusive_two_phase(d_in, d_out.data(), block_size_check, init, num_items, env);

  check_counting_output(stream, device, d_out, init);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device scan exclusive-sum two-phase can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const auto num_items = static_cast<int>(target_block_size);
  auto d_block_size    = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  // ExclusiveSum can not take a custom scan_op, so the input iterator records the block size
  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  stream.sync();
  device_scan_exclusive_sum_two_phase(d_in, d_out.data(), num_items, env);

  check_counting_output(stream, device, d_out, 0);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device scan inclusive-scan two-phase can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  const int num_items = 3;
  auto d_in           = cuda::constant_iterator(1);
  auto d_out          = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  stream.sync();
  device_scan_inclusive_two_phase(d_in, d_out.data(), block_size_check, num_items, env);

  check_counting_output(stream, device, d_out, 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device scan inclusive-scan-init two-phase can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_t block_size_check{d_block_size.data()};

  const int num_items = 3;
  auto d_in           = cuda::constant_iterator(1);
  auto d_out          = c2h::make_device_buffer<int>(stream, device, num_items, cuda::no_init);
  const int init{10};

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  stream.sync();
  device_scan_inclusive_init_two_phase(d_in, d_out.data(), block_size_check, init, num_items, env);

  check_counting_output(stream, device, d_out, init + 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

// The two-phase overloads read the determinism requirement from the environment like the single-phase ones: a
// run_to_run floating-point sum selects the stable reduction order, so both overloads produce bitwise equal results.
CUB_TEST_CASE("Device scan exclusive-scan two-phase honors determinism requirements", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const int num_items = 10'000;
  auto d_in           = c2h::make_device_buffer<float>(stream, num_items, cuda::no_init);
  c2h::gen(C2H_SEED(1), d_in, -100.0f, 100.0f);

  const auto env =
    stdexec::env{cuda::stream_ref{stream}, cuda::execution::require(cuda::execution::determinism::run_to_run)};
  const auto op = cuda::std::plus<float>{};

  auto d_reference = c2h::make_device_buffer<float>(stream, num_items, cuda::no_init);
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in.data(), d_reference.data(), op, 0.0f, num_items, env));

  auto d_out = c2h::make_device_buffer<float>(stream, num_items, cuda::no_init);
  device_scan_exclusive_two_phase(d_in.data(), d_out.data(), op, 0.0f, num_items, env);

  auto reference = c2h::make_host_buffer<float>(stream, num_items, cuda::no_init);
  auto result    = c2h::make_host_buffer<float>(stream, num_items, cuda::no_init);
  c2h::copy_to_host(stream, d_reference, reference);
  c2h::copy_to_host(stream, d_out, result);
  REQUIRE_THAT(reference, detail::BitwiseEqualsRange(result));
}

#endif // TEST_LAUNCH == 0

// The graph launch (TEST_LAUNCH == 2) appends its own stream argument, so it cannot be combined with an environment.
#if TEST_LAUNCH != 2

// Before the environment parameter, the two-phase overloads took `cudaStream_t stream = nullptr`, so callers passing
// nullptr or a literal 0 for the stream exist. Both must keep compiling and keep running on the default stream.
CUB_TEST_CASE("Device scan two-phase overloads accept legacy null stream arguments", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  const stream_scope scope{cudaStream_t{}};

  SECTION("exclusive-sum with nullptr")
  {
    stream.sync();
    device_scan_exclusive_sum_two_phase(d_in.data(), d_out.data(), num_items, nullptr);
  }

  SECTION("exclusive-sum with literal 0")
  {
    stream.sync();
    device_scan_exclusive_sum_two_phase(d_in.data(), d_out.data(), num_items, 0);
  }

  SECTION("exclusive-scan with nullptr")
  {
    stream.sync();
    device_scan_exclusive_two_phase(d_in.data(), d_out.data(), cuda::std::plus<>{}, 0, num_items, nullptr);
  }

  SECTION("exclusive-scan with literal 0")
  {
    stream.sync();
    device_scan_exclusive_two_phase(d_in.data(), d_out.data(), cuda::std::plus<>{}, 0, num_items, 0);
  }

  const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 3, 6});
  REQUIRE_THAT(d_out, c2h::Equals(stream, expected));
}

CUB_TEST_CASE("Device scan in-place two-phase overloads accept a legacy nullptr stream", "[scan][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_data = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  const int num_items{static_cast<int>(d_data.size())};

  const stream_scope scope{cudaStream_t{}};
  stream.sync();
  device_scan_inclusive_sum_two_phase(d_data.data(), num_items, nullptr);

  const auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, 10});
  REQUIRE_THAT(d_data, c2h::Equals(stream, expected));
}

#endif // TEST_LAUNCH != 2

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test ScanPolicy properties", "[scan][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::ScanPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ScanPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::ScanLookbackPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ScanLookbackPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::ScanLookaheadPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ScanLookaheadPolicy>);

  // aggregate init
  constexpr auto p1_lb = cub::ScanLookbackPolicy{
    256,
    11,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_DEFAULT,
    cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
    cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 832, 1165}};
  constexpr auto p1_la = cub::ScanLookaheadPolicy{3, 8, 4, 2};
  constexpr auto p1    = cub::ScanPolicy{cub::ScanAlgorithm::lookback, p1_lb, p1_la};
  STATIC_REQUIRE(p1_lb.tiles_per_batch == 65536);
  constexpr auto custom_batch = [](auto p) {
    p.tiles_per_batch = 17;
    return p;
  }(p1_lb);
  STATIC_REQUIRE(custom_batch.tiles_per_batch == 17);
  STATIC_REQUIRE(custom_batch != p1_lb);
  STATIC_REQUIRE_FALSE(custom_batch == p1_lb);

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2_lb = cub::ScanLookbackPolicy{
    .threads_per_block = 256,
    .items_per_thread  = 11,
    .load_algorithm    = cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    .load_modifier     = cub::CacheLoadModifier::LOAD_DEFAULT,
    .store_algorithm   = cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
    .scan_algorithm    = cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
    .lookback_delay    = cub::LookbackDelayPolicy{
      .kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 832, .l2_write_latency = 1165}};
  constexpr auto p2_la = cub::ScanLookaheadPolicy{
    .reduce_and_scan_warps = 3, .items_per_thread = 8, .lookahead_items_per_thread = 4, .lookahead_stages = 2};

  constexpr auto p2 = cub::ScanPolicy{.algorithm = cub::ScanAlgorithm::lookback, .lookback = p2_lb, .lookahead = p2_la};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2_lb = p1_lb;
  constexpr auto p2_la = p1_la;
  constexpr auto p2    = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1_lb == p2_lb);
  STATIC_REQUIRE_FALSE(p1_lb != p2_lb);

  STATIC_REQUIRE(p1_la == p2_la);
  STATIC_REQUIRE_FALSE(p1_la != p2_la);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1_lb)
          == "ScanLookbackPolicy { .threads_per_block = 256, .items_per_thread = 11"
             ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
             ", .store_algorithm = BLOCK_STORE_DIRECT, .scan_algorithm = BLOCK_SCAN_RAKING"
             ", .lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
             ", .delay = 832, .l2_write_latency = 1165 }, .tiles_per_batch = 65536 }");
  REQUIRE(to_string(p1_la)
          == "ScanLookaheadPolicy { .reduce_and_scan_warps = 3, .items_per_thread = 8"
             ", .lookahead_items_per_thread = 4, .lookahead_stages = 2, .block_idx_stages = -1 }");
  REQUIRE(
    to_string(p1)
    == "ScanPolicy { .algorithm = ScanAlgorithm::lookback"
       ", .lookback = ScanLookbackPolicy { .threads_per_block = 256, .items_per_thread = 11"
       ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
       ", .store_algorithm = BLOCK_STORE_DIRECT, .scan_algorithm = BLOCK_SCAN_RAKING"
       ", .lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 832, .l2_write_latency = 1165 }, .tiles_per_batch = 65536 }"
       ", .lookahead = ScanLookaheadPolicy { .reduce_and_scan_warps = 3, .items_per_thread = 8"
       ", .lookahead_items_per_thread = 4, .lookahead_stages = 2, .block_idx_stages = -1 } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
