// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <thrust/copy.h>

#include <cuda/__execution/tune.h>
#include <cuda/std/functional>

#include <cstdint>

#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);

// trivially constructible types to allow uninitialized thrust vector
using types    = c2h::type_list<std::int32_t, std::int64_t>;
using offset_t = std::int32_t;

// The last element must not be read in an exclusive scan, to be confirmed by compute-sanitizer initcheck
CUB_TEST("Device exclusive scan ignores last input element", "[scan][device]", CUB_SMALL, types)
{
  using type = c2h::get<0, TestType>;
  using op_t = ::cuda::std::plus<>;

  const offset_t initialized_size =
    GENERATE_COPY(values({0, 1, 2, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097}), take(3, random(1, 1'000'000)));
  const offset_t size = initialized_size + 1;
  CAPTURE(size, c2h::type_name<type>());

  c2h::device_vector<type> input(initialized_size);
  c2h::gen(C2H_SEED(1), input);
  // Initialize all but the last value of the device vector
  c2h::device_vector<type> data(size, thrust::no_init);
  thrust::copy(c2h::device_policy, input.cbegin(), input.cend(), data.begin());
  auto d_data = thrust::raw_pointer_cast(data.data());

  c2h::host_vector<type> host_input(input);
  host_input.push_back(type{}); // the last value doesn't matter here either
  c2h::host_vector<type> expected(size);

  type init_value{};
  compute_exclusive_scan_reference(host_input.cbegin(), host_input.cend(), expected.begin(), init_value, op_t{});

  device_exclusive_scan(d_data, d_data, op_t{}, init_value, size);

  REQUIRE_THAT_QUIET(expected, Equals(data));
}

constexpr int valid_value   = 1;
constexpr int invalid_value = 2;

// an operator that only expects the value 1 and only ever returns 1
// and records if it ever was passed any other value.
struct checking_op
{
  template <typename T>
  __device__ T operator()(T lhs, T rhs) const
  {
    if (lhs != valid_value || rhs != valid_value)
    {
      *invalid = true;
    }
    return valid_value;
  }
  bool* invalid;
};

// The lookahead scan algorithm is only compiled when these hold, see `can_use_lookahead` in
// cub/device/dispatch/tuning/tuning_scan.cuh. Requesting it through the tuning environment bypasses that check, so we
// have to mirror the conditions here to avoid instantiating a kernel that cannot be built.
#if __cccl_ptx_isa >= 860 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) \
  && !(_CCCL_COMPILER(MSVC) && _CCCL_CUDA_COMPILER(NVCC, <, 13, 1)) && !defined(CCCL_DISABLE_WARPSPEED_SCAN)
#  define TEST_HAS_LOOKAHEAD_SCAN 1
#else
#  define TEST_HAS_LOOKAHEAD_SCAN 0
#endif

//! Requests the lookahead scan algorithm wherever it is supported, with a tuning small enough to fit into the
//! architecture independent shared memory limit.
//!
//! The default tuning picks lookback for this test: `checking_op` is a user-defined operator, for which the sm_120
//! tuning raises `items_per_thread` to 127. The resulting tile no longer fits into 48 KiB, so `can_use_lookahead`
//! rejects lookahead and silently falls back. Without this tuning, the test would never exercise lookahead on sm_120.
template <typename T>
struct lookahead_tuning
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> cub::ScanPolicy
  {
    // The policy is instantiated for every compiled architecture, so fall back to the default tuning wherever
    // lookahead is unavailable.
    const bool supported =
      TEST_HAS_LOOKAHEAD_SCAN && cc >= ::cuda::compute_capability{9, 0} // codegen bug in nvcc below 13.4 on GB20x, see
                                                                        // https://github.com/NVIDIA/cccl/issues/8528
      && !(_CCCL_CUDACC_BELOW(13, 4) && cc == ::cuda::compute_capability{12, 0});
    if (!supported)
    {
      return cub::detail::scan::
        policy_selector_from_types<T*, T*, T, cub::detail::choose_offset_t<offset_t>, checking_op>{}(cc);
    }

    // 256 / sizeof(T) - 1 items per thread is what the sm_100 tuning uses and keeps us within 48 KiB.
    // Aggregate init instead of designated init, because the test is also compiled as C++17.
    return cub::ScanPolicy{
      cub::ScanAlgorithm::lookahead,
      {},
      cub::ScanLookaheadPolicy{/* reduce_and_scan_warps */ 4,
                               /* items_per_thread */ 256 / int{sizeof(T)} - 1,
                               /* lookahead_items_per_thread */ 4}};
  }
};

template <typename T, typename EnvT>
void check_scan_op_receives_only_valid_values(offset_t size, const EnvT& env)
{
  constexpr offset_t padding = 100;

  c2h::device_vector<T> data(size, valid_value);
  data.resize(size + padding, invalid_value);

  c2h::host_vector<T> expected(size + 1, valid_value);
  expected.resize(size + padding, invalid_value);

  c2h::device_vector<bool> accessed(1, false);
  auto d_data = thrust::raw_pointer_cast(data.data());
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(
            d_data, d_data, checking_op{accessed.data().get()}, static_cast<T>(valid_value), size + 1, env));

  REQUIRE_THAT_QUIET(expected, Equals(data));
  REQUIRE(accessed.front() == false);
}

CUB_TEST("Device exclusive scan operator only receives valid values", "[scan][device]", CUB_SMALL, types)
{
  using type = c2h::get<0, TestType>;

  // 1 << 19 spans more than the 32 tiles a lookahead step covers for all tunings used here, so the lookahead warp has
  // to combine tile aggregates across several steps, including partially filled windows of tile states
  const offset_t size = GENERATE_COPY(
    values({0, 1, 2, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097, 1 << 19}), take(3, random(1, 1'000'000)));
  CAPTURE(size, c2h::type_name<type>());

  SECTION("default tuning")
  {
    check_scan_op_receives_only_valid_values<type>(size, ::cuda::std::execution::env<>{});
  }

  SECTION("lookahead tuning")
  {
    check_scan_op_receives_only_valid_values<type>(size, ::cuda::execution::tune(lookahead_tuning<type>{}));
  }
}
