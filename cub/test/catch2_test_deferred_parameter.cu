// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% TEST_ASSERTIONS assertions 0:1

#if !TEST_ASSERTIONS
#  undef CCCL_ENABLE_ASSERTIONS
#  undef CCCL_ENABLE_HOST_ASSERTIONS
#  undef CCCL_ENABLE_DEVICE_ASSERTIONS
#endif

#include <cub/detail/deferred_parameter.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda/argument>
#include <cuda/execution>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

using count_types = c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t, cuda::std::int64_t, cuda::std::uint64_t>;

template <typename OffsetT, typename KernelArgT>
_CCCL_KERNEL_ATTRIBUTES void read_num_items(KernelArgT num_items, OffsetT* result)
{
  *result = cub::detail::parameter_from_device<OffsetT>(num_items);
}

CUB_TEST(
  "Problem-size conversion preserves nonnegative immediate values", "[deferred_parameter]", CUB_SMALL, count_types)
{
  using count_t       = c2h::get<0, TestType>;
  using offset_t      = cub::detail::num_items_offset_t<count_t>;
  const count_t count = GENERATE_COPY(count_t{0}, count_t{1}, cuda::std::numeric_limits<count_t>::max());
  const auto expected = static_cast<offset_t>(count);

  REQUIRE(cub::detail::make_num_items_dispatch_arg(count) == expected);
  REQUIRE(cub::detail::make_num_items_dispatch_arg(cuda::args::immediate{count}) == expected);
  REQUIRE(cub::detail::make_num_items_kernel_arg(count) == expected);
  REQUIRE(cub::detail::make_num_items_kernel_arg(cuda::args::immediate{count}) == expected);
  REQUIRE(cub::detail::num_items_upper_bound(count) == expected);
  REQUIRE(cub::detail::num_items_upper_bound(cuda::args::immediate{count}) == expected);
}

CUB_TEST("Deferred problem-size conversion does not read the source on the host",
         "[deferred_parameter]",
         CUB_SMALL,
         count_types)
{
  using count_t    = c2h::get<0, TestType>;
  const auto count = cuda::args::deferred{
    static_cast<count_t*>(nullptr),
    cuda::args::bounds<count_t{0}, count_t{20}>(),
    cuda::args::bounds(count_t{10}, count_t{30})};
  using offset_t        = cub::detail::num_items_offset_t<decltype(count)>;
  const auto kernel_arg = cub::detail::make_num_items_kernel_arg(count);

#if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
  static_assert(cuda::args::__is_deferred_v<cuda::std::remove_cv_t<decltype(kernel_arg)>>);
  REQUIRE(cuda::args::__unwrap(kernel_arg) == nullptr);
  REQUIRE(cuda::args::__lowest_(kernel_arg) == count_t{10});
  REQUIRE(cuda::args::__highest_(kernel_arg) == count_t{20});
#else
  static_assert(cuda::std::is_same_v<decltype(kernel_arg), count_t* const>);
  REQUIRE(kernel_arg == nullptr);
#endif
  REQUIRE(cub::detail::num_items_upper_bound(count) == offset_t{20});
}

CUB_TEST("Deferred problem sizes accept the effective bound endpoints on device",
         "[deferred_parameter]",
         CUB_SMALL,
         count_types)
{
  using count_t       = c2h::get<0, TestType>;
  using offset_t      = cub::detail::num_items_offset_t<count_t>;
  const count_t count = GENERATE_COPY(count_t{10}, count_t{20});
  c2h::device_vector<count_t> device_count(1, count);
  c2h::device_vector<offset_t> device_result(1);
  const auto arg = cuda::args::deferred{
    device_count.data().get(),
    cuda::args::bounds<count_t{0}, count_t{20}>(),
    cuda::args::bounds(count_t{10}, count_t{30})};

  read_num_items<<<1, 1>>>(cub::detail::make_num_items_kernel_arg(arg), device_result.data().get());
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  const c2h::host_vector<offset_t> result = device_result;
  REQUIRE(result[0] == static_cast<offset_t>(count));
}

struct fixed_lookback_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const -> cub::ScanPolicy
  {
    using value_t    = cuda::std::int32_t;
    using selector_t = cub::detail::scan::
      policy_selector_from_types<const value_t*, value_t*, value_t, cuda::std::uint32_t, cuda::std::plus<>>;
    auto policy                       = selector_t{}(cc);
    policy.algorithm                  = cub::ScanAlgorithm::lookback;
    policy.lookback.threads_per_block = 128;
    policy.lookback.items_per_thread  = 1;
    return policy;
  }
};

CUB_TEST("Bounded scans skip tiles beyond the actual deferred count", "[device][scan][deferred]", CUB_SMALL)
{
  using value_t              = cuda::std::int32_t;
  constexpr value_t capacity = 4096;
  const value_t count =
    GENERATE_COPY(value_t{0}, value_t{1}, value_t{127}, value_t{128}, value_t{129}, value_t{1024}, capacity);
  const c2h::device_vector<value_t> input(capacity, value_t{1});
  c2h::device_vector<value_t> device_count(1, count);
  c2h::device_vector<value_t> deferred_result(capacity, value_t{-1});
  c2h::device_vector<value_t> immediate_result(capacity, value_t{-1});
  c2h::host_vector<value_t> expected(capacity, value_t{-1});
  for (value_t i = 0; i < count; ++i)
  {
    expected[i] = i;
  }

  const auto arg = cuda::args::deferred{device_count.data().get(), cuda::args::bounds<value_t{0}, capacity>()};
  const auto env = cuda::execution::tune(fixed_lookback_policy_selector{});
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), deferred_result.begin(), arg, env));
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), immediate_result.begin(), count, env));
  REQUIRE(expected == deferred_result);
  REQUIRE(expected == immediate_result);
}
