// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/detail/deferred_parameter.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/fill.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/argument>
#include <cuda/execution>
#include <cuda/iterator>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstdint>

#include <cuda_runtime_api.h>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/generators.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSum, device_exclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSum, device_inclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

using deferred_count_t = decltype(cuda::args::deferred{static_cast<int32_t*>(nullptr)});
static_assert(cub::detail::is_deferred_v<deferred_count_t>);
static_assert(!cub::detail::is_deferred_v<int32_t>);
static_assert(cub::detail::is_num_items_v<deferred_count_t>);
static_assert(cub::detail::is_num_items_v<int32_t>);
static_assert(cuda::std::is_same_v<cub::detail::num_items_offset_t<deferred_count_t>, uint32_t>);

using value_t = int32_t;
using count_t = int32_t;

template <typename InputT, typename OutputT, typename AccumT, typename ScanOpT>
struct small_batch_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const -> cub::ScanPolicy
  {
    using selector_t =
      cub::detail::scan::policy_selector_from_types<const InputT*, OutputT*, AccumT, uint64_t, ScanOpT>;
    auto policy                       = selector_t{}(cc);
    policy.algorithm                  = cub::ScanAlgorithm::lookback;
    policy.lookback.threads_per_block = 128;
    policy.lookback.items_per_thread  = 1;
    return policy;
  }
};

template <typename InputT, typename OutputT, typename AccumT, typename ScanOpT>
struct lookahead_preferred_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const -> cub::ScanPolicy
  {
    auto policy      = small_batch_policy_selector<InputT, OutputT, AccumT, ScanOpT>{}(cc);
    policy.algorithm = cub::ScanAlgorithm::lookahead;
    policy.lookahead = cub::ScanLookaheadPolicy{4, 79, 3};
    return policy;
  }
};

using lookahead_preferred_int_policy =
  lookahead_preferred_policy_selector<value_t, value_t, value_t, cuda::std::plus<>>;
static_assert(lookahead_preferred_int_policy{}(cuda::compute_capability{10, 0}).algorithm
              == cub::ScanAlgorithm::lookahead);
static_assert(
  cub::detail::scan::deferred_policy_selector<lookahead_preferred_int_policy>{}(cuda::compute_capability{10, 0}).algorithm
  == cub::ScanAlgorithm::lookback);

struct is_even_t
{
  _CCCL_DEVICE_API bool operator()(value_t value) const
  {
    return value % 2 == 0;
  }
};

struct less_than_t
{
  value_t bound;

  _CCCL_HOST_DEVICE_API bool operator()(value_t value) const
  {
    return value < bound;
  }
};

struct affine_value_t
{
  uint32_t scale;
  uint32_t shift;
};

struct compose_affine_t
{
  _CCCL_HOST_DEVICE_API affine_value_t operator()(affine_value_t lhs, affine_value_t rhs) const
  {
    return {rhs.scale * lhs.scale, rhs.scale * lhs.shift + rhs.shift};
  }
};

struct select_then_scan_t
{
  template <typename InputIteratorT,
            typename SelectedOutputIteratorT,
            typename NumSelectedIteratorT,
            typename ScanOutputIteratorT,
            typename NumItemsT,
            typename SelectOpT>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    uint8_t* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanOutputIteratorT d_scan_out,
    NumItemsT num_items,
    SelectOpT select_op,
    cudaStream_t stream = nullptr) const
  {
    size_t select_temp_storage_bytes{};
    if (const cudaError_t error = cub::DeviceSelect::If(
          nullptr, select_temp_storage_bytes, d_in, d_selected_out, d_num_selected_out, num_items, select_op, stream);
        error != cudaSuccess)
    {
      return error;
    }

    const auto deferred_num_items = cuda::args::deferred{d_num_selected_out};
    size_t scan_temp_storage_bytes{};
    if (const cudaError_t error = cub::DeviceScan::ExclusiveSum(
          nullptr, scan_temp_storage_bytes, d_selected_out, d_scan_out, deferred_num_items, stream);
        error != cudaSuccess)
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = cuda::std::max(select_temp_storage_bytes, scan_temp_storage_bytes);
      return cudaSuccess;
    }

    if (const cudaError_t error = cub::DeviceSelect::If(
          d_temp_storage,
          select_temp_storage_bytes,
          d_in,
          d_selected_out,
          d_num_selected_out,
          num_items,
          select_op,
          stream);
        error != cudaSuccess)
    {
      return error;
    }

    return cub::DeviceScan::ExclusiveSum(
      d_temp_storage, scan_temp_storage_bytes, d_selected_out, d_scan_out, deferred_num_items, stream);
  }
};

static c2h::host_vector<value_t> reference_exclusive_sum(const c2h::host_vector<value_t>& values, count_t num_items)
{
  c2h::host_vector<value_t> result(static_cast<size_t>(num_items));
  value_t running{};
  for (count_t i = 0; i < num_items; ++i)
  {
    result[i] = running;
    running += values[i];
  }
  return result;
}

CUB_TEST("DeviceScan::ExclusiveSum consumes a count produced by DeviceSelect::If", "[device][scan][deferred]", CUB_SMALL)
{
  constexpr count_t num_items = 100'000;

  c2h::device_vector<value_t> input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), input, value_t{0}, value_t{9});

  c2h::device_vector<value_t> selected_items(num_items, value_t{-1});
  c2h::device_vector<count_t> device_num_selected(1, count_t{-1});
  c2h::device_vector<value_t> output(num_items, value_t{-1});

  const auto d_input          = thrust::raw_pointer_cast(input.data());
  const auto d_selected_items = thrust::raw_pointer_cast(selected_items.data());
  const auto d_num_selected   = thrust::raw_pointer_cast(device_num_selected.data());
  const auto d_output         = thrust::raw_pointer_cast(output.data());

  launch(select_then_scan_t{}, d_input, d_selected_items, d_num_selected, d_output, num_items, is_even_t{});

  const count_t selected_count = device_num_selected[0];
  REQUIRE(selected_count >= count_t{0});
  REQUIRE(selected_count <= num_items);

  const c2h::host_vector<value_t> host_selected = selected_items;
  const c2h::host_vector<value_t> host_output   = output;
  const auto expected                           = reference_exclusive_sum(host_selected, selected_count);

  for (count_t i = 0; i < selected_count; ++i)
  {
    REQUIRE(host_output[i] == expected[i]);
  }

  if (selected_count < num_items)
  {
    REQUIRE(host_output[selected_count] == value_t{-1});
  }
}

CUB_TEST("DeviceScan entry points accept deferred num_items", "[device][scan][deferred]", CUB_SMALL)
{
  constexpr count_t capacity = 10'000;
  const count_t num_items    = GENERATE_COPY(count_t{0}, count_t{1}, count_t{2}, count_t{1'000}, capacity);
  CAPTURE(num_items);

  const c2h::device_vector<value_t> input(capacity, value_t{2});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};

  device_exclusive_sum(input.begin(), output.begin(), count);
  for (count_t i = 0; i < num_items; ++i)
  {
    REQUIRE(output[i] == value_t{2} * i);
  }

  thrust::fill(output.begin(), output.end(), value_t{-1});
  device_inclusive_sum(input.begin(), output.begin(), count);
  for (count_t i = 0; i < num_items; ++i)
  {
    REQUIRE(output[i] == value_t{2} * (i + 1));
  }

  thrust::fill(output.begin(), output.end(), value_t{-1});
  device_exclusive_scan(input.begin(), output.begin(), cuda::std::plus<>{}, value_t{7}, count);
  for (count_t i = 0; i < num_items; ++i)
  {
    REQUIRE(output[i] == value_t{7} + value_t{2} * i);
  }

  thrust::fill(output.begin(), output.end(), value_t{-1});
  device_inclusive_scan(input.begin(), output.begin(), cuda::std::plus<>{}, count);
  for (count_t i = 0; i < num_items; ++i)
  {
    REQUIRE(output[i] == value_t{2} * (i + 1));
  }
}

// The in-place entry points take num_items where the out-of-place ones take an output iterator, so they are
// separated by a constraint on the trailing environment parameter. A deferred count has to land on the in-place
// overload here and on the out-of-place one when an output iterator is present.
CUB_TEST("DeviceScan in-place entry points accept deferred num_items", "[device][scan][deferred]", CUB_SMALL)
{
  constexpr count_t capacity = 10'000;
  const count_t num_items    = GENERATE_COPY(count_t{0}, count_t{1}, count_t{1'000}, capacity);
  CAPTURE(num_items);

  const c2h::device_vector<count_t> device_num_items(1, num_items);
  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};

  // Items past num_items must be left untouched, which also proves the deferred count reached the kernel.
  const auto check = [&](const c2h::device_vector<value_t>& data, auto expected_at) {
    const c2h::host_vector<value_t> host = data;
    for (count_t i = 0; i < num_items; ++i)
    {
      REQUIRE(host[i] == expected_at(i));
    }
    if (num_items < capacity)
    {
      REQUIRE(host[num_items] == value_t{2});
    }
  };

  c2h::device_vector<value_t> data(capacity, value_t{2});
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(data.begin(), count));
  check(data, [](count_t i) {
    return value_t{2} * i;
  });

  thrust::fill(data.begin(), data.end(), value_t{2});
  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(data.begin(), count));
  check(data, [](count_t i) {
    return value_t{2} * (i + 1);
  });

  thrust::fill(data.begin(), data.end(), value_t{2});
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(data.begin(), cuda::std::plus<>{}, value_t{7}, count));
  check(data, [](count_t i) {
    return value_t{7} + value_t{2} * i;
  });

  thrust::fill(data.begin(), data.end(), value_t{2});
  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScan(data.begin(), cuda::std::plus<>{}, count));
  check(data, [](count_t i) {
    return value_t{2} * (i + 1);
  });
}

// InclusiveScanInit accepts a deferred count both with an immediate initial value and with a deferred one, i.e. with
// the problem size and the seed each read from device memory.
CUB_TEST("DeviceScan::InclusiveScanInit accepts a deferred num_items", "[device][scan][deferred]", CUB_SMALL)
{
  constexpr count_t capacity = 10'000;
  const count_t num_items    = GENERATE_COPY(count_t{0}, count_t{1}, count_t{1'000}, capacity);
  CAPTURE(num_items);

  constexpr value_t init = 7;
  const c2h::device_vector<value_t> input(capacity, value_t{2});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  const c2h::device_vector<value_t> device_init(1, init);
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};

  const auto check = [&] {
    const c2h::host_vector<value_t> host = output;
    for (count_t i = 0; i < num_items; ++i)
    {
      REQUIRE(host[i] == init + value_t{2} * (i + 1));
    }
    if (num_items < capacity)
    {
      REQUIRE(host[num_items] == value_t{-1});
    }
  };

  REQUIRE(
    cudaSuccess == cub::DeviceScan::InclusiveScanInit(input.begin(), output.begin(), cuda::std::plus<>{}, init, count));
  check();

  thrust::fill(output.begin(), output.end(), value_t{-1});
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScanInit(
            input.begin(),
            output.begin(),
            cuda::std::plus<>{},
            cuda::args::deferred{thrust::raw_pointer_cast(device_init.data())},
            count));
  check();
}

CUB_TEST("DeviceScan with a deferred size forces a lookahead policy to lookback", "[device][scan][deferred]", CUB_SMALL)
{
  constexpr count_t capacity  = 10'000;
  constexpr count_t num_items = 1'000;

  const c2h::device_vector<value_t> input(capacity, value_t{1});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};
  const auto env   = cuda::execution::tune(lookahead_preferred_int_policy{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), output.begin(), count, env));
  for (count_t i = 0; i < num_items; ++i)
  {
    REQUIRE(output[i] == i);
  }
}

// The tests below drive cub::DeviceScan directly rather than through the launch helper, either because they
// need explicit streams, graph capture or repeated invocations. Running them once, under the direct-launch
// variant, avoids rebuilding identical work for every TEST_LAUNCH value.
#if TEST_LAUNCH == 0
// Smoke-tests the batched kernel under the default tuning, where one batch covers the whole input. Batch
// transitions are covered separately by the tests using small_batch_policy_selector.
CUB_TEST("DeviceScan::ExclusiveSum uses batched lookback within a single batch for an unbounded deferred int64 count",
         "[device][scan][deferred][batch]",
         CUB_SMALL)
{
  using large_count_t               = int64_t;
  constexpr large_count_t num_items = 5'000'000;

  const c2h::device_vector<value_t> input(static_cast<size_t>(num_items), value_t{1});
  const c2h::device_vector<large_count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> expected(static_cast<size_t>(num_items), value_t{-1});
  c2h::device_vector<value_t> output(static_cast<size_t>(num_items), value_t{-1});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), expected.begin(), num_items));
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            input.begin(), output.begin(), cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())}));
  REQUIRE(output == expected);
}

CUB_TEST("DeviceScan reusable lookback batches handle exact boundaries", "[device][scan][deferred][batch]", CUB_SMALL)
{
  using large_count_t                 = int64_t;
  constexpr large_count_t batch_items = static_cast<large_count_t>(cub::detail::scan::tiles_per_batch) * 128;
  const large_count_t num_items       = GENERATE_COPY(
    large_count_t{0}, large_count_t{1}, batch_items - 1, batch_items, batch_items + 1, 2 * batch_items + 17);
  CAPTURE(num_items);

  const c2h::device_vector<value_t> input(static_cast<size_t>(num_items), value_t{1});
  const c2h::device_vector<large_count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> expected(static_cast<size_t>(num_items), value_t{-1});
  c2h::device_vector<value_t> output(static_cast<size_t>(num_items), value_t{-1});
  const auto env = cuda::execution::tune(small_batch_policy_selector<value_t, value_t, value_t, cuda::std::plus<>>{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), expected.begin(), num_items, env));
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      input.begin(), output.begin(), cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())}, env));
  REQUIRE(output == expected);

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(input.begin(), expected.begin(), num_items, env));
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSum(
      input.begin(), output.begin(), cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())}, env));
  REQUIRE(output == expected);

  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(
            input.begin(), expected.begin(), cuda::std::plus<>{}, value_t{7}, num_items, env));
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      input.begin(),
      output.begin(),
      cuda::std::plus<>{},
      value_t{7},
      cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())},
      env));
  REQUIRE(output == expected);

  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScan(input.begin(), expected.begin(), cuda::std::plus<>{}, num_items, env));
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScan(
            input.begin(),
            output.begin(),
            cuda::std::plus<>{},
            cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())},
            env));
  REQUIRE(output == expected);
}

CUB_TEST("DeviceScan batched lookback preserves noncommutative order", "[device][scan][deferred][batch]", CUB_SMALL)
{
  using large_count_t               = int64_t;
  constexpr large_count_t num_items = static_cast<large_count_t>(cub::detail::scan::tiles_per_batch) * 128 + 17;

  c2h::host_vector<affine_value_t> host_input(static_cast<size_t>(num_items));
  for (large_count_t i = 0; i < num_items; ++i)
  {
    host_input[static_cast<size_t>(i)] = affine_value_t{static_cast<uint32_t>(i % 3 + 1), static_cast<uint32_t>(i % 5)};
  }
  const c2h::device_vector<affine_value_t> input = host_input;
  const c2h::device_vector<large_count_t> device_num_items(1, num_items);
  c2h::device_vector<affine_value_t> expected(static_cast<size_t>(num_items));
  c2h::device_vector<affine_value_t> output(static_cast<size_t>(num_items));
  const auto env = cuda::execution::tune(
    small_batch_policy_selector<affine_value_t, affine_value_t, affine_value_t, compose_affine_t>{});

  REQUIRE(
    cudaSuccess == cub::DeviceScan::InclusiveScan(input.begin(), expected.begin(), compose_affine_t{}, num_items, env));
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScan(
            input.begin(),
            output.begin(),
            compose_affine_t{},
            cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())},
            env));

  const c2h::host_vector<affine_value_t> host_expected = expected;
  const c2h::host_vector<affine_value_t> host_output   = output;
  bool equal                                           = true;
  for (large_count_t i = 0; i < num_items; ++i)
  {
    const auto index = static_cast<size_t>(i);
    if (host_expected[index].scale != host_output[index].scale
        || host_expected[index].shift != host_output[index].shift)
    {
      CAPTURE(
        i, host_expected[index].scale, host_output[index].scale, host_expected[index].shift, host_output[index].shift);
      equal = false;
      break;
    }
  }
  REQUIRE(equal);
}

// Two persistent grids share the device while each spans several batches, so CTAs of one scan wait for a batch
// handover while the other scan competes for SMs. The small tile forces the batch transitions; with the default
// tuning a single batch would cover the whole input and no CTA would ever wait.
CUB_TEST("DeviceScan batched lookback makes progress on concurrent streams",
         "[device][scan][deferred][batch]",
         CUB_SMALL)
{
  using large_count_t                 = int64_t;
  constexpr large_count_t batch_items = static_cast<large_count_t>(cub::detail::scan::tiles_per_batch) * 128;
  constexpr large_count_t num_items   = 2 * batch_items + 12'345;

  const c2h::device_vector<value_t> input(static_cast<size_t>(num_items), value_t{1});
  const c2h::device_vector<large_count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> first(static_cast<size_t>(num_items), value_t{-1});
  c2h::device_vector<value_t> second(static_cast<size_t>(num_items), value_t{-1});
  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};

  cudaStream_t first_stream{};
  cudaStream_t second_stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&first_stream));
  REQUIRE(cudaSuccess == cudaStreamCreate(&second_stream));

  const auto tuning =
    cuda::execution::tune(small_batch_policy_selector<value_t, value_t, value_t, cuda::std::plus<>>{});
  const auto first_env  = cuda::std::execution::env{cuda::stream_ref{first_stream}, tuning};
  const auto second_env = cuda::std::execution::env{cuda::stream_ref{second_stream}, tuning};

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), first.begin(), count, first_env));
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), second.begin(), count, second_env));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(first_stream));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(second_stream));
  REQUIRE(cudaSuccess == cudaStreamDestroy(first_stream));
  REQUIRE(cudaSuccess == cudaStreamDestroy(second_stream));

  REQUIRE(first.front() == value_t{0});
  REQUIRE(first.back() == static_cast<value_t>(num_items - 1));
  REQUIRE(second.front() == value_t{0});
  REQUIRE(second.back() == static_cast<value_t>(num_items - 1));
}

CUB_TEST("DeviceScan::ExclusiveSum consumes a deferred count produced in another stream after an event",
         "[device][scan][deferred]",
         CUB_SMALL)
{
  constexpr count_t capacity       = 100'000;
  constexpr count_t selected_count = 1'000;

  cudaStream_t producer{};
  cudaStream_t consumer{};
  cudaEvent_t count_ready{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&producer));
  REQUIRE(cudaSuccess == cudaStreamCreate(&consumer));
  REQUIRE(cudaSuccess == cudaEventCreate(&count_ready));

  const auto input = cuda::counting_iterator<value_t>{value_t{0}};
  c2h::device_vector<value_t> selected_items(capacity, value_t{-1});
  c2h::device_vector<count_t> device_num_selected(1, count_t{-1});
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto d_selected_items = thrust::raw_pointer_cast(selected_items.data());
  const auto d_num_selected   = thrust::raw_pointer_cast(device_num_selected.data());
  const auto d_output         = thrust::raw_pointer_cast(output.data());
  const auto num_selected     = cuda::args::deferred{
    d_num_selected, cuda::args::bounds<count_t{0}, capacity>(), cuda::args::bounds(count_t{0}, capacity)};

  size_t select_temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      nullptr,
      select_temp_storage_bytes,
      input,
      d_selected_items,
      d_num_selected,
      capacity,
      less_than_t{selected_count},
      producer));

  size_t scan_temp_storage_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            nullptr, scan_temp_storage_bytes, d_selected_items, d_output, num_selected, consumer));

  const size_t temp_storage_bytes = cuda::std::max(select_temp_storage_bytes, scan_temp_storage_bytes);
  c2h::device_vector<uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      d_temp_storage,
      select_temp_storage_bytes,
      input,
      d_selected_items,
      d_num_selected,
      capacity,
      less_than_t{selected_count},
      producer));
  REQUIRE(cudaSuccess == cudaEventRecord(count_ready, producer));
  REQUIRE(cudaSuccess == cudaStreamWaitEvent(consumer, count_ready, 0));

  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            d_temp_storage, scan_temp_storage_bytes, d_selected_items, d_output, num_selected, consumer));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(consumer));

  REQUIRE(device_num_selected[0] == selected_count);
  for (count_t i = 0; i < selected_count; ++i)
  {
    REQUIRE(output[i] == i * (i - 1) / 2);
  }
  REQUIRE(output[selected_count] == value_t{-1});

  REQUIRE(cudaSuccess == cudaEventDestroy(count_ready));
  REQUIRE(cudaSuccess == cudaStreamDestroy(producer));
  REQUIRE(cudaSuccess == cudaStreamDestroy(consumer));
}
#endif // TEST_LAUNCH == 0

CUB_TEST("DeviceScan::ExclusiveSum with a deferred size handles tile boundaries", "[device][scan][deferred]", CUB_SMALL)
{
  constexpr count_t capacity = 200'000;
  const count_t num_items =
    GENERATE_COPY(count_t{0}, count_t{1}, count_t{127}, count_t{8'447}, count_t{8'448}, count_t{8'449}, capacity);
  CAPTURE(num_items);

  const c2h::device_vector<value_t> input(capacity, value_t{1});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};
  device_exclusive_sum(input.begin(), output.begin(), count);

  const c2h::host_vector<value_t> host_output = output;
  for (count_t i = 0; i < num_items; ++i)
  {
    REQUIRE(host_output[i] == i);
  }
  if (num_items < capacity)
  {
    REQUIRE(host_output[num_items] == value_t{-1});
  }
}

CUB_TEST("DeviceScan::ExclusiveSum accepts deferred span, iterator and bounded sources",
         "[device][scan][deferred]",
         CUB_SMALL)
{
  constexpr count_t capacity  = 50'000;
  constexpr count_t num_items = 1'000;

  const c2h::device_vector<value_t> input(capacity, value_t{1});
  c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto d_num_items = thrust::raw_pointer_cast(device_num_items.data());

  const auto check = [&] {
    const c2h::host_vector<value_t> host_output = output;
    for (count_t i = 0; i < num_items; ++i)
    {
      REQUIRE(host_output[i] == i);
    }
  };

  const auto const_count_span = cuda::std::span<const count_t, 1>{d_num_items, 1};
  device_exclusive_sum(input.begin(), output.begin(), cuda::args::deferred{const_count_span});
  check();

  thrust::fill(output.begin(), output.end(), value_t{-1});
  const auto count_transform = cuda::transform_iterator(d_num_items, cuda::std::identity{});
  device_exclusive_sum(input.begin(), output.begin(), cuda::args::deferred{count_transform});
  check();

  thrust::fill(output.begin(), output.end(), value_t{-1});
  const auto bounded_count = cuda::args::deferred{
    d_num_items, cuda::args::bounds<count_t{0}, capacity>(), cuda::args::bounds(count_t{0}, capacity)};
  device_exclusive_sum(input.begin(), output.begin(), bounded_count);
  check();
}

// The tests below drive cub::DeviceScan directly rather than through the launch helper, either because they
// need explicit streams, graph capture or repeated invocations. Running them once, under the direct-launch
// variant, avoids rebuilding identical work for every TEST_LAUNCH value.
#if TEST_LAUNCH == 0
CUB_TEST("DeviceScan::ExclusiveSum with a deferred size returns the same result on repeated invocations",
         "[device][scan][deferred]",
         CUB_SMALL)
{
  constexpr count_t capacity  = 100'000;
  constexpr count_t num_items = 60'000;

  c2h::device_vector<value_t> input(capacity, thrust::no_init);
  c2h::gen(C2H_SEED(1), input, value_t{0}, value_t{9});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> first(capacity, value_t{-1});
  c2h::device_vector<value_t> second(capacity, value_t{-1});

  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), first.begin(), count));
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), second.begin(), count));
  REQUIRE(first == second);
  REQUIRE(first[num_items] == value_t{-1});
}

CUB_TEST("DeviceScan::ExclusiveSum with a deferred size matches the immediate result",
         "[device][scan][deferred]",
         CUB_SMALL)
{
  constexpr count_t capacity = 100'000;
  const count_t num_items    = GENERATE_COPY(count_t{0}, count_t{1}, count_t{8'448}, count_t{60'000}, capacity);
  CAPTURE(num_items);

  c2h::device_vector<value_t> input(capacity, thrust::no_init);
  c2h::gen(C2H_SEED(1), input, value_t{0}, value_t{9});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> reference(capacity, value_t{-1});
  c2h::device_vector<value_t> deferred(capacity, value_t{-1});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), reference.begin(), num_items));
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            input.begin(), deferred.begin(), cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())}));
  REQUIRE(reference == deferred);
}

// A deferred size is compatible with every determinism guarantee. Integral types with a known operator are exactly
// associative, so the guarantee holds however tiles are grouped into batches; fp plus engages the stable look-back
// order, whose 32-tile anchor windows are indexed within a batch and so line up with the batch seeding.
CUB_TEST("DeviceScan::ExclusiveSum with a deferred size accepts run_to_run and gpu_to_gpu determinism",
         "[device][scan][deferred][determinism]",
         CUB_SMALL)
{
  using large_count_t                 = int64_t;
  constexpr large_count_t batch_items = static_cast<large_count_t>(cub::detail::scan::tiles_per_batch) * 128;
  const large_count_t num_items       = GENERATE_COPY(large_count_t{30'000}, batch_items + 4'567);
  CAPTURE(num_items);

  const c2h::device_vector<value_t> input(static_cast<size_t>(num_items), value_t{1});
  const c2h::device_vector<large_count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(static_cast<size_t>(num_items), value_t{-1});
  const auto count  = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};
  const auto tuning = small_batch_policy_selector<value_t, value_t, value_t, cuda::std::plus<>>{};

  // Exactly associative, so the result must match the closed form regardless of the batch grouping.
  const auto check_exact = [&] {
    const c2h::host_vector<value_t> host_output = output;
    for (large_count_t i = 0; i < num_items; ++i)
    {
      if (host_output[static_cast<size_t>(i)] != static_cast<value_t>(i))
      {
        CAPTURE(i, host_output[static_cast<size_t>(i)]);
        return false;
      }
    }
    return true;
  };

  const auto run_to_run = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::run_to_run), cuda::execution::tune(tuning)};
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), output.begin(), count, run_to_run));
  REQUIRE(check_exact());

  thrust::fill(output.begin(), output.end(), value_t{-1});
  const auto gpu_to_gpu = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::gpu_to_gpu), cuda::execution::tune(tuning)};
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), output.begin(), count, gpu_to_gpu));
  REQUIRE(check_exact());
}

// run_to_run over floating-point plus engages the stable look-back order. Repeated launches of the same
// configuration must agree bit for bit, including across batch boundaries.
CUB_TEST("DeviceScan::ExclusiveSum with a deferred size is bitwise reproducible under run_to_run",
         "[device][scan][deferred][determinism]",
         CUB_SMALL)
{
  using large_count_t                 = int64_t;
  constexpr large_count_t batch_items = static_cast<large_count_t>(cub::detail::scan::tiles_per_batch) * 128;
  constexpr large_count_t num_items   = batch_items + 4'567;

  c2h::device_vector<float> input(static_cast<size_t>(num_items), thrust::no_init);
  c2h::gen(C2H_SEED(1), input, -1.0e5f, 1.0e5f);
  const c2h::device_vector<large_count_t> device_num_items(1, num_items);
  c2h::device_vector<float> first(static_cast<size_t>(num_items));
  c2h::device_vector<float> repeat(static_cast<size_t>(num_items));

  const auto count = cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())};
  const auto env   = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::run_to_run),
    cuda::execution::tune(small_batch_policy_selector<float, float, float, cuda::std::plus<>>{})};

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), first.begin(), count, env));
  for (int i = 0; i < 3; ++i)
  {
    REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(input.begin(), repeat.begin(), count, env));
    REQUIRE(repeat == first);
  }
}

CUB_TEST("DeviceScan::ExclusiveSum with a deferred size supports not_guaranteed determinism",
         "[device][scan][deferred]",
         CUB_SMALL)
{
  constexpr count_t capacity  = 50'000;
  constexpr count_t num_items = 30'000;

  const c2h::device_vector<value_t> input(capacity, value_t{1});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      input.begin(), output.begin(), cuda::args::deferred{thrust::raw_pointer_cast(device_num_items.data())}, env));

  const c2h::host_vector<value_t> host_output = output;
  for (count_t i = 0; i < num_items; ++i)
  {
    REQUIRE(host_output[i] == i);
  }
}
#endif // TEST_LAUNCH == 0

// Graph capture is what the lid_2 variant exercises, so this replay test only applies there.
#if TEST_LAUNCH == 2
CUB_TEST("captured DeviceScan::ExclusiveSum replays with zero and nonzero deferred counts",
         "[device][scan][deferred]",
         CUB_SMALL)
{
  constexpr count_t capacity = 100'000;

  c2h::device_vector<value_t> input(capacity, value_t{1});
  c2h::device_vector<count_t> device_num_items(1, count_t{-1});
  c2h::device_vector<value_t> output(capacity, value_t{-1});

  const auto d_num_items = thrust::raw_pointer_cast(device_num_items.data());
  const auto d_output    = thrust::raw_pointer_cast(output.data());
  const auto count       = cuda::args::deferred{
    d_num_items, cuda::args::bounds<count_t{0}, capacity>(), cuda::args::bounds(count_t{0}, capacity)};

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  size_t temp_storage_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, input.begin(), output.begin(), count, stream));
  c2h::device_vector<uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input.begin(), output.begin(), count, stream));
  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t executable{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

  for (const count_t num_items : {capacity, count_t{0}, count_t{1'000}, capacity})
  {
    REQUIRE(cudaSuccess == cudaMemcpyAsync(d_num_items, &num_items, sizeof(num_items), cudaMemcpyHostToDevice, stream));
    REQUIRE(cudaSuccess == cudaMemsetAsync(d_output, 0xff, sizeof(value_t) * static_cast<size_t>(capacity), stream));
    REQUIRE(cudaSuccess == cudaGraphLaunch(executable, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));

    if (num_items == 0)
    {
      REQUIRE(output[0] == value_t{-1});
    }
    else
    {
      REQUIRE(output[0] == value_t{0});
      REQUIRE(output[num_items - 1] == num_items - 1);
      if (num_items < capacity)
      {
        REQUIRE(output[num_items] == value_t{-1});
      }
    }
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(executable));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}
#endif // TEST_LAUNCH == 2
