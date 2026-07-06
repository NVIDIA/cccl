// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/util_device.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/functional.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/argument>
#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/generators.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Min, device_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Max, device_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::TransformReduce, device_transform_reduce);

// %PARAM% TEST_LAUNCH lid 0:1:2

template <typename T>
struct select_less_than_device_value_t
{
  const T* bound;

  _CCCL_DEVICE_API bool operator()(const T& value) const
  {
    return value < *bound;
  }
};

struct fixed_reduce_policy_selector_t
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::detail::reduce::reduce_policy
  {
    const auto policy =
      cub::detail::reduce::agent_reduce_policy{32, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy};
  }
};

struct fixed_grid_factory_t : cub::detail::TripleChevronFactory
{
  CUB_RUNTIME_FUNCTION cudaError_t MultiProcessorCount(int& sm_count) const
  {
    sm_count = 1;
    return cudaSuccess;
  }

  template <typename KernelT>
  CUB_RUNTIME_FUNCTION cudaError_t
  MaxSmOccupancy(int& sm_occupancy, KernelT, int, [[maybe_unused]] int dynamic_smem_bytes = 0)
  {
    sm_occupancy = 1;
    return cudaSuccess;
  }
};

struct fixed_grid_reduce_t
{
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT, typename ReductionOpT, typename InitT>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream = nullptr) const
  {
    return cub::detail::reduce::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      reduction_op,
      init,
      stream,
      cuda::std::identity{},
      fixed_reduce_policy_selector_t{},
      {},
      fixed_grid_factory_t{});
  }
};

struct select_then_reduce_t
{
  template <typename InputIteratorT,
            typename SelectedOutputIteratorT,
            typename NumSelectedIteratorT,
            typename ReduceOutputIteratorT,
            typename NumItemsT,
            typename SelectOpT,
            typename ReductionOpT,
            typename InitT>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    InputIteratorT d_in,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    ReduceOutputIteratorT d_reduce_out,
    NumItemsT num_items,
    SelectOpT select_op,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream = nullptr) const
  {
    std::size_t select_temp_storage_bytes{};
    if (const cudaError_t error = cub::DeviceSelect::If(
          nullptr, select_temp_storage_bytes, d_in, d_selected_out, d_num_selected_out, num_items, select_op, stream);
        error != cudaSuccess)
    {
      return error;
    }

    const auto deferred_num_items = cuda::args::deferred{d_num_selected_out};
    std::size_t reduce_temp_storage_bytes{};
    if (const cudaError_t error = cub::DeviceReduce::Reduce(
          nullptr,
          reduce_temp_storage_bytes,
          d_selected_out,
          d_reduce_out,
          deferred_num_items,
          reduction_op,
          init,
          stream);
        error != cudaSuccess)
    {
      return error;
    }

    temp_storage_bytes =
      select_temp_storage_bytes < reduce_temp_storage_bytes ? reduce_temp_storage_bytes : select_temp_storage_bytes;
    if (d_temp_storage == nullptr)
    {
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

    return cub::DeviceReduce::Reduce(
      d_temp_storage,
      reduce_temp_storage_bytes,
      d_selected_out,
      d_reduce_out,
      deferred_num_items,
      reduction_op,
      init,
      stream);
  }
};

using count_types = c2h::type_list<std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;

C2H_TEST("DeviceReduce::Reduce accepts deferred num_items", "[device][reduce][deferred]", count_types)
{
  using value_t = std::int64_t;
  using count_t = typename c2h::get<0, TestType>;

  constexpr count_t max_num_items = 100'000;
  const count_t num_items         = GENERATE_COPY(count_t{0}, count_t{1}, count_t{1'000}, max_num_items);
  constexpr value_t init          = 42;

  const c2h::device_vector<value_t> input(max_num_items, value_t{1});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto d_input     = thrust::raw_pointer_cast(input.data());
  const auto d_output    = thrust::raw_pointer_cast(output.data());
  const auto d_num_items = thrust::raw_pointer_cast(device_num_items.data());

  device_reduce(d_input, d_output, cuda::args::deferred{d_num_items}, cuda::std::plus<>{}, init);

  REQUIRE(output[0] == init + static_cast<value_t>(num_items));
}

C2H_TEST("DeviceReduce::Reduce with deferred num_items is run-to-run reproducible",
         "[device][reduce][deferred][run_to_run]")
{
  using value_t = float;
  using count_t = std::int32_t;

  constexpr count_t num_items = 100'000;
  constexpr int num_runs      = 3;

  c2h::device_vector<value_t> input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), input, value_t{-100}, value_t{100});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> reference(1, thrust::no_init);
  c2h::device_vector<value_t> output(1, thrust::no_init);

  const auto d_input     = thrust::raw_pointer_cast(input.data());
  const auto d_reference = thrust::raw_pointer_cast(reference.data());
  const auto d_output    = thrust::raw_pointer_cast(output.data());
  const auto count       = cuda::args::deferred{device_num_items.begin()};

  device_reduce(d_input, d_reference, count, cuda::std::plus<>{}, value_t{});
  for (int run = 1; run < num_runs; ++run)
  {
    device_reduce(d_input, d_output, count, cuda::std::plus<>{}, value_t{});
    REQUIRE_THAT(detail::to_vec(reference), detail::BitwiseEqualsRange(detail::to_vec(output)));
  }
}

C2H_TEST("DeviceReduce::Reduce handles deferred grid boundaries", "[device][reduce][deferred]")
{
  using value_t = std::int64_t;
  using count_t = std::int32_t;

  constexpr count_t tile_size  = 32;
  constexpr count_t max_blocks = cub::detail::subscription_factor;
  const count_t num_items      = GENERATE_COPY(
    tile_size - 1,
    tile_size,
    tile_size + 1,
    max_blocks * tile_size - 1,
    max_blocks * tile_size,
    max_blocks * tile_size + 1);

  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto d_output = thrust::raw_pointer_cast(output.data());
  const auto count    = cuda::args::deferred{device_num_items.begin()};
  launch(fixed_grid_reduce_t{},
         cuda::constant_iterator<value_t>{value_t{1}},
         d_output,
         count,
         cuda::std::plus<>{},
         value_t{7});
  REQUIRE(output[0] == value_t{7} + num_items);
}

C2H_TEST("DeviceReduce::Reduce accepts deferred span and fancy-iterator sources", "[device][reduce][deferred]")
{
  using value_t = std::int64_t;
  using count_t = std::int32_t;

  constexpr count_t capacity  = 100'000;
  constexpr count_t num_items = 1'000;

  const c2h::device_vector<value_t> input(capacity, value_t{1});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto d_input     = thrust::raw_pointer_cast(input.data());
  const auto d_output    = thrust::raw_pointer_cast(output.data());
  const auto d_num_items = thrust::raw_pointer_cast(device_num_items.data());

  const auto count_span = cuda::std::span<const count_t, 1>{d_num_items, 1};
  device_reduce(d_input, d_output, cuda::args::deferred{count_span}, cuda::std::plus<>{}, value_t{});
  REQUIRE(output[0] == num_items);

  output[0]                  = value_t{-2};
  const auto count_transform = cuda::transform_iterator(d_num_items, cuda::std::identity{});
  device_reduce(d_input, d_output, cuda::args::deferred{count_transform}, cuda::std::plus<>{}, value_t{});
  REQUIRE(output[0] == num_items);

  output[0]                = value_t{-3};
  const auto bounded_count = cuda::args::deferred{
    d_num_items, cuda::args::bounds<count_t{0}, capacity>(), cuda::args::bounds(count_t{0}, capacity)};
  device_reduce(d_input, d_output, bounded_count, cuda::std::plus<>{}, value_t{});
  REQUIRE(output[0] == num_items);
}

C2H_TEST("DeviceReduce entry points sharing reduce dispatch accept deferred num_items", "[device][reduce][deferred]")
{
  using value_t = std::int64_t;
  using count_t = std::int32_t;

  constexpr count_t capacity  = 100'000;
  constexpr count_t num_items = 1'000;

  c2h::device_vector<value_t> input(capacity, value_t{2});
  input[num_items]     = value_t{1};
  input[num_items + 1] = value_t{3};
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto d_input     = thrust::raw_pointer_cast(input.data());
  const auto d_output    = thrust::raw_pointer_cast(output.data());
  const auto d_num_items = thrust::raw_pointer_cast(device_num_items.data());
  const auto count       = cuda::args::deferred{d_num_items};

  output[0] = value_t{-2};
  device_sum(d_input, d_output, count);
  REQUIRE(output[0] == value_t{2} * num_items);

  output[0] = value_t{-3};
  device_min(d_input, d_output, count);
  REQUIRE(output[0] == value_t{2});

  output[0] = value_t{-4};
  device_max(d_input, d_output, count);
  REQUIRE(output[0] == value_t{2});

  output[0] = value_t{-5};
  device_transform_reduce(d_input, d_output, count, cuda::std::plus<>{}, thrust::square<value_t>{}, value_t{});
  REQUIRE(output[0] == value_t{4} * num_items);
}

C2H_TEST("DeviceReduce::Reduce consumes a count produced by DeviceSelect::If", "[device][reduce][deferred]")
{
  using value_t = std::int64_t;
  using count_t = std::int32_t;

  constexpr count_t num_items  = 100'000;
  const count_t selected_count = GENERATE_COPY(count_t{0}, count_t{1}, count_t{1'000}, num_items);
  CAPTURE(selected_count);

  const auto d_input = cuda::counting_iterator<value_t>{value_t{0}};
  c2h::device_vector<value_t> selected_items(num_items, value_t{num_items + 1});
  c2h::device_vector<count_t> device_num_selected(1, count_t{-1});
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto d_selected_items = thrust::raw_pointer_cast(selected_items.data());
  const auto d_num_selected   = thrust::raw_pointer_cast(device_num_selected.data());
  const auto d_output         = thrust::raw_pointer_cast(output.data());
  const auto select_op        = thrust::placeholders::_1 < value_t{selected_count};

  launch(select_then_reduce_t{},
         d_input,
         d_selected_items,
         d_num_selected,
         d_output,
         num_items,
         select_op,
         cuda::std::plus<>{},
         value_t{});

  const auto expected_sum =
    static_cast<value_t>(selected_count) * static_cast<value_t>(selected_count - 1) / value_t{2};
  REQUIRE(device_num_selected[0] == selected_count);
  REQUIRE(output[0] == expected_sum);
}

#if TEST_LAUNCH == 0
C2H_TEST("DeviceReduce::Reduce consumes a deferred count produced in another stream after an event",
         "[device][reduce][deferred]")
{
  using value_t = std::int64_t;
  using count_t = std::int32_t;

  constexpr count_t capacity       = 100'000;
  constexpr count_t selected_count = 1'000;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));
  cuda::stream producer{cuda::devices[current_device]};
  cuda::stream consumer{cuda::devices[current_device]};

  const auto d_input = cuda::counting_iterator<value_t>{value_t{0}};
  c2h::device_vector<value_t> selected_items(capacity, value_t{capacity + 1});
  c2h::device_vector<count_t> device_num_selected(1, count_t{-1});
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto d_selected_items = thrust::raw_pointer_cast(selected_items.data());
  const auto d_num_selected   = thrust::raw_pointer_cast(device_num_selected.data());
  const auto d_output         = thrust::raw_pointer_cast(output.data());
  const auto num_selected     = cuda::args::deferred{d_num_selected};
  const auto select_op        = thrust::placeholders::_1 < value_t{selected_count};

  std::size_t select_temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      nullptr,
      select_temp_storage_bytes,
      d_input,
      d_selected_items,
      d_num_selected,
      capacity,
      select_op,
      producer.get()));

  std::size_t reduce_temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(
      nullptr,
      reduce_temp_storage_bytes,
      d_selected_items,
      d_output,
      num_selected,
      cuda::std::plus<>{},
      value_t{},
      consumer.get()));

  const std::size_t temp_storage_bytes =
    select_temp_storage_bytes < reduce_temp_storage_bytes ? reduce_temp_storage_bytes : select_temp_storage_bytes;
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      d_temp_storage,
      select_temp_storage_bytes,
      d_input,
      d_selected_items,
      d_num_selected,
      capacity,
      select_op,
      producer.get()));

  const auto count_ready = producer.record_event();
  consumer.wait(count_ready);

  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(
      d_temp_storage,
      reduce_temp_storage_bytes,
      d_selected_items,
      d_output,
      num_selected,
      cuda::std::plus<>{},
      value_t{},
      consumer.get()));
  consumer.sync();

  REQUIRE(device_num_selected[0] == selected_count);
  const value_t expected_sum =
    static_cast<value_t>(selected_count) * static_cast<value_t>(selected_count - 1) / value_t{2};
  REQUIRE(output[0] == expected_sum);
}

C2H_TEST("DeviceReduce environment entry points accept deferred num_items", "[device][reduce][deferred]")
{
  using value_t = std::int64_t;
  using count_t = std::int32_t;

  constexpr count_t num_items = 1'000;
  c2h::device_vector<value_t> input(num_items + 2, value_t{2});
  input[num_items]     = value_t{1};
  input[num_items + 1] = value_t{3};
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto count = cuda::args::deferred{device_num_items.begin()};

  REQUIRE(
    cudaSuccess == cub::DeviceReduce::Reduce(input.begin(), output.begin(), count, cuda::std::plus<>{}, value_t{7}));
  REQUIRE(output[0] == value_t{7} + value_t{2} * num_items);

  output[0] = value_t{-2};
  REQUIRE(cudaSuccess == cub::DeviceReduce::Sum(input.begin(), output.begin(), count));
  REQUIRE(output[0] == value_t{2} * num_items);

  output[0] = value_t{-3};
  REQUIRE(cudaSuccess == cub::DeviceReduce::Min(input.begin(), output.begin(), count));
  REQUIRE(output[0] == value_t{2});

  output[0] = value_t{-4};
  REQUIRE(cudaSuccess == cub::DeviceReduce::Max(input.begin(), output.begin(), count));
  REQUIRE(output[0] == value_t{2});

  output[0] = value_t{-5};
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::TransformReduce(
            input.begin(), output.begin(), count, cuda::std::plus<>{}, thrust::square<value_t>{}, value_t{}));
  REQUIRE(output[0] == value_t{4} * num_items);
}

C2H_TEST("DeviceReduce::Reduce storage query does not read deferred num_items", "[device][reduce][deferred]")
{
  using value_t = std::int64_t;
  using count_t = std::int32_t;

  std::size_t temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(
      nullptr,
      temp_storage_bytes,
      static_cast<const value_t*>(nullptr),
      static_cast<value_t*>(nullptr),
      cuda::args::deferred{static_cast<const count_t*>(nullptr)},
      cuda::std::plus<>{},
      value_t{}));
  REQUIRE(temp_storage_bytes > 0);
}

C2H_TEST("DeviceReduce::Reduce supports deferred num_items with not_guaranteed determinism",
         "[device][reduce][deferred]")
{
  using value_t = float;
  using count_t = std::int32_t;

  constexpr count_t capacity = 100'000;
  const count_t num_items    = GENERATE_COPY(count_t{0}, count_t{1}, count_t{1'000}, capacity);
  constexpr value_t init     = 3.0f;

  const c2h::device_vector<value_t> input(capacity, value_t{1});
  const c2h::device_vector<count_t> device_num_items(1, num_items);
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(
      input.begin(), output.begin(), cuda::args::deferred{device_num_items.begin()}, cuda::std::plus<>{}, init, env));
  REQUIRE(output[0] == init + static_cast<value_t>(num_items));
}
#endif // TEST_LAUNCH == 0

#if TEST_LAUNCH == 2
C2H_TEST("captured DeviceSelect::If to DeviceReduce::Reduce pipeline replays with changing counts",
         "[device][reduce][deferred]")
{
  using value_t  = std::int64_t;
  using count_t  = std::int32_t;
  using offset_t = cub::detail::choose_offset_t<count_t>;

  constexpr count_t capacity = 100'000;
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const auto policy       = cub::detail::reduce::policy_selector_from_types<value_t, offset_t, cuda::std::plus<>>{}(cc);
  const count_t tile_size = static_cast<count_t>(policy.reduce.threads_per_block * policy.reduce.items_per_thread);

  const auto d_input = cuda::counting_iterator<value_t>{value_t{0}};
  c2h::device_vector<value_t> selected_items(capacity, value_t{capacity + 1});
  c2h::device_vector<value_t> device_bound(1, value_t{-1});
  c2h::device_vector<count_t> device_num_selected(1, count_t{-1});
  c2h::device_vector<value_t> output(1, value_t{-1});

  const auto d_selected_items = thrust::raw_pointer_cast(selected_items.data());
  const auto d_bound          = thrust::raw_pointer_cast(device_bound.data());
  const auto d_num_selected   = thrust::raw_pointer_cast(device_num_selected.data());
  const auto d_output         = thrust::raw_pointer_cast(output.data());
  const auto select_op        = select_less_than_device_value_t<value_t>{d_bound};
  const auto num_selected     = cuda::args::deferred{d_num_selected};

  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));

  std::size_t select_temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      nullptr, select_temp_storage_bytes, d_input, d_selected_items, d_num_selected, capacity, select_op, stream));

  std::size_t reduce_temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(
      nullptr,
      reduce_temp_storage_bytes,
      d_selected_items,
      d_output,
      num_selected,
      cuda::std::plus<>{},
      value_t{},
      stream));

  const std::size_t temp_storage_bytes =
    select_temp_storage_bytes < reduce_temp_storage_bytes ? reduce_temp_storage_bytes : select_temp_storage_bytes;
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      d_temp_storage, select_temp_storage_bytes, d_input, d_selected_items, d_num_selected, capacity, select_op, stream));
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(
      d_temp_storage,
      reduce_temp_storage_bytes,
      d_selected_items,
      d_output,
      num_selected,
      cuda::std::plus<>{},
      value_t{},
      stream));
  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t executable{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

  for (const count_t selected_count :
       {count_t{0}, count_t{1}, tile_size - 1, tile_size, tile_size + 1, capacity, count_t{0}})
  {
    const value_t bound = selected_count;
    REQUIRE(cudaSuccess == cudaMemcpyAsync(d_bound, &bound, sizeof(value_t), cudaMemcpyHostToDevice, stream));
    REQUIRE(cudaSuccess == cudaGraphLaunch(executable, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));
    REQUIRE(device_num_selected[0] == selected_count);
    const value_t expected_sum =
      static_cast<value_t>(selected_count) * static_cast<value_t>(selected_count - 1) / value_t{2};
    REQUIRE(output[0] == expected_sum);
  }

  REQUIRE(cudaSuccess == cudaGraphExecDestroy(executable));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}
#endif // TEST_LAUNCH == 2
