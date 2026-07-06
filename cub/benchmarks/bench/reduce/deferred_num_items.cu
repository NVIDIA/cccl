// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/argument>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <cstddef>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

template <bool UseDeferred>
void sum_num_items(nvbench::state& state)
{
  using value_t  = float;
  using offset_t = cuda::std::int64_t;

  const auto elements  = state.get_int64("Elements{io}");
  const auto num_items = static_cast<offset_t>(elements);

  const thrust::device_vector<value_t> input = generate(elements);
  thrust::device_vector<value_t> output(1);
  const thrust::device_vector<offset_t> device_num_items(1, num_items);

  const auto d_input                  = thrust::raw_pointer_cast(input.data());
  const auto d_output                 = thrust::raw_pointer_cast(output.data());
  [[maybe_unused]] const auto d_count = thrust::raw_pointer_cast(device_num_items.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<value_t>(elements, "Size");
  state.add_global_memory_writes<value_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env = cub_bench_env(alloc, launch);
    if constexpr (UseDeferred)
    {
      _CCCL_TRY_CUDA_API(cub::DeviceReduce::Sum, "Sum failed", d_input, d_output, cuda::args::deferred{d_count}, env);
    }
    else
    {
      _CCCL_TRY_CUDA_API(cub::DeviceReduce::Sum, "Sum failed", d_input, d_output, num_items, env);
    }
  });
}

void sum_immediate_num_items(nvbench::state& state)
{
  sum_num_items<false>(state);
}

void sum_deferred_num_items(nvbench::state& state)
{
  sum_num_items<true>(state);
}

template <typename T>
struct select_less_than_t
{
  T bound;

  _CCCL_HOST_DEVICE_API bool operator()(const T& value) const
  {
    return value < bound;
  }
};

template <typename T>
class pinned_value_t
{
  T* value_{};

public:
  pinned_value_t()
  {
    void* allocation{};
    NVBENCH_CUDA_CALL(cudaMallocHost(&allocation, sizeof(T)));
    value_ = static_cast<T*>(allocation);
  }

  ~pinned_value_t()
  {
    if (value_ != nullptr)
    {
      NVBENCH_CUDA_CALL_NOEXCEPT(cudaFreeHost(value_));
    }
  }

  pinned_value_t(const pinned_value_t&)            = delete;
  pinned_value_t& operator=(const pinned_value_t&) = delete;

  [[nodiscard]] T* data() const
  {
    return value_;
  }

  [[nodiscard]] const T& value() const
  {
    return *value_;
  }
};

class captured_graph_t
{
  cudaGraph_t graph_{};
  cudaGraphExec_t executable_{};

public:
  captured_graph_t() = default;

  ~captured_graph_t()
  {
    if (executable_ != nullptr)
    {
      NVBENCH_CUDA_CALL_NOEXCEPT(cudaGraphExecDestroy(executable_));
    }
    if (graph_ != nullptr)
    {
      NVBENCH_CUDA_CALL_NOEXCEPT(cudaGraphDestroy(graph_));
    }
  }

  captured_graph_t(const captured_graph_t&)            = delete;
  captured_graph_t& operator=(const captured_graph_t&) = delete;

  template <typename ActionT>
  void capture(cudaStream_t stream, ActionT action)
  {
    NVBENCH_CUDA_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    action();
    NVBENCH_CUDA_CALL(cudaStreamEndCapture(stream, &graph_));
    NVBENCH_CUDA_CALL(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));
    NVBENCH_CUDA_CALL(cudaGraphUpload(executable_, stream));
    NVBENCH_CUDA_CALL(cudaStreamSynchronize(stream));
  }

  [[nodiscard]] cudaGraphExec_t get() const
  {
    return executable_;
  }
};

enum class pipeline_mode
{
  stream_deferred,
  stream_host_round_trip,
  graph_deferred,
  graph_host_round_trip
};

template <pipeline_mode Mode>
void select_reduce_pipeline(nvbench::state& state)
{
  using value_t  = cuda::std::int32_t;
  using accum_t  = cuda::std::int64_t;
  using offset_t = cuda::std::int64_t;

  constexpr bool use_deferred = Mode == pipeline_mode::stream_deferred || Mode == pipeline_mode::graph_deferred;
  constexpr bool use_graph    = Mode == pipeline_mode::graph_deferred || Mode == pipeline_mode::graph_host_round_trip;

  const auto elements  = state.get_int64("Elements{io}");
  const auto num_items = static_cast<offset_t>(elements);
  const auto select_op = select_less_than_t<value_t>{static_cast<value_t>(num_items / 2)};
  const auto stream    = state.get_cuda_stream().get_stream();

  thrust::device_vector<value_t> input(elements);
  thrust::sequence(input.begin(), input.end());
  thrust::device_vector<value_t> selected(elements, thrust::no_init);
  thrust::device_vector<offset_t> device_num_selected(1);
  thrust::device_vector<accum_t> output(1);

  const auto d_input        = thrust::raw_pointer_cast(input.data());
  const auto d_selected     = thrust::raw_pointer_cast(selected.data());
  const auto d_num_selected = thrust::raw_pointer_cast(device_num_selected.data());
  const auto d_output       = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<value_t>(elements, "Size");
  state.add_global_memory_writes<value_t>(num_items / 2);
  state.add_global_memory_reads<value_t>(num_items / 2);
  state.add_global_memory_writes<accum_t>(1);

  std::size_t select_temp_storage_bytes{};
  _CCCL_TRY_CUDA_API(
    cub::DeviceSelect::If,
    "Select temporary-storage query failed",
    nullptr,
    select_temp_storage_bytes,
    d_input,
    d_selected,
    d_num_selected,
    num_items,
    select_op,
    stream);

  const auto deferred_num_items = cuda::args::deferred{d_num_selected};
  std::size_t reduce_temp_storage_bytes{};
  if constexpr (use_deferred)
  {
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::Reduce,
      "Deferred Reduce temporary-storage query failed",
      nullptr,
      reduce_temp_storage_bytes,
      d_selected,
      d_output,
      deferred_num_items,
      cuda::std::plus<>{},
      accum_t{},
      stream);
  }
  else
  {
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::Reduce,
      "Reduce temporary-storage query failed",
      nullptr,
      reduce_temp_storage_bytes,
      d_selected,
      d_output,
      num_items,
      cuda::std::plus<>{},
      accum_t{},
      stream);
  }

  const auto temp_storage_bytes =
    select_temp_storage_bytes < reduce_temp_storage_bytes ? reduce_temp_storage_bytes : select_temp_storage_bytes;
  thrust::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  const auto d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  pinned_value_t<offset_t> host_num_selected;
  captured_graph_t graph;

  if constexpr (use_graph)
  {
    graph.capture(stream, [&] {
      auto select_bytes = select_temp_storage_bytes;
      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::If,
        "Select failed during graph capture",
        d_temp_storage,
        select_bytes,
        d_input,
        d_selected,
        d_num_selected,
        num_items,
        select_op,
        stream);

      if constexpr (use_deferred)
      {
        auto reduce_bytes = reduce_temp_storage_bytes;
        _CCCL_TRY_CUDA_API(
          cub::DeviceReduce::Reduce,
          "Deferred Reduce failed during graph capture",
          d_temp_storage,
          reduce_bytes,
          d_selected,
          d_output,
          deferred_num_items,
          cuda::std::plus<>{},
          accum_t{},
          stream);
      }
      else
      {
        // An immediate reduction cannot be captured until the host has observed the selected count. Capture the
        // producer and count copy here; the timed replay synchronizes and launches the reduction separately.
        NVBENCH_CUDA_CALL(
          cudaMemcpyAsync(host_num_selected.data(), d_num_selected, sizeof(offset_t), cudaMemcpyDeviceToHost, stream));
      }
    });
  }

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    const auto launch_stream = launch.get_stream().get_stream();

    if constexpr (Mode == pipeline_mode::stream_deferred)
    {
      auto select_bytes = select_temp_storage_bytes;
      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::If,
        "Select failed",
        d_temp_storage,
        select_bytes,
        d_input,
        d_selected,
        d_num_selected,
        num_items,
        select_op,
        launch_stream);

      auto reduce_bytes = reduce_temp_storage_bytes;
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::Reduce,
        "Deferred Reduce failed",
        d_temp_storage,
        reduce_bytes,
        d_selected,
        d_output,
        deferred_num_items,
        cuda::std::plus<>{},
        accum_t{},
        launch_stream);
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(launch_stream));
    }
    else if constexpr (Mode == pipeline_mode::stream_host_round_trip)
    {
      auto select_bytes = select_temp_storage_bytes;
      _CCCL_TRY_CUDA_API(
        cub::DeviceSelect::If,
        "Select failed",
        d_temp_storage,
        select_bytes,
        d_input,
        d_selected,
        d_num_selected,
        num_items,
        select_op,
        launch_stream);
      NVBENCH_CUDA_CALL(cudaMemcpyAsync(
        host_num_selected.data(), d_num_selected, sizeof(offset_t), cudaMemcpyDeviceToHost, launch_stream));
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(launch_stream));

      auto reduce_bytes = reduce_temp_storage_bytes;
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::Reduce,
        "Reduce failed",
        d_temp_storage,
        reduce_bytes,
        d_selected,
        d_output,
        host_num_selected.value(),
        cuda::std::plus<>{},
        accum_t{},
        launch_stream);
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(launch_stream));
    }
    else if constexpr (Mode == pipeline_mode::graph_deferred)
    {
      NVBENCH_CUDA_CALL(cudaGraphLaunch(graph.get(), launch_stream));
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(launch_stream));
    }
    else
    {
      NVBENCH_CUDA_CALL(cudaGraphLaunch(graph.get(), launch_stream));
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(launch_stream));

      auto reduce_bytes = reduce_temp_storage_bytes;
      _CCCL_TRY_CUDA_API(
        cub::DeviceReduce::Reduce,
        "Reduce failed",
        d_temp_storage,
        reduce_bytes,
        d_selected,
        d_output,
        host_num_selected.value(),
        cuda::std::plus<>{},
        accum_t{},
        launch_stream);
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(launch_stream));
    }
  });

  const offset_t expected_num_selected = num_items / 2;
  const accum_t expected_output =
    static_cast<accum_t>(expected_num_selected) * static_cast<accum_t>(expected_num_selected - 1) / 2;
  if (device_num_selected[0] != expected_num_selected || output[0] != expected_output)
  {
    state.skip("Select-to-reduce pipeline produced an incorrect result.");
  }
}

void select_reduce_stream_deferred(nvbench::state& state)
{
  select_reduce_pipeline<pipeline_mode::stream_deferred>(state);
}

void select_reduce_stream_host_round_trip(nvbench::state& state)
{
  select_reduce_pipeline<pipeline_mode::stream_host_round_trip>(state);
}

void select_reduce_graph_deferred(nvbench::state& state)
{
  select_reduce_pipeline<pipeline_mode::graph_deferred>(state);
}

void select_reduce_graph_host_round_trip(nvbench::state& state)
{
  select_reduce_pipeline<pipeline_mode::graph_host_round_trip>(state);
}

NVBENCH_BENCH(sum_immediate_num_items)
  .set_name("sum/immediate")
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(0, 28, 4));

NVBENCH_BENCH(sum_deferred_num_items)
  .set_name("sum/deferred")
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(0, 28, 4));

NVBENCH_BENCH(select_reduce_stream_deferred)
  .set_name("select_reduce/stream/deferred")
  .add_int64_axis("Elements{io}", {1 << 10, 1 << 20, 1 << 26});

NVBENCH_BENCH(select_reduce_stream_host_round_trip)
  .set_name("select_reduce/stream/host_round_trip")
  .add_int64_axis("Elements{io}", {1 << 10, 1 << 20, 1 << 26});

NVBENCH_BENCH(select_reduce_graph_deferred)
  .set_name("select_reduce/graph/deferred")
  .add_int64_axis("Elements{io}", {1 << 10, 1 << 20, 1 << 26});

NVBENCH_BENCH(select_reduce_graph_host_round_trip)
  .set_name("select_reduce/graph/host_round_trip")
  .add_int64_axis("Elements{io}", {1 << 10, 1 << 20, 1 << 26});
