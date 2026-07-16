//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cuda/experimental/__cuco/hyperloglog.cuh>

#include "common/defaults.cuh"
#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;
namespace bench = cudax::cuco::benchmark;

namespace
{
template <typename Key>
void add_relative_error_summary(
  nvbench::state& state,
  cudax::cuco::hyperloglog<Key>& estimator,
  cuda::stream_ref stream,
  Key* first,
  cuda::std::size_t num_items)
{
  estimator.add(stream, first, first + num_items);
  const auto estimated_cardinality = estimator.estimate(stream);
  const auto relative_error =
    cuda::std::abs(static_cast<double>(estimated_cardinality) / static_cast<double>(num_items) - 1.0);
  estimator.clear(stream);

  auto& summary = state.add_summary("MeanRelativeError");
  summary.set_string("hint", "MRelErr");
  summary.set_string("short_name", "MeanRelativeError");
  summary.set_string("description", "Mean relative approximation error.");
  summary.set_float64("value", relative_error);
}
} // namespace

/**
 * @brief A benchmark evaluating `cudax::cuco::hyperloglog` end-to-end performance.
 */
template <typename Key>
void hyperloglog_e2e(nvbench::state& state, nvbench::type_list<Key>)
{
  using estimator_type      = cudax::cuco::hyperloglog<Key>;
  using sketch_size_kb_type = typename estimator_type::sketch_size_kb;

  const auto num_items      = static_cast<cuda::std::size_t>(state.get_int64("NumInputs"));
  const auto sketch_size_kb = sketch_size_kb_type{static_cast<double>(state.get_int64("SketchSizeKB"))};

  const auto device = cuda::device_ref{0};
  cuda::stream stream{device};
  const cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(device);

  auto items = cuda::make_device_buffer<Key>(stream, device, num_items, cuda::no_init);
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), items.begin(), items.end(), Key{0});

  estimator_type estimator{stream, mr, sketch_size_kb};
  stream.sync();

  state.add_element_count(num_items);
  state.add_global_memory_reads<Key>(num_items, "InputSize");

  add_relative_error_summary(state, estimator, stream, items.data(), num_items);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    estimator.add_async({launch.get_stream()}, items.begin(), items.end());
    [[maybe_unused]] const auto estimated_cardinality = estimator.estimate({launch.get_stream()});
    timer.stop();

    estimator.clear_async({launch.get_stream()});
  });
}

/**
 * @brief A benchmark evaluating `cudax::cuco::hyperloglog::add_async` performance.
 */
template <typename Key>
void hyperloglog_add(nvbench::state& state, nvbench::type_list<Key>)
{
  using estimator_type      = cudax::cuco::hyperloglog<Key>;
  using sketch_size_kb_type = typename estimator_type::sketch_size_kb;

  const auto num_items      = static_cast<cuda::std::size_t>(state.get_int64("NumInputs"));
  const auto sketch_size_kb = sketch_size_kb_type{static_cast<double>(state.get_int64("SketchSizeKB"))};

  const auto device = cuda::device_ref{0};
  cuda::stream stream{device};
  const cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(device);

  auto items = cuda::make_device_buffer<Key>(stream, device, num_items, cuda::no_init);
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), items.begin(), items.end(), Key{0});

  estimator_type estimator{stream, mr, sketch_size_kb};
  stream.sync();

  state.add_element_count(num_items);
  state.add_global_memory_reads<Key>(num_items, "InputSize");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    estimator.add_async({launch.get_stream()}, items.begin(), items.end());
    timer.stop();

    estimator.clear_async({launch.get_stream()});
  });
}

NVBENCH_BENCH_TYPES(hyperloglog_e2e, NVBENCH_TYPE_AXES(bench::defaults::key_type_range))
  .set_name("hyperloglog_e2e")
  .set_type_axes_names({"Key"})
  .add_int64_power_of_two_axis("NumInputs", {30})
  .add_int64_axis("SketchSizeKB", {8, 16, 32, 64, 128, 256});

NVBENCH_BENCH_TYPES(hyperloglog_add, NVBENCH_TYPE_AXES(bench::defaults::key_type_range))
  .set_name("hyperloglog_add")
  .set_type_axes_names({"Key"})
  .add_int64_power_of_two_axis("NumInputs", {30})
  .add_int64_axis("SketchSizeKB", {8, 16, 32, 64, 128, 256});
