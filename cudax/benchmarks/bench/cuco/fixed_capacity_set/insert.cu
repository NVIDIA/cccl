// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/execution_policy.h>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cuda/experimental/__cuco/fixed_capacity_set.cuh>
#include <cuda/experimental/__cuco/types.cuh>

#include "../common/defaults.cuh"
#include "../common/key_generator.cuh"
#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;
namespace bench = cudax::cuco::benchmark;

template <typename Key, typename Dist>
void fixed_capacity_set_insert(nvbench::state& state, nvbench::type_list<Key, Dist>)
{
  const auto num_keys  = state.get_int64("NumInputs");
  const auto occupancy = state.get_float64("Occupancy");
  // Multiplicity sweeps keep capacity fixed; duplicates reduce the final occupancy.
  const auto capacity = static_cast<cuda::std::size_t>(static_cast<double>(num_keys) / occupancy);

  const cuda::stream_ref stream{state.get_cuda_stream().get_stream()};
  const auto mr          = cuda::device_default_memory_pool(stream.device());
  const auto exec_policy = thrust::cuda::par_nosync.on(stream.get());

  auto keys = cuda::make_buffer<Key>(stream, mr, num_keys, cuda::no_init);
  bench::key_generator gen{};
  gen.generate(bench::dist_from_state<Dist>(state), keys.begin(), keys.end(), exec_policy);

  cudax::cuco::fixed_capacity_set<Key> set{stream, mr, capacity, cudax::cuco::empty_key<Key>{-1}};
  stream.sync();

  state.add_element_count(num_keys);
  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    set.insert_async({launch.get_stream()}, keys.begin(), keys.end());
    timer.stop();
    // Reset outside the timed interval so every sample inserts into an empty set.
    set.clear_async({launch.get_stream()});
  });
}

NVBENCH_BENCH_TYPES(fixed_capacity_set_insert,
                    NVBENCH_TYPE_AXES(bench::defaults::key_type_range, nvbench::type_list<bench::distribution::unique>))
  .set_name("fixed_capacity_set_insert_unique_capacity")
  .set_type_axes_names({"Key", "Distribution"})
  .add_int64_axis("NumInputs", bench::defaults::n_range_cache)
  .add_float64_axis("Occupancy", {bench::defaults::occupancy});

NVBENCH_BENCH_TYPES(fixed_capacity_set_insert,
                    NVBENCH_TYPE_AXES(bench::defaults::key_type_range, nvbench::type_list<bench::distribution::unique>))
  .set_name("fixed_capacity_set_insert_unique_occupancy")
  .set_type_axes_names({"Key", "Distribution"})
  .add_int64_axis("NumInputs", {bench::defaults::n})
  .add_float64_axis("Occupancy", bench::defaults::occupancy_range);

NVBENCH_BENCH_TYPES(fixed_capacity_set_insert,
                    NVBENCH_TYPE_AXES(bench::defaults::key_type_range, nvbench::type_list<bench::distribution::uniform>))
  .set_name("fixed_capacity_set_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Distribution"})
  .add_int64_axis("NumInputs", {bench::defaults::n})
  .add_float64_axis("Occupancy", {bench::defaults::occupancy})
  .add_float64_axis("Multiplicity", bench::defaults::multiplicity_range);
