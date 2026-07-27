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
#include <thrust/transform.h>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>
#include <cuda/experimental/__cuco/types.cuh>

#include "../common/defaults.cuh"
#include "../common/key_generator.cuh"
#include <nvbench/nvbench.cuh>

namespace cudax = cuda::experimental;
namespace bench = cudax::cuco::benchmark;

/**
 * @brief A benchmark evaluating `cudax::cuco::fixed_capacity_map::insert_async` performance.
 */
template <typename Key, typename Value, typename Dist>
void fixed_capacity_map_insert(nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  if constexpr (sizeof(Key) != sizeof(Value))
  {
    state.skip("Key and Value must have the same size.");
  }
  else
  {
    using pair_type = cuda::std::pair<Key, Value>;
    using map_type  = cudax::cuco::fixed_capacity_map<Key, Value>;

    const auto num_keys  = state.get_int64("NumInputs");
    const auto occupancy = state.get_float64("Occupancy");

    const auto size = static_cast<cuda::std::size_t>(static_cast<double>(num_keys) / occupancy);

    const auto device = cuda::device_ref{0};
    cuda::stream stream{device};
    const cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(device);
    const auto exec_policy                = thrust::cuda::par_nosync.on(stream.get());

    auto keys = cuda::make_device_buffer<Key>(stream, device, num_keys, cuda::no_init);

    bench::key_generator gen{};
    gen.generate(bench::dist_from_state<Dist>(state), keys.begin(), keys.end(), exec_policy);

    auto pairs = cuda::make_device_buffer<pair_type>(stream, device, num_keys, cuda::no_init);
    thrust::transform(exec_policy, keys.begin(), keys.end(), pairs.begin(), [] __device__(Key const& key) {
      return pair_type{key, Value{}};
    });

    map_type map{stream, mr, size, cudax::cuco::empty_key(Key{-1}), cudax::cuco::empty_value(Value{-1})};
    stream.sync();

    state.add_element_count(num_keys);
    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      map.insert_async({launch.get_stream()}, pairs.begin(), pairs.end());
      timer.stop();
      map.clear_async({launch.get_stream()});
    });
  }
}

NVBENCH_BENCH_TYPES(fixed_capacity_map_insert,
                    NVBENCH_TYPE_AXES(bench::defaults::key_type_range,
                                      bench::defaults::value_type_range,
                                      nvbench::type_list<bench::distribution::unique>))
  .set_name("fixed_capacity_map_insert_unique_capacity")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .add_int64_axis("NumInputs", bench::defaults::n_range_cache)
  .add_float64_axis("Occupancy", {bench::defaults::occupancy});

NVBENCH_BENCH_TYPES(fixed_capacity_map_insert,
                    NVBENCH_TYPE_AXES(bench::defaults::key_type_range,
                                      bench::defaults::value_type_range,
                                      nvbench::type_list<bench::distribution::unique>))
  .set_name("fixed_capacity_map_insert_unique_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .add_int64_axis("NumInputs", {bench::defaults::n})
  .add_float64_axis("Occupancy", bench::defaults::occupancy_range);

NVBENCH_BENCH_TYPES(fixed_capacity_map_insert,
                    NVBENCH_TYPE_AXES(bench::defaults::key_type_range,
                                      bench::defaults::value_type_range,
                                      nvbench::type_list<bench::distribution::uniform>))
  .set_name("fixed_capacity_map_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .add_int64_axis("NumInputs", {bench::defaults::n})
  .add_float64_axis("Occupancy", {bench::defaults::occupancy})
  .add_float64_axis("Multiplicity", bench::defaults::multiplicity_range);
