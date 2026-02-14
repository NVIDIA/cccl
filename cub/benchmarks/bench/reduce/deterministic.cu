// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/dispatch/dispatch_reduce_deterministic.cuh>

#include <nvbench_helper.cuh>

#include <nvbench/range.cuh>
#include <nvbench/types.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 3:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#if !TUNE_BASE

struct AgentReducePolicy
{
  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = 4;

  /// Cooperative block-wide reduction algorithm to use
  static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING;

  /// Cache load modifier for reading input elements
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::CacheLoadModifier::LOAD_DEFAULT;
  constexpr static int ITEMS_PER_THREAD                 = TUNE_ITEMS_PER_THREAD;
  constexpr static int BLOCK_THREADS                    = TUNE_THREADS_PER_BLOCK;
};

struct policy_hub_t
{
  struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy350>
  {
    using ReducePolicy = AgentReducePolicy;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy350;
};
#endif // !TUNE_BASE

template <class T>
void deterministic_sum(nvbench::state& state, nvbench::type_list<T>)
{
  using input_it_t  = const T*;
  using output_it_t = T*;

  using init_t      = T;
  using accum_t     = T;
  using transform_t = ::cuda::std::identity;

  using dispatch_t = cub::detail::rfa::dispatch_t<
    input_it_t,
    output_it_t,
    int,
    init_t,
    transform_t,
    accum_t
#if !TUNE_BASE
    ,
    policy_hub_t
#endif
    >;

  const auto elements = static_cast<int>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(out.size());

  std::size_t temp_storage_bytes{};
  dispatch_t::Dispatch(nullptr, temp_storage_bytes, d_in, d_out, elements, {}, 0);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  auto* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(d_temp_storage, temp_storage_bytes, d_in, d_out, elements, {}, launch.get_stream());
  });
}

using types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(deterministic_sum, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
