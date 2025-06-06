/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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

template <class T, typename OffsetT>
void deterministic_sum(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t  = const T*;
  using output_it_t = T*;
  using offset_t    = cub::detail::choose_offset_t<OffsetT>;

  using init_t      = T;
  using accum_t     = T;
  using transform_t = ::cuda::std::identity;

  using dispatch_t = cub::detail::DispatchReduceDeterministic<
    input_it_t,
    output_it_t,
    offset_t,
    init_t,
    transform_t,
    accum_t
#if !TUNE_BASE
    ,
    policy_hub_t
#endif
    >;

  const auto elements = static_cast<T>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(out.size());

  std::size_t temp_storage_bytes{};
  dispatch_t::Dispatch(nullptr, temp_storage_bytes, d_in, d_out, static_cast<offset_t>(elements), {}, 0);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  auto* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, static_cast<offset_t>(elements), {}, launch.get_stream());
  });
}

using types               = nvbench::type_list<float, double>;
using custom_offset_types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(deterministic_sum, NVBENCH_TYPE_AXES(types, custom_offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
