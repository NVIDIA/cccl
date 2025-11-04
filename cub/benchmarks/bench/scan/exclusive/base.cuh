// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_scan.cuh>

#include <cuda/std/__functional/invoke.h>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename AccumT>
struct policy_hub_t
{
  template <int NOMINAL_BLOCK_THREADS_4B,
            int NOMINAL_ITEMS_PER_THREAD_4B,
            typename ComputeT,
            cub::BlockLoadAlgorithm LOAD_ALGORITHM,
            cub::CacheLoadModifier LOAD_MODIFIER,
            cub::BlockStoreAlgorithm STORE_ALGORITHM,
            cub::BlockScanAlgorithm SCAN_ALGORITHM>
  using agent_policy_t = cub::AgentScanPolicy<
    NOMINAL_BLOCK_THREADS_4B,
    NOMINAL_ITEMS_PER_THREAD_4B,
    ComputeT,
    LOAD_ALGORITHM,
    LOAD_MODIFIER,
    STORE_ALGORITHM,
    SCAN_ALGORITHM,
    cub::detail::MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>,
    delay_constructor_t>;

  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using ScanPolicyT =
      agent_policy_t<TUNE_THREADS,
                     TUNE_ITEMS,
                     AccumT,
                     TUNE_LOAD_ALGORITHM,
                     TUNE_LOAD_MODIFIER,
                     TUNE_STORE_ALGORITHM,
                     cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};
#endif // TUNE_BASE

template <typename T, typename OffsetT>
static void basic(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using init_t         = T;
  using wrapped_init_t = cub::detail::InputValue<init_t>;
  using accum_t        = ::cuda::std::__accumulator_t<op_t, init_t, T>;
  using input_it_t     = const T*;
  using output_it_t    = T*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::
    DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  T* d_input  = thrust::raw_pointer_cast(input.data());
  T* d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(
    nullptr, tmp_size, d_input, d_output, op_t{}, wrapped_init_t{T{}}, static_cast<int>(input.size()), 0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      d_input,
      d_output,
      op_t{},
      wrapped_init_t{T{}},
      static_cast<int>(input.size()),
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(all_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
