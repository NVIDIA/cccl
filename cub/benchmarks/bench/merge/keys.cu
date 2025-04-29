// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_merge.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/utility>

#include <cstdint>

#include "merge_common.cuh"
#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

template <typename KeyT, typename OffsetT>
void keys(nvbench::state& state, nvbench::type_list<KeyT, OffsetT>)
{
  using key_t            = KeyT;
  using value_t          = cub::NullType;
  using key_input_it_t   = key_t*;
  using value_input_it_t = value_t*;
  using key_it_t         = key_t*;
  using value_it_t       = value_t*;
  using offset_t         = OffsetT;
  using compare_op_t     = less_t;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<key_t>;
  using dispatch_t = cub::cub::detail::merge::
    dispatch_t<key_it_t, value_it_t, key_it_t, value_it_t, key_it_t, value_it_t, offset_t, compare_op_t, policy_t>;
#else // TUNE_BASE
  using dispatch_t = cub::detail::merge::
    dispatch_t<key_it_t, value_it_t, key_it_t, value_it_t, key_it_t, value_it_t, offset_t, compare_op_t>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  const auto num_items_lhs  = elements / 2;
  const auto num_items_rhs  = elements - num_items_lhs;
  auto [keys_lhs, keys_rhs] = generate_lhs_rhs<KeyT>(num_items_lhs, num_items_rhs, entropy);

  thrust::device_vector<KeyT> keys_out(elements);
  key_t* d_keys_lhs = thrust::raw_pointer_cast(keys_lhs.data());
  key_t* d_keys_rhs = thrust::raw_pointer_cast(keys_rhs.data());
  key_t* d_keys_out = thrust::raw_pointer_cast(keys_out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_writes<KeyT>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::dispatch(
    nullptr,
    temp_size,
    d_keys_lhs,
    nullptr,
    num_items_lhs,
    d_keys_rhs,
    nullptr,
    num_items_rhs,
    d_keys_out,
    nullptr,
    compare_op_t{},
    cudaStream_t{});

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::dispatch(
      temp_storage,
      temp_size,
      d_keys_lhs,
      nullptr,
      num_items_lhs,
      d_keys_rhs,
      nullptr,
      num_items_rhs,
      d_keys_out,
      nullptr,
      compare_op_t{},
      launch.get_stream());
  });
}

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = fundamental_types;
#endif // TUNE_KeyT

NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(key_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
