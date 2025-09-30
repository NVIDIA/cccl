// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_topk.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_BLOCK_LOAD_ALGORITHM ld 0:2:1

#if TUNE_BLOCK_LOAD_ALGORITHM == 0
#  define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#elif TUNE_BLOCK_LOAD_ALGORITHM == 1
#  define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#elif TUNE_BLOCK_LOAD_ALGORITHM == 2
#  define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_VECTORIZE
#endif

#if !TUNE_BASE
template <class KeyInT, class OffsetT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int nominal_4b_items_per_thread = TUNE_ITEMS_PER_THREAD;

    static constexpr int items_per_thread = cuda::std::max(1, (nominal_4b_items_per_thread * 4 / sizeof(KeyInT)));

    static constexpr int bits_per_pass = cub::detail::topk::calc_bits_per_pass<KeyInT>();
    using TopKPolicyT =
      cub::detail::topk::AgentTopKPolicy<TUNE_THREADS_PER_BLOCK,
                                         items_per_thread,
                                         bits_per_pass,
                                         cub::BLOCK_LOAD_VECTORIZE,
                                         cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT, typename OutOffsetT>
void topk_pairs(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT, OutOffsetT>)
{
  using key_input_it_t    = const KeyT*;
  using key_output_it_t   = KeyT*;
  using value_input_it_t  = const ValueT*;
  using value_output_it_t = ValueT*;
  using offset_t          = cub::detail::choose_offset_t<OffsetT>;
  using out_offset_t =
    cuda::std::conditional_t<sizeof(offset_t) < sizeof(cub::detail::choose_offset_t<OutOffsetT>),
                             offset_t,
                             cub::detail::choose_offset_t<OutOffsetT>>;
  using dispatch_t = cub::detail::topk::DispatchTopK<
    key_input_it_t,
    key_output_it_t,
    value_input_it_t,
    value_output_it_t,
    offset_t,
    out_offset_t,
    cub::detail::topk::select::max
#if !TUNE_BASE
    ,
    policy_hub_t<KeyT, OffsetT>
#endif // !TUNE_BASE
    >;

  // Retrieve axis parameters
  const auto elements          = static_cast<size_t>(state.get_int64("Elements{io}"));
  const auto selected_elements = static_cast<size_t>(state.get_int64("SelectedElements"));
  const bit_entropy entropy    = str_to_entropy(state.get_string("Entropy"));

  // Skip benchmarks at runtime
  if (selected_elements >= elements)
  {
    state.skip("We only support the case where the variable SelectedElements is smaller than the variable "
               "Elements{io}.");
    return;
  }

  thrust::device_vector<KeyT> in_keys     = generate(elements, entropy);
  thrust::device_vector<ValueT> in_values = generate(elements);
  thrust::device_vector<KeyT> out_keys(selected_elements, thrust::no_init);
  thrust::device_vector<ValueT> out_values(selected_elements, thrust::no_init);

  key_input_it_t d_keys_in       = thrust::raw_pointer_cast(in_keys.data());
  key_output_it_t d_keys_out     = thrust::raw_pointer_cast(out_keys.data());
  value_input_it_t d_values_in   = thrust::raw_pointer_cast(in_values.data());
  value_output_it_t d_values_out = thrust::raw_pointer_cast(out_values.data());

  state.add_element_count(elements, "NumElements");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_reads<ValueT>(elements, "InputValues");
  state.add_global_memory_writes<KeyT>(selected_elements, "OutputKeys");
  state.add_global_memory_writes<ValueT>(selected_elements, "OutputVales");

  // allocate temporary storage
  size_t temp_size;
  dispatch_t::Dispatch(
    nullptr, temp_size, d_keys_in, d_keys_out, d_values_in, d_values_out, elements, selected_elements, 0);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  // run the algorithm
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      elements,
      selected_elements,
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(topk_pairs, NVBENCH_TYPE_AXES(integral_types, integral_types, offset_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}", "OutOffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SelectedElements", nvbench::range(3, 23, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201", "0.000"});
