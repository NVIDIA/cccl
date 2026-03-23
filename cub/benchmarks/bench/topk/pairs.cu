// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_topk.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_BLOCK_LOAD_ALGORITHM ld 0:2:1

#if !TUNE_BASE
template <class KeyInT, class OffsetT>
struct policy_selector_t
{
  _CCCL_HOST_DEVICE constexpr auto operator()(::cuda::arch_id) const -> cub::detail::topk::topk_policy
  {
#  if TUNE_BLOCK_LOAD_ALGORITHM == 0
    constexpr auto load_alg = cub::BLOCK_LOAD_DIRECT;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 1
    constexpr auto load_alg = cub::BLOCK_LOAD_WARP_TRANSPOSE;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 2
    constexpr auto load_alg = cub::BLOCK_LOAD_VECTORIZE;
#  endif

    constexpr int nominal_4b_items_per_thread = TUNE_ITEMS_PER_THREAD;
    constexpr int items_per_thread            = cuda::std::max(1, (nominal_4b_items_per_thread * 4 / sizeof(KeyInT)));
    return cub::detail::topk::topk_policy{
      TUNE_THREADS_PER_BLOCK,
      items_per_thread,
      cub::detail::topk::calc_bits_per_pass<KeyInT>(),
      load_alg,
      cub::BLOCK_SCAN_WARP_SCANS};
  }
};
#endif // !TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT, typename OutOffsetT>
void topk_pairs(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT, OutOffsetT>)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;
  using out_offset_t =
    cuda::std::conditional_t<sizeof(offset_t) < sizeof(cub::detail::choose_offset_t<OutOffsetT>),
                             offset_t,
                             cub::detail::choose_offset_t<OutOffsetT>>;
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

  const KeyT* d_keys_in     = thrust::raw_pointer_cast(in_keys.data());
  KeyT* d_keys_out          = thrust::raw_pointer_cast(out_keys.data());
  const ValueT* d_values_in = thrust::raw_pointer_cast(in_values.data());
  ValueT* d_values_out      = thrust::raw_pointer_cast(out_values.data());

  state.add_element_count(elements, "NumElements");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_reads<ValueT>(elements, "InputValues");
  state.add_global_memory_writes<KeyT>(selected_elements, "OutputKeys");
  state.add_global_memory_writes<ValueT>(selected_elements, "OutputVales");

  // allocate temporary storage
  size_t temp_size;
  cub::detail::topk::dispatch<cub::detail::topk::select::max>(
    nullptr,
    temp_size,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    static_cast<offset_t>(elements),
    static_cast<out_offset_t>(selected_elements),
    cub::detail::identity_decomposer_t{},
    0
#if !TUNE_BASE
    ,
    policy_selector_t<KeyT, OffsetT>{}
#endif // !TUNE_BASE
  );
  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  // run the algorithm
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::topk::dispatch<cub::detail::topk::select::max>(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      static_cast<offset_t>(elements),
      static_cast<out_offset_t>(selected_elements),
      cub::detail::identity_decomposer_t{},
      launch.get_stream()
#if !TUNE_BASE
        ,
      policy_selector_t<KeyT, OffsetT>{}
#endif // !TUNE_BASE
    );
  });
}

NVBENCH_BENCH_TYPES(topk_pairs, NVBENCH_TYPE_AXES(integral_types, integral_types, offset_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}", "OutOffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SelectedElements", nvbench::range(3, 23, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201", "0.000"});
