// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/dispatch/dispatch_three_way_partition.cuh>
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
template <typename InputT>
struct tuned_policy_selector_t
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id) const
    -> cub::detail::three_way_partition::three_way_partition_policy
  {
    return {TUNE_THREADS_PER_BLOCK,
            TUNE_ITEMS_PER_THREAD,
            TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE,
            cub::LOAD_DEFAULT,
            cub::BLOCK_SCAN_WARP_SCANS,
            cub::detail::delay_constructor_policy_from_type<delay_constructor_t>};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void partition(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t        = const T*;
  using output_it_t       = T*;
  using num_selected_it_t = OffsetT*;
  using select_op_t       = less_then_t<T>;
  using offset_t          = OffsetT;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  T min_val{};
  T max_val = ::cuda::std::numeric_limits<T>::max();

  T left_border  = max_val / 3;
  T right_border = left_border * 2;

  select_op_t select_op_1{left_border};
  select_op_t select_op_2{right_border};

  thrust::device_vector<T> in = generate(elements, entropy, min_val, max_val);
  thrust::device_vector<offset_t> num_selected(1);
  thrust::device_vector<T> out_1(elements);
  thrust::device_vector<T> out_2(elements);
  thrust::device_vector<T> out_3(elements);

  input_it_t d_in                  = thrust::raw_pointer_cast(in.data());
  output_it_t d_out_1              = thrust::raw_pointer_cast(out_1.data());
  output_it_t d_out_2              = thrust::raw_pointer_cast(out_2.data());
  output_it_t d_out_3              = thrust::raw_pointer_cast(out_3.data());
  num_selected_it_t d_num_selected = thrust::raw_pointer_cast(num_selected.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<offset_t>(1);

  std::size_t temp_size{};
  auto dispatch = [&](void* temp_storage, cudaStream_t stream) {
    return cub::detail::three_way_partition::dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_out_1,
      d_out_2,
      d_out_3,
      d_num_selected,
      select_op_1,
      select_op_2,
      static_cast<offset_t>(elements),
      stream
#if !TUNE_BASE
      ,
      policy_selector_t{}
#endif // !TUNE_BASE
    );
  };

  dispatch(nullptr, nullptr);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch(temp_storage, launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(partition, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
