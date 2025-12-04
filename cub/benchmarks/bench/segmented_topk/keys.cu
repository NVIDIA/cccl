// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_segmented_topk.cuh>

#include <cuda/iterator>

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

    using topk_policy_t =
      cub::detail::topk::AgentTopKPolicy<TUNE_THREADS_PER_BLOCK,
                                         items_per_thread,
                                         bits_per_pass,
                                         TUNE_LOAD_ALGORITHM,
                                         cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename KeyT, typename OffsetT, typename OutOffsetT>
void topk_keys(nvbench::state& state, nvbench::type_list<KeyT, OffsetT, OutOffsetT>)
{
  // A guarantee that segment size and k fit in shared memory
  constexpr auto min_segment_size    = 1;
  constexpr auto max_segment_size    = 44 * 1024 / sizeof(KeyT);
  constexpr auto min_k               = 1;
  constexpr auto max_k               = max_segment_size;
  constexpr auto min_num_total_items = 1;
  constexpr auto max_num_total_items = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();

  using key_input_it_t  = cuda::strided_iterator<cuda::counting_iterator<const KeyT*>>;
  using key_output_it_t = cuda::strided_iterator<cuda::counting_iterator<KeyT*>>;

  // Statically constrains segment size to what can fit in shared memory
  using seg_size_t = cub::detail::topk::segment_size_uniform<min_segment_size, max_segment_size>;

  // Statically constrains k to the maximum segment size
  using k_value_t = cub::detail::topk::k_uniform<min_k, max_k>;

  using select_direction_value_t = cub::detail::topk::select_direction_static<cub::detail::topk::select::max>;
  using total_num_items_guarantee_t =
    cub::detail::topk::total_num_items_guarantee<min_num_total_items, max_num_total_items>;
  using offset_t     = ::cuda::std::int32_t;
  using out_offset_t = ::cuda::std::int32_t;

  using dispatch_t = cub::detail::topk::DispatchSegmentedTopK<
    key_input_it_t,
    key_output_it_t,
    cub::NullType**,
    cub::NullType**,
    seg_size_t,
    k_value_t,
    select_direction_value_t,
    total_num_items_guarantee_t,
    offset_t,
    out_offset_t,
    cub::detail::topk::select::max
#if !TUNE_BASE
    ,
    policy_hub_t<KeyT, OffsetT>
#endif // !TUNE_BASE
    >;

  // Retrieve axis parameters
  const auto max_elements      = static_cast<size_t>(state.get_int64("Elements{io}"));
  const auto segment_size      = static_cast<size_t>(state.get_int64("SegmentSize"));
  const auto selected_elements = static_cast<size_t>(state.get_int64("SelectedElements"));
  const auto num_segments      = ::cuda::std::max<std::size_t>(1, (max_elements / segment_size));
  const auto elements          = num_segments * segment_size;
  const auto total_num_items   = total_num_items_guarantee_t{static_cast<::cuda::std::int64_t>(elements)};
  const bit_entropy entropy    = str_to_entropy(state.get_string("Entropy"));

  // Skip workloads where k exceeds the segment size
  if (selected_elements >= elements)
  {
    state.skip("We only support the case where the variable SelectedElements is smaller than the variable "
               "Elements{io}.");
    return;
  }

  // Skip workloads where the segment size exceeds shared memory capacity
  if (segment_size > max_segment_size)
  {
    state.skip("The specified SegmentSize exceeds the maximum segment size that can fit in shared memory for the given "
               "KeyT.");
    return;
  }

  thrust::device_vector<KeyT> in_keys_buffer = generate(elements, entropy);
  thrust::device_vector<KeyT> out_keys_buffer(selected_elements * num_segments, thrust::no_init);
  key_input_it_t d_keys_in_ptr   = thrust::raw_pointer_cast(in_keys_buffer.data());
  key_output_it_t d_keys_out_ptr = thrust::raw_pointer_cast(out_keys_buffer.data());
  auto d_keys_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), selected_elements);

  auto segment_sizes     = seg_size_t{static_cast<::cuda::std::int64_t>(segment_size)};
  auto k                 = k_value_t{static_cast<::cuda::std::int64_t>(selected_elements)};
  auto select_directions = select_direction_value_t{};

  state.add_element_count(elements, "NumElements");
  state.add_element_count(segment_size, "SegmentSize");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(selected_elements * num_segments, "OutputKeys");

  // allocate temporary storage
  size_t temp_size;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys_in,
    d_keys_out,
    static_cast<cub::NullType**>(nullptr),
    static_cast<cub::NullType**>(nullptr),
    segment_sizes,
    k,
    select_directions,
    num_segments,
    0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  // run the algorithm
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes,
      k,
      select_directions,
      num_segments,
      launch.get_stream());
  });
}

using key_type_list          = nvbench::type_list<float>;
using segment_size_type_list = nvbench::type_list<uint32_t>;
using out_offset_type_list   = nvbench::type_list<uint32_t>;

// seq_lens = [seq_len] if seq_len > 0 else [2049, 4096, 8192, 16384, 32768, 65536, 131072, 256 * 1024, 512 * 1024]
// (i.e., 2**11 to 2**19)

NVBENCH_BENCH_TYPES(fixed_seg_size_topk_keys, NVBENCH_TYPE_AXES(key_type_list, segment_size_type_list, out_offset_type_list))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "SegmentSize{ct}", "OutOffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(28, 28, 4))
  .add_int64_power_of_two_axis("SegmentSize", nvbench::range(10, 14, 1)
  .add_int64_power_of_two_axis("SelectedElements", nvbench::range(3, 23, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201", "0.000"});
